import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from recordutil import get_scg_rhc_segments

class AttentionGate(nn.Module):

  def __init__(self, F_g, F_l, F_int):
    super(AttentionGate, self).__init__()
    
    self.W_g = nn.Sequential(
      nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
      nn.BatchNorm1d(F_int))

    self.W_x = nn.Sequential(
      nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
      nn.BatchNorm1d(F_int))

    self.psi = nn.Sequential(
      nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
      nn.BatchNorm1d(1),
      nn.Sigmoid())

    self.relu = nn.ReLU(inplace=True)

  def forward(self, g, x):
    g1 = self.W_g(g)
    x1 = self.W_x(x)
    psi = self.psi(self.relu(g1 + x1))
    return x * psi

class AttentionUNetGenerator(nn.Module):
  
  def __init__(self, in_channels, out_channels):
    super(AttentionUNetGenerator, self).__init__()
    self.encoder1 = self.conv_block(in_channels, 64)
    self.encoder2 = self.conv_block(64, 128)
    self.encoder3 = self.conv_block(128, 256)
    self.encoder4 = self.conv_block(256, 512)
    self.bottleneck = self.conv_block(512, 1024)
    self.decoder4 = self.conv_block(1024, 512)
    self.decoder3 = self.conv_block(512, 256)
    self.decoder2 = self.conv_block(256, 128)
    self.decoder1 = self.conv_block(128, 64)
    self.attention4 = AttentionGate(512, 512, 256)
    self.attention3 = AttentionGate(256, 256, 128)
    self.attention2 = AttentionGate(128, 128, 64)
    self.up4 = self.up_conv(1024, 512)
    self.up3 = self.up_conv(512, 256)
    self.up2 = self.up_conv(256, 128)
    self.up1 = self.up_conv(128, 64)
    self.final = nn.Conv1d(64, out_channels, kernel_size=1)

  def conv_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(inplace=True))

  def up_conv(self, in_channels, out_channels):
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

  def max_pool(self, x):
    return F.max_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
  
  def pad_size(self, target, source):
    if source.size(2) < target.size(2):
      target = target[..., :source.size(-1)]
    elif source.size(2) > target.size(2):
      target = F.pad(target, (0, source.size(2) - target.size(2)))
    return target

  def forward(self, x):
    e1 = self.encoder1(x)
    e2 = self.encoder2(self.max_pool(e1))
    e3 = self.encoder3(self.max_pool(e2))
    e4 = self.encoder4(self.max_pool(e3))
    
    b = self.bottleneck(self.max_pool(e4))

    d4 = self.up4(b)
    d4 = self.pad_size(d4, e4)
    e4 = self.attention4(d4, e4)
    d4 = self.decoder4(torch.cat((d4, e4), dim=1))

    d3 = self.up3(d4)
    d3 = self.pad_size(d3, e3)
    e3 = self.attention3(d3, e3)
    d3 = self.decoder3(torch.cat((d3, e3), dim=1))

    d2 = self.up2(d3)
    d2 = self.pad_size(d2, e2)
    e2 = self.attention2(d2, e2)
    d2 = self.decoder2(torch.cat((d2, e2), dim=1))

    d1 = self.up1(d2)
    d1 = self.pad_size(d1, e1)
    d1 = self.decoder1(torch.cat((d1, e1), dim=1))
   
    return self.final(d1)

class PatchGANDiscriminator(nn.Module):
  
  def __init__(self, in_channels, condition_channels, ndf, num_layers):
    super(PatchGANDiscriminator, self).__init__()
    
    layers = [
      nn.Conv1d(in_channels + condition_channels, ndf, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.2, inplace=True)
    ]

    for i in range(1, num_layers):
      in_channels = ndf * min(2 ** (i - 1), 8)
      out_channels = ndf * min(2 ** i, 8)
      layers += [
        nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
      ]
      self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

class SCGDataset(Dataset):

  def __init__(self, segments, segment_size):
    self.segments = segments
    self.segment_size = segment_size

  def pad(self, tensor):
    if tensor.shape[-1] < self.segment_size:
        padding = self.segment_size - tensor.shape[-1]
        tensor = torch.nn.functional.pad(tensor, (0, padding))
    elif tensor.shape[-1] > self.segment_size:
        tensor = tensor[:, :, :self.segment_size]
    return tensor

  def minmax_norm(self, tensor):
    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 0.0001)
    return tensor

  def invert(self, tensor):
    return torch.tensor(tensor.T, dtype=torch.float32)

  def __len__(self):
    return len(self.segments)

  def __getitem__(self, index):
    segments = self.segments[index]
    scg = self.pad(self.invert(self.minmax_norm(segments[0])))
    rhc = self.pad(self.invert(self.minmax_norm(segments[1])))
    return scg, rhc

def least_squares(tensor):
  return torch.mean(tensor ** 2)

if __name__ == '__main__':
  scg_channels = ['patch_ACC_lat', 'patch_ACC_hf']
  in_channels = len(scg_channels)
  out_channels = 1
  segment_size = 750
  total_epochs = 50
  alpha = 0.00005
  batch_size = 16
  beta1 = 0.5
  beta2 = 0.999
  n_critic = 2
  criterion = nn.MSELoss()

  all_segments = get_scg_rhc_segments(scg_channels, segment_size)
  train_segments, test_segments = train_test_split(all_segments, train_size=0.9)
  train_set = SCGDataset(train_segments, segment_size)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_set = SCGDataset(test_segments, segment_size)
  test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

  with open('waveform_loader_test.pickle', 'wb') as f:
    pickle.dump(train_loader, f)

  with open('waveform_loader_train.pickle', 'wb') as f:
    pickle.dump(test_loader, f)

  generator = AttentionUNetGenerator(in_channels, out_channels)
  g_optimizer = optim.Adam(generator.parameters(), lr=alpha, betas=(0.5, 0.999))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  generator = generator.to(device)
  criterion = criterion.to(device)

  g_losses = []

  for epoch in range(total_epochs):
    for i, (scg, rhc) in enumerate(train_loader, start=1):
      scg = scg.to(device)
      rhc = rhc.to(device)

      pred_rhc = generator(scg)
      g_loss = criterion(rhc, pred_rhc)
      g_losses.append(g_loss.item())
      g_loss.backward()
      g_optimizer.step()

      if i % 100 == 0 or i == len(train_loader) - 1:
        print(f'Epoch {epoch+1}/{total_epochs} | Batch {i+1}/{len(train_loader)}')
        plt.plot(g_losses, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('waveform_losses.png')
        plt.close()

    checkpoint = {
      'epoch': epoch,
      'scg_channels': scg_channels,
      'in_channels': in_channels,
      'out_channels': out_channels,
      'segment_size': segment_size,
      'generator_state_dict': generator.state_dict()
    }
    torch.save(checkpoint, 'waveform_checkpoint.pth')

