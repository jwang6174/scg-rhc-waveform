import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from paramutil import Params
from recordutil import load_dataloader, SCGDataset
from time import time
from timelog import timelog

class AttentionBlock(nn.Module):
  """
  Let x be the activation map of a chosen layer l, g be the global feature
  vector from the previous layer, which provides information to attention
  module to disambiguate task-irrelevant feature content in x. Then the query
  vector q can be obtained from the following formula:
  
  q = psi(sigma_1(W_x(x) + W_g(g) + b_xg)) + b_psi

  where psi is a linear function, W_x and W_g represent linear transformations
  of of x and g, respectively, b_psi and b_xg are their bias terms. The linear
  transformations are computed using 1x1 convolutions. Sigma_1 is the ReLU
  activation function. The output is squeezed in the channel domain through
  psi to produce a spatial attention weight map. The attention coefficient 
  alpha is then obtained by sigma_2, the sigmoid activation function.

  alpha = sigma_2(q)

  The final output via the attention module is 

  x_hat = alpha * x

  Additive attention is used to obtain the attention coefficient. Although more
  computationally expensive, additive attention has experimentally demonstrated
  higher accuracy than multiplicative attention.
  """
  def __init__(self, F_x, F_g, F_int):
    super(AttentionBlock, self).__init__() 

    self.W_x = nn.Sequential(
      nn.Conv1d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
      nn.InstanceNorm1d(F_int))

    self.W_g = nn.Sequential(
      nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
      nn.InstanceNorm1d(F_int))

    self.psi = nn.Sequential(
      nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
      nn.InstanceNorm1d(1),
      nn.Sigmoid())

    self.relu = nn.ReLU(inplace=True)

  def forward(self, g, x):
    g1 = self.W_g(g)
    x1 = self.W_x(x)
    psi = self.psi(self.relu(g1 + x1))
    return x * psi
    

class Generator(nn.Module):
  """
  Unlike traditional GAN, where the generator receives noise z from noise prior
  Pg(z) as input, the proposed framework feeds the generator SCG signals
  corresponding to the expected PAP signals output. This allows controlling the
  characteristics of the reconstructed PAP signal generated.

  To enable effective end-to-end training with limited data, the generator
  architecture utilizes a U-net structure. The left half of the model acts as
  an encoder, extracting features to better characterize the dynamics of the
  SCG signal. The right half forms the decoder, tasked with reconstructing the
  corresponding PAP signal. Through upsampling layers and skip connections, the
  decoder progressively recovers finger details and the original spatial
  dimensions of the PAP signal.

  Considering the physiological characteristics and sampling rate constraints
  of the SCG and PAP signals, the convolution layers of the encoder and
  decoder in the generator use kernels with width 3 and stride 1. Additionally,
  the dropout layers are added to both the encoder and decoder sections to
  reduce overfitting.

  We introduce the attention module in the generator. Attention is a mechanism
  which mimics cognitive attention. Trainable attention mechanisms can be
  categorized into hard attention and soft attention. Hard attention selects a
  single element from the input sequence at each decoding step. It involves
  sampling from the attention distribution, introducing non-differentiability
  and making gradient computation more challenging.

  While the output of the soft attention mechanism can be obtained by computing
  the weighted average of a sequence of vectors based on the attention
  distribution, which can be trained directly using standard backpropagation.
  Thus, a soft attention module is designed for the skip-connection part of the
  generator.
  """
  def __init__(self, in_channels):
    super(Generator, self).__init__()
    self.enc1 = self.conv_block(in_channels, 64)
    self.enc2 = self.conv_block(64, 128)
    self.enc3 = self.conv_block(128, 256)
    self.bottleneck = self.conv_block(256, 512)
    self.dec3 = self.conv_block(512, 256)
    self.dec2 = self.conv_block(256, 128)
    self.dec1 = self.conv_block(128, 64)
    self.att3 = AttentionBlock(256, 256, 128)
    self.att2 = AttentionBlock(128, 128, 64)
    self.att1 = AttentionBlock(64, 64, 32)
    self.up3 = self.upsample(512, 256)
    self.up2 = self.upsample(256, 128)
    self.up1 = self.upsample(128, 64)
    self.final = nn.Conv1d(64, 1, kernel_size=1)
    self.dropout = nn.Dropout(0.3)

  def conv_block(self, in_channels, out_channels):
    """
    Conv1d performs a 1D convolutional operation. It first slides a window
    (also called a kernel) over the input data, which is typically a sequence
    or a signal. At each position, the filter performs a dot product with the
    corresponding elements of the input, extracting features from the local
    region. The result of these dot products is a new sequence, where each
    element represents the filtered response at that position.

    BatchNorm1d refers to a Batch Normalization operation applied to a 1D input,
    meaning it normalizes the values within each data point across a batch of
    data, essentially scaling the values to have a mean of 0 and standard
    deviation f 1, which can help improve the training process of a neural
    network by stabilizing gradients and accelerating convergence.
    """
    return nn.Sequential(
      nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
      nn.InstanceNorm1d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
      nn.InstanceNorm1d(out_channels),
      nn.ReLU(inplace=True)
    )

  def upsample(self, in_channels, out_channels):
    """
    ConvTranspose1d performs a 1D transposed convolution, also known as 
    deconvolution or fractionally-strided convolution. It increases the spatial
    dimension of the input, effectively 'upsampling' it. Like regular
    convolution, ConvTranspose1d learns a set of filters (kernels) that are
    applied to the input. It applies these filters to the input, but in a way
    that reverses the effect of a regular convolution, resulting in an
    upsampled output.
    """
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1)
  
  def max_pool(self, x):
    """
    MaxPool1d performs 1D max pooling over an input signal composed of several
    input planes. It slides a window of a specified size (kernel_size) over the
    input tensor, and for each window, it outputs the maximum value. This
    operation is useful for downsampling the input, reduce its dimensionality,
    and extracting the most prominent features.
    """
    return F.max_pool1d(x, kernel_size=3, stride=1, ceil_mode=True)

  def pad_size(self, A, B):
    """
    Modify tensor A to be the same size as tensor B. 
    """
    if A.size(2) > B.size(2):
      A = A[..., :B.size(-1)]
    elif A.size(2) < B.size(2):
      A = F.pad(B, (0, B.size(2) - A.size(2)))
    return A

  def forward(self, x):
    e1 = self.enc1(x)
    e2 = self.enc2(self.dropout(self.max_pool(e1)))
    e3 = self.enc3(self.dropout(self.max_pool(e2)))
    b = self.bottleneck(self.dropout(self.max_pool(e3)))
    
    d3 = self.pad_size(self.dropout(self.up3(b)), e3)
    a3 = self.att3(d3, e3)
    d3 = self.dec3(torch.cat((d3, a3), dim=1))

    d2 = self.pad_size(self.dropout(self.up2(d3)), e2)
    a2 = self.att2(d2, e2)
    d2 = self.dec2(torch.cat((d2, a2), dim=1))

    d1 = self.pad_size(self.dropout(self.up1(d2)), e1)
    a1 = self.att1(d1, e1)
    d1 = self.dec1(torch.cat((d1, a1), dim=1))

    f = self.final(d1)
    f = self.pad_size(f, x)
    return f


class Discriminator(nn.Module):
  """
  The discriminator differentiates between the reconstructed PAP waveforms and
  the original PAP waveforms. The sigmoid activation function is often used at
  the output layer, producing a probability value between 0 and 1.

  However, averaging the entire signal fails to capture local variations and
  details. To address this limitation, we implemented a PatchGAN discriminator
  that penalizes the structure only at the scale of the patch. The receptive
  field, which refers to the association between an output feature and an
  input region, is determined through

  r0 = Sigma((k - 1) * Pi(s)) + 1

  Where r is the receptive field size at the input layer, k is the kernel size
  at the i-th layer, and s is the stride at the i-th layer. The proposed
  discriminator consists entirely of 3 discriminator blocks and 1 convolutional
  layer.

  The coarse scale discriminator block has the widest receptive field,
  providing a global view of the waveform. In contrast, the fine-scale block
  focuses on guiding the generator to produce localized details. Finally,
  the discriminator makes a judgement on each Nx1 patch extracted from the 
  PAP waveform. This approach produces high-quality results even when the
  patch size is much smaller than the signal length.
  """
  def __init__(self, in_channels, condition_channels=1, ndf=64):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
        nn.Conv1d(in_channels + condition_channels, ndf, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv1d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm1d(ndf * 2),
        nn.ReLU(inplace=True),

        nn.Conv1d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm1d(ndf * 4),
        nn.ReLU(inplace=True),

        nn.Conv1d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm1d(ndf * 8),
        nn.ReLU(inplace=True),

        nn.Conv1d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)
    )
   
  def forward(self, x):
    return self.model(x) 


def compute_gp(discriminator, scg, real_rhc, pred_rhc):
  """
  Penalize the norm of gradient of the discriminator with respect to its input
  to avoid parameter binarization. Used as an alternative to weight clipping to
  satisfy the Lipschitz continuity constraint of the discriminator.
  """
  batch_size = real_rhc.size(0)
  
  # Random weight term for interprolation between real and predicted output.
  # Interpolation is a technique to estimate the values of unknown data points
  # that fall in between existing, known data points.
  eps = torch.rand(batch_size, 1, 1).to(real_rhc.device)

  # Get random interpolation between real and fake samples.
  interpolated = (eps * real_rhc + ((1 - eps) * pred_rhc))
  interpolated.requires_grad_(True)

  # Forward pass through the discriminator.
  scores = discriminator(torch.cat((scg, interpolated), dim=1))
  
  # Predictive RHC label tensor.
  pred = torch.ones_like(scores).to(real_rhc.device)

  # Compute gradients of the discriminator with respect to interpolation.
  gradients = autograd.grad(
    outputs=scores,
    inputs=interpolated,
    grad_outputs=pred,
    create_graph=True,
    retain_graph=True,
    only_inputs=True
  )[0]

  # Compute the gradient norm.
  gradients = gradients.view(gradients.size(0), -1)
  gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

  return gp


def get_last_checkpoint_path(dirpath):
  """
  Get last checkpoint path or None if starting new.
  """
  try:
    return sorted(os.listdir(dirpath), reverse=True)[0]
  except:
    return None


def run(params):
  """
  Run waveform training from given checkpoint, if any, and parameters.
  """
  in_channels = len(params.in_channels)
  alpha = params.alpha
  beta1 = params.beta1
  beta2 = params.beta2
  n_critic = params.n_critic
  lambda_gp = params.lambda_gp
  lambda_aux = params.lambda_aux
  total_epochs = params.total_epochs
  train_path = params.train_path
  dir_path = params.dir_path
  checkpoint_dir_path = params.checkpoint_dir_path
  print(timelog('Loaded params', time()))

  if not os.path.exists(checkpoint_dir_path):
    os.makedirs(checkpoint_dir_path)

  train_loader = load_dataloader(train_path)
  print(timelog('Loaded training set', time()))
  
  generator = Generator(in_channels)
  discriminator = Discriminator(in_channels)
  g_optimizer = optim.Adam(generator.parameters(), lr=alpha, betas=(beta1, beta2))
  d_optimizer = optim.Adam(discriminator.parameters(), lr=alpha, betas=(beta1, beta2))
  aux_loss = nn.MSELoss()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  generator = generator.to(device)
  discriminator = discriminator.to(device)
  aux_loss = aux_loss.to(device)

  last_checkpoint_path = get_last_checkpoint_path(checkpoint_dir_path)

  if last_checkpoint_path is not None:
    print(timelog(f'Loaded prior checkpoint: {last_checkpoint_path}', time()))
    checkpoint = torch.load(os.path.join(checkpoint_dir_path, last_checkpoint_path), weights_only=False)
    start_time = checkpoint['start_time']
    epoch = checkpoint['epoch'] + 1
    g_losses = checkpoint['g_losses']
    d_losses = checkpoint['d_losses']
    generator.load_state_dict(checkpoint['g_state_dict'])
    discriminator.load_state_dict(checkpoint['d_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
  else:
    print(timelog(f'Started new checkpoint', time()))
    start_time = time()
    epoch = 0
    g_losses = []
    d_losses = []

  g_loss_total = sum(g_losses)
  d_loss_total = sum(d_losses)

  while epoch < total_epochs:
    for i, segment in enumerate(train_loader):
      scg = segment[0]
      rhc = segment[1]

      scg = scg.to(device)
      rhc = rhc.to(device)
      
      for _ in range(n_critic):
        pred_rhc = generator(scg)
        pred_validity = discriminator(torch.cat((scg, pred_rhc), dim=1)) 
        real_validity = discriminator(torch.cat((scg, rhc), dim=1))
        gp = compute_gp(discriminator, scg, rhc, pred_rhc)
        d_loss = -torch.mean(real_validity) + torch.mean(pred_validity) + (lambda_gp * gp)
        d_losses.append(d_loss.item())
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

      pred_rhc = generator(scg)
      pred_validity = discriminator(torch.cat((scg, rhc), dim=1))
      g_loss = -torch.mean(pred_validity) + (lambda_aux * aux_loss(pred_rhc, rhc))
      g_losses.append(g_loss.item())
      g_optimizer.zero_grad()
      g_loss.backward()
      g_optimizer.step()

      if i > 0 and (i % 10 == 0 or i == len(train_loader) - 1):
        g_loss_sum = sum(g_losses)
        d_loss_sum = sum(d_losses)
        print(timelog(f'Epoch {epoch}/{total_epochs} | Batch {i}/{len(train_loader)}', start_time))
        print(f'  G Loss Diff: {g_loss_sum - g_loss_total}')
        print(f'  D Loss Diff: {d_loss_sum - d_loss_total}')
        g_loss_total = g_loss_sum
        d_loss_total = d_loss_sum
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.title(f'Epoch {epoch}/{total_epochs} | Batch {i}/{len(train_loader)}')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(os.path.join(dir_path, 'train_losses.png'))
        plt.close()
      
    checkpoint = {
      'start_time': start_time,
      'epoch': epoch,
      'g_losses': g_losses,
      'd_losses': d_losses,
      'g_state_dict': generator.state_dict(),
      'd_state_dict': discriminator.state_dict(),
      'g_optimizer_state_dict': g_optimizer.state_dict(),
      'd_optimizer_state_dict': d_optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir_path, f'{epoch:03d}.checkpoint'))

    epoch += 1

if __name__ == '__main__':
  with open('project_active.json', 'r') as f:
    data = json.load(f)
  path = data['params_path']
  print(timelog(f'Starting waveform training with {path}', time()))
  run(Params(path))
