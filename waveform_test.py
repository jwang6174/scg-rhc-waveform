import matplotlib.pyplot as plt
import os
import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from waveform_train import AttentionUNetGenerator, SCGDataset

with open('waveform_loader_test.pickle', 'rb') as f:
  test_loader = pickle.load(f)

checkpoint = torch.load('waveform_checkpoint.pth', weights_only=False)
generator = AttentionUNetGenerator(
  in_channels=checkpoint['in_channels'], 
  out_channels=checkpoint['out_channels'])
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

for i, (scg, real_rhc) in enumerate(test_loader, start=1):
  pred_rhc = generator(scg)[0, 0, :]
  plt.plot(pred_rhc.detach().numpy(), label='Pred RHC')
  plt.plot(real_rhc[0, 0, :], label='Real RHC')
  plt.xlabel('Sample')
  plt.ylabel('mmHg')
  plt.legend()
  plt.savefig(f'waveform_test_plot_{i}.png')
  plt.close()
  if i == 3:
    break
