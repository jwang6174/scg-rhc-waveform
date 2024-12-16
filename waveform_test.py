import matplotlib.pyplot as plt
import os
import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from waveform_train import Generator, SCGDataset

with open('waveform_loader_test.pickle', 'rb') as f:
  test_loader = pickle.load(f)

checkpoint = torch.load('waveform_checkpoint.pth', weights_only=False)
generator = Generator(checkpoint['in_channels'])
generator.load_state_dict(checkpoint['g_state_dict'])
generator.eval()

for i, (scg, real_rhc, filename, start_idx, stop_idx) in enumerate(test_loader, start=1):
  real_rhc = real_rhc.detach().numpy()[0, 0, :]
  pred_rhc = generator(scg).detach().numpy()[0, 0, :]
  plt.plot(pred_rhc, label='Pred RHC')
  plt.plot(real_rhc, label='Real RHC')
  plt.xlabel('Sample')
  plt.ylabel('mmHg')
  plt.legend()
  plt.savefig(f'waveform_test_plot_{i}.png')
  plt.close()
  if i == 3:
    break
