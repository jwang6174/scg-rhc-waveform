import os
import pickle
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from waveform_train import AttentionUNetGenerator, SCGDataset

# Define results directory.
results_dir = 'waveform_results'

# Define test set path.
test_set_path = os.path.join(results_dir, 'test_set.pickle')

# Define checkpoint path.
checkpoint_path = os.path.join(results_dir, 'checkpoint.pth')

# Define test set.
with open(test_set_path, 'rb') as f:
  test_set = SCGDataset(segments=pickle.load(f), segment_size=375)

# Define test loader.
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# Load checkpoint and model.
checkpoint = torch.load(checkpoint_path, weights_only=False)
generator = AttentionUNetGenerator(in_channels=3, out_channels=1)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Plot predicted and actual RHC waveforms for a sample.
for i, (scg, real_rhc) in enumerate(test_loader):
  pred_rhc = generator(scg)[0, 0, :]
  plt.plot(pred_rhc.detach().numpy())
  plt.plot(real_rhc[0, 0, :])
  plt.xlabel('Frame')
  plt.ylabel('mmHg')
  plt.legend()
  plt.savefig(os.path.join(results_dir, f'test_{i}.png'))
  plt.close()
  if i == 10:
    break
