import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
from datetime import datetime
from fastdtw import fastdtw
from recordutil import PROCESSED_DATA_PATH
from scipy.signal import hilbert, coherence
from scipy.spatial.distance import euclidean, cosine
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from waveform_train import Generator, SCGDataset


def save_pred_plots(generator, loader, num_plots=3):
  """
  Save a certain number of predicted vs real RHC plots.
  """
  for i, (scg, real_rhc, filename, start_idx, stop_idx) in enumerate(loader, start=1):
    timestamp = str(datetime.now().strftime('%Y-%m-%d %H-%M-%S')).replace(' ', '_')
    real_rhc = real_rhc.detach().numpy()[0, 0, :]
    pred_rhc = generator(scg).detach().numpy()[0, 0, :]
    plt.plot(pred_rhc, label='Pred RHC')
    plt.plot(real_rhc, label='Real RHC')
    plt.xlabel('Sample')
    plt.ylabel('mmHg')
    plt.legend()
    plt.savefig(f'waveform_pred_plot_{timestamp}_{i}.png')
    plt.close()
    if i == num_plots:
      break


def get_waveform_comparisons(generator, loader):
  """
  Get waveform comparions for real and predicted RHC waveforms.
  """
  comparisons = []

  for i, (scg, real_rhc, filename, start_idx, stop_idx) in enumerate(loader, start=1):
    filename = filename[0]
    x = real_rhc.detach().numpy()[0, 0, :]
    y = generator(scg).detach().numpy()[0, 0, :]
    comparison = {
      'filename': filename,
      'start_idx': int(start_idx),
      'stop_idx': int(stop_idx),
      'real_rhc': x,
      'pred_rhc': y,
      'DTW': fastdtw(np.array([x]), np.array([y]), dist=euclidean)[0]
    }
    comparisons.append(comparison)

  for comparison in comparisons:
    filepath = os.path.join(PROCESSED_DATA_PATH, f"{comparison['filename']}.json")
    with open(filepath, 'r') as f:
      comparison.update(json.load(f))

  return comparisons


def run(params_path):
  """
  Run tests.
  """

  with open(params_path, 'r') as f:
    params = json.load(f)
  
  with open(params['train_path'], 'rb') as f:
    train_loader = pickle.load(f)

  with open(params['test_path'], 'rb') as f:
    test_loader = pickle.load(f)

  checkpoint = torch.load(params['checkpoint_path'], weights_only=False)
  generator = Generator(len(params['in_channels']))
  generator.load_state_dict(checkpoint['g_state_dict'])
  generator.eval()

  # save_pred_plots(generator, train_loader, num_plots=5)
  # save_pred_plots(generator, test_loader, num_plots=5)
  
  comparisons = get_waveform_comparisons(generator, test_loader)
  comparisons_df = pd.DataFrame(comparisons)
  comparisons_df.to_csv(params['comparisons_path'], index=False)


if __name__ == '__main__':
  params_path = '01_waveform_params.json'
  run(params_path)

