import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from fastdtw import fastdtw
from paramutil import Params
from recordutil import PROCESSED_DATA_PATH, SCGDataset
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from waveform_train import Generator, get_last_checkpoint_path


def save_random_pred_plots(dirpath, generator, loader, prefix, num_plots):
  """
  Save a certain number of predicted vs real RHC plots.
  """
  for i, segment in enumerate(loader, start=1):
    scg = segment[0]
    real_rhc = segment[1]
    timestamp = str(datetime.now().strftime('%Y-%m-%d %H-%M-%S')).replace(' ', '_')
    real_rhc = real_rhc.detach().numpy()[0, 0, :]
    pred_rhc = generator(scg).detach().numpy()[0, 0, :]
    plt.plot(pred_rhc, label='Pred RHC')
    plt.plot(real_rhc, label='Real RHC')
    plt.xlabel('Sample')
    plt.ylabel('mmHg')
    plt.legend()
    plt.savefig(os.path.join(dirpath, f'random_pred_plot_{timestamp}_{prefix}_{i}.png'))
    plt.close()
    if i == num_plots:
      break


def save_top_pred_plots(params, generator, sorted_comparisons, num_plots):
  """
  Save most similar predicted RHC plots.
  """
  for i, comparison in enumerate(sorted_comparisons[:num_plots], start=1):
    pcc = comparison['pcc']
    filename = comparison['filename']
    start_idx = comparison['start_idx']
    stop_idx = comparison['stop_idx']
    real_rhc = comparison['real_rhc']
    pred_rhc = comparison['pred_rhc']
    plt.plot(pred_rhc, label='Pred RHC')
    plt.plot(real_rhc, label='Real RHC')
    plt.title(f'PCC: {pcc:.2f}')
    plt.xlabel('Sample')
    plt.ylabel('mmHg')
    plt.legend()
    plot_name = f'top_pred_plot_{i:03d}_{filename}_{start_idx}-{stop_idx}'
    plot_path = os.path.join(params.pred_top_dir_path, plot_name)
    plt.savefig(plot_path)
    plt.close()


def get_waveform_comparisons(generator, loader):
  """
  Get waveform comparions for real and predicted RHC waveforms.
  """
  comparisons = []
  for segment in loader.dataset:
    scg = segment[0].unsqueeze(0)
    real_rhc = segment[1]
    filename = segment[2]
    start_idx = segment[3]
    stop_idx = segment[4]
    x = real_rhc.detach().numpy()[0, :]
    y = generator(scg).detach().numpy()[0, 0, :]
    comparison = {
      'filename': filename,
      'start_idx': int(start_idx),
      'stop_idx': int(stop_idx),
      'real_rhc': x,
      'pred_rhc': y,
      'pcc': np.corrcoef(x, y)[0, 1],
      'rmse': np.sqrt(np.mean((y - x) ** 2)),
      'mae': mean_absolute_error(x, y)
      }
    comparisons.append(comparison)

  for comparison in comparisons:
    filepath = os.path.join(PROCESSED_DATA_PATH, f"{comparison['filename']}.json")
    with open(filepath, 'r') as f:
      comparison.update(json.load(f))

  return comparisons


def run(params, checkpoint_path=None):
  """
  Run tests.
  """ 
  with open(params.train_path, 'rb') as f:
    train_loader = pickle.load(f)

  with open(params.valid_path, 'rb') as f:
    valid_loader = pickle.load(f)

  with open(params.test_path, 'rb') as f:
    test_loader = pickle.load(f)

  if checkpoint_path is None:
    checkpoint_path = get_last_checkpoint_path(params.checkpoint_dir_path)

  checkpoint = torch.load(os.path.join(params.checkpoint_dir_path, checkpoint_path), weights_only=False)
  generator = Generator(len(params.in_channels))
  generator.load_state_dict(checkpoint['g_state_dict'])
  generator.eval()

  save_random_pred_plots(params.pred_rand_dir_path, generator, train_loader, 'train', num_plots=5)
  
  comparisons = get_waveform_comparisons(generator, valid_loader)
  comparisons.sort(key=lambda x: x['pcc'], reverse=True)
  
  comparisons_df = pd.DataFrame(comparisons)
  comparisons_df.to_csv(params.comparisons_path, index=False)
  
  save_top_pred_plots(params, generator, comparisons, num_plots=100)


if __name__ == '__main__':
  with open('active_project.txt', 'r') as f:
    path = f.readline().strip('\n')
  params = Params(path)
  run(params)

