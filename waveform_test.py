import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from paramutil import Params
from recordutil import PROCESSED_DATA_PATH
from scipy.spatial.distance import euclidean
from torch.utils.data import DataLoader
from waveform_train import Generator, SCGDataset


def save_random_pred_plots(dirpath, generator, loader, num_plots):
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
    plt.savefig(os.path.join(dirpath, f'random_pred_plot_{timestamp}_{i}.png'))
    plt.close()
    if i == num_plots:
      break


def save_top_pred_plots(params, generator, sorted_comparisons, num_plots):
  """
  Save most similar predicted RHC plots.
  """
  for i, comparison in enumerate(sorted_comparisons, start=1):
    dtw = comparison['dtw']
    filename = comparison['filename']
    start_idx = comparison['start_idx']
    stop_idx = comparison['stop_idx']
    real_rhc = comparison['real_rhc']
    pred_rhc = comparison['pred_rhc']
    plt.plot(pred_rhc, label='Pred RHC')
    plt.plot(real_rhc, label='Real RHC')
    plt.title(f'DTW: {dtw:.2f}')
    plt.xlabel('Sample')
    plt.ylabel('mmHg')
    plt.legend()
    plot_name = f'top_pred_plot_{i}_{filename}_{start_idx}-{stop_idx}'
    plot_path = os.path.join(params.pred_RHC_path, plot_name)
    plt.savefig(plot_path)
    plt.close()


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
      'dtw': fastdtw(np.array([x]), np.array([y]), dist=euclidean)[0]
    }
    comparisons.append(comparison)

  for comparison in comparisons:
    filepath = os.path.join(PROCESSED_DATA_PATH, f"{comparison['filename']}.json")
    with open(filepath, 'r') as f:
      comparison.update(json.load(f))

  return comparisons


def run(params):
  """
  Run tests.
  """ 
  with open(params.train_path, 'rb') as f:
    train_loader = pickle.load(f)

  with open(params.test_path, 'rb') as f:
    test_loader = pickle.load(f)

  checkpoint = torch.load(params.checkpoint_path, weights_only=False)
  generator = Generator(len(params.in_channels))
  generator.load_state_dict(checkpoint['g_state_dict'])
  generator.eval()

  # save_random_pred_plots(params.dir_path, generator, train_loader, num_plots=5)
  # save_random_pred_plots(params.dir_path, generator, test_loader, num_plots=5)
  
  comparisons = get_waveform_comparisons(generator, test_loader)
  comparisons.sort(key=lambda x: x['dtw'])

  comparisons_df = pd.DataFrame(comparisons)
  comparisons_df.to_csv(params.comparisons_path, index=False)

  save_top_pred_plots(params, generator, comparisons, num_plots=10)


if __name__ == '__main__':
  params = Params('01_waveform/params.json')
  run(params)

