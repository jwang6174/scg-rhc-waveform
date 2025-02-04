import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from paramutil import Params
from recordutil import PROCESSED_DATA_PATH, SCGDataset
from scipy.stats import pearsonr
from time import time
from timelog import timelog
from torch.utils.data import DataLoader
from waveform_train import Generator, get_last_checkpoint_path


def save_pred_rand_plots(dirpath, generator, loader, prefix, num_plots):
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
    plot_name = f'random_pred_plot_{timestamp}_{prefix}_{i}.png'
    plt.savefig(os.path.join(dirpath, plot_name))
    plt.close()
    if i == num_plots:
      break


def save_pred_top_plots(dirpath, generator, sorted_comparisons):
  """
  Save most similar predicted RHC plots.
  """
  for i, comparison in enumerate(sorted_comparisons, start=1):
    pcc_r = comparison['pcc_r']
    pcc_p = comparison['pcc_p']
    filename = comparison['filename']
    start_idx = comparison['start_idx']
    stop_idx = comparison['stop_idx']
    real_rhc = comparison['real_rhc']
    pred_rhc = comparison['pred_rhc']
    plt.plot(pred_rhc, label='Pred RHC')
    plt.plot(real_rhc, label='Real RHC')
    plt.title(f'{pcc_r:.3f}, {pcc_p:.3f}')
    plt.xlabel('Sample')
    plt.ylabel('mmHg')
    plt.legend()
    plot_name = f'top_pred_plot_{i:03d}_{filename}_{start_idx}-{stop_idx}'
    plot_path = os.path.join(dirpath, plot_name)
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
    pcc_r, pcc_p = pearsonr(x, y)
    comparison = {
      'filename': filename,
      'start_idx': int(start_idx),
      'stop_idx': int(stop_idx),
      'real_rhc': x,
      'pred_rhc': y,
      'pcc_r': pcc_r,
      'pcc_p': pcc_p,
      }
    comparisons.append(comparison)

  for comparison in comparisons:
    filepath = os.path.join(PROCESSED_DATA_PATH, f"{comparison['filename']}.json")
    with open(filepath, 'r') as f:
      comparison.update(json.load(f))

  return comparisons


def run(params, loader_type, checkpoint_path=None):
  """
  Run tests.
  """
  start_time = time()
  print(timelog(f"Running waveform test for {params.dir_path}, {loader_type}, {checkpoint_path if checkpoint_path else 'last checkpoint'}", start_time))
  
  if loader_type == 'train':
    loader_path = params.train_path
  elif loader_type == 'valid':
    loader_path = params.valid_path
  elif loader_type == 'test':
    loader_path = params.test_path
  else:
    raise Exception('Invalid loader type')

  with open(loader_path, 'rb') as f:
    loader = pickle.load(f)

  if checkpoint_path == 'all':
    checkpoint_paths = sorted(os.listdir(params.checkpoint_dir_path))[:params.total_epochs]
  elif checkpoint_path is None:
    checkpoint_paths = [get_last_checkpoint_path(params.checkpoint_dir_path)]
  else:
    checkpoint_paths = [checkpoint_path]
  
  pred_top_dir_path = os.path.join(params.pred_top_dir_path, loader_type)
  if os.path.exists(pred_top_dir_path):
    raise Exception('Directory {pred_top_dir_path} already exists!')
  else:
    os.makedirs(pred_top_dir_path)

  comp_dir_path = os.path.join(params.comparison_dir_path, loader_type)
  if os.path.exists(comp_dir_path):
    raise Exception('Directory {comp_dir_path} already exists!')
  else:
    os.makedirs(comp_dir_path)

  for i, checkpoint_path in enumerate(checkpoint_paths):
    print(timelog(f'{i}/{len(checkpoint_paths)}', start_time))
    checkpoint = torch.load(os.path.join(params.checkpoint_dir_path, checkpoint_path), weights_only=False)
    generator = Generator(len(params.in_channels))
    generator.load_state_dict(checkpoint['g_state_dict'])
    generator.eval()

    comparisons = get_waveform_comparisons(generator, loader)
    comparisons.sort(key=lambda x: x['pcc_r'], reverse=True)
    
    checkpoint_str = checkpoint_path.split('.')[0]
    comparison_path = os.path.join(comp_dir_path, f'{checkpoint_str}.csv')

    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(comparison_path, index=False)
  
    save_pred_top_plots(pred_top_dir_path, generator, comparisons)


if __name__ == '__main__':
  with open('project_active.json', 'r') as f:
    data = json.load(f)
  params_path = data['params_path']
  loader_type = data['loader_type']
  checkpoint_path = data['checkpoint_path']
  params = Params(params_path)
  run(params, loader_type, checkpoint_path=checkpoint_path)

