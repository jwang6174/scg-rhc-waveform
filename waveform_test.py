import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from paramutil import Params
from recordutil import PROCESSED_DATA_PATH, SCGDataset
from scipy.stats import pearsonr, t
from sklearn.metrics import mean_squared_error
from time import time
from timelog import timelog
from torch.utils.data import DataLoader
from waveform_train import Generator, get_last_checkpoint_path


def reverse_minmax(tensor, orig_min, orig_max):
  """
  Reverse min-max normalization.
  """
  return tensor * (orig_max - orig_min) + orig_min


def get_pcc(x, y):
  """
  Get pearson correlation coefficient values.
  """
  result = pearsonr(x, y)
  pcc_r = result.statistic
  pcc_p = result.pvalue
  pcc_ci95 = result.confidence_interval(confidence_level=0.95)
  return pcc_r, pcc_ci95.low, pcc_ci95.high


def get_rmse(x, y):
  """
  Get root mean squared error values.
  """
  alpha = 0.05
  n = len(x)
  rmse = np.sqrt(mean_squared_error(x, y))
  se = np.sqrt(rmse / (2 * n))
  t_crit = t.ppf(1 - alpha/2, df=n - 1)
  rmse_ci95_lower = rmse - t_crit * se
  rmse_ci95_upper = rmse + t_crit * se
  return rmse, rmse_ci95_lower, rmse_ci95_upper


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
    min_rhc, max_rhc = segment[6]
    
    x = reverse_minmax(real_rhc.detach().numpy()[0, :], min_rhc, max_rhc)
    y = reverse_minmax(generator(scg).detach().numpy()[0, 0, :], min_rhc, max_rhc)

    pcc_r, pcc_ci95_lower, pcc_ci95_upper = get_pcc(x, y)
    rmse, rmse_ci95_lower, rmse_ci95_upper = get_rmse(x, y)
    
    comparison = {
      'filename': filename,
      'start_idx': int(start_idx),
      'stop_idx': int(stop_idx),
      'real_rhc': str(x.tolist()),
      'pred_rhc': str(y.tolist()),
      'pcc_r': pcc_r,
      'pcc_ci95_lower': pcc_ci95_lower,
      'pcc_ci95_upper': pcc_ci95_upper,
      'rmse': rmse,
      'rmse_ci95_lower': rmse_ci95_lower,
      'rmse_ci95_upper': rmse_ci95_upper,
    }
    comparisons.append(comparison)
  return comparisons


def get_processed_checkpoints(comp_dir_path):
  """
  Returns checkpoints that have already been processed in a given directory.
  """
  return frozenset(f"{filename.split('.')[0]}.checkpoint" for filename in os.listdir(comp_dir_path))


def run(params, loader_type, checkpoint_path, resume):
  """
  Run tests.
  """
  start_time = time()
  checkpoint_message = f"{checkpoint_path if checkpoint_path else 'last checkpoint'}"
  print(timelog(f"Run waveform_test for {params.dir_path} | {loader_type} | {checkpoint_message}", start_time))
  
  # Check loader type
  if loader_type == 'train':
    loader_path = params.train_path
  elif loader_type == 'valid':
    loader_path = params.valid_path
  elif loader_type == 'test':
    loader_path = params.test_path
  else:
    raise Exception('Invalid loader type')

  # Open pickled data loader
  with open(loader_path, 'rb') as f:
    loader = pickle.load(f)

  # Set checkpoint paths either 'all', 'last', or a specific checkpoint
  if checkpoint_path == 'all':
    checkpoint_paths = sorted(os.listdir(params.checkpoint_dir_path))[:params.total_epochs]
  elif checkpoint_path == 'last':
    checkpoint_paths = [get_last_checkpoint_path(params.checkpoint_dir_path)]
  else:
    checkpoint_paths = [checkpoint_path]

  # Make comparison directory path if not exists
  comp_dir_path = os.path.join(params.comparison_dir_path, loader_type)
  if not os.path.exists(comp_dir_path):
    os.makedirs(comp_dir_path)

  # Get prior processed checkpoints or empty list if intend to re-calculate 
  # performance for all checkpoints
  processed_checkpoints = get_processed_checkpoints(comp_dir_path) if resume else []

  # Iterate through each checkpoint, calculate PCC and RMSE, and output 
  # checkpoint with best RMSE
  for i, checkpoint_path in enumerate(checkpoint_paths):

    print(timelog(f'waveform_test | {params.dir_path} | {loader_type} | {checkpoint_message} | {i}/{len(checkpoint_paths)}', start_time))
    if checkpoint_path in processed_checkpoints:
      continue

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


if __name__ == '__main__':
  dir_path = sys.argv[1]
  loader_type = sys.argv[2]
  checkpoint_path = sys.argv[3]
  resume = sys.argv[4] != 'redo'
  params = Params(os.path.join(dir_path, 'params.json'))
  run(params, loader_type, checkpoint_path, resume)

