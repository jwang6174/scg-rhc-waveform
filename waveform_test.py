import json
import matplotlib.pyplot as plt
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
from scipy.stats import pearsonr
from time import time
from timelog import timelog
from torch.utils.data import DataLoader
from waveform_train import Generator, get_last_checkpoint_path


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
    result = pearsonr(x, y)
    pcc_r = result.statistic
    pcc_p = result.pvalue
    pcc_ci95 = result.confidence_interval(confidence_level=0.95)
    comparison = {
      'filename': filename,
      'start_idx': int(start_idx),
      'stop_idx': int(stop_idx),
      'real_rhc': str(x.tolist()),
      'pred_rhc': str(y.tolist()),
      'pcc_r': pcc_r,
      'pcc_p': pcc_p,
      'pcc_ci95_low': pcc_ci95.low,
      'pcc_ci95_high': pcc_ci95.high,
    }
    comparisons.append(comparison)
  return comparisons


def run(params, loader_type, checkpoint_path):
  """
  Run tests.
  """
  start_time = time()
  print(timelog(f"Run waveform_test for {params.dir_path}, {loader_type}, {checkpoint_path if checkpoint_path else 'last checkpoint'}", start_time))
  
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
  elif checkpoint_path == 'last':
    checkpoint_paths = [get_last_checkpoint_path(params.checkpoint_dir_path)]
  else:
    checkpoint_paths = [checkpoint_path]

  comp_dir_path = os.path.join(params.comparison_dir_path, loader_type)
  if os.path.exists(comp_dir_path):
    raise Exception(f'Directory {comp_dir_path} already exists!')
  else:
    os.makedirs(comp_dir_path)

  for i, checkpoint_path in enumerate(checkpoint_paths):
    print(timelog(f'waveform_test | {params.dir_path} | {i}/{len(checkpoint_paths)}', start_time))
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
  params = Params(os.path.join(dir_path, 'params.json'))
  run(params, loader_type, checkpoint_path)
