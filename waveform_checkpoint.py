import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from paramutil import Params
from time import time
from timelog import timelog
from waveform_test import get_pcc, get_rmse


def get_float_array(s):
  """
  Convert string representation of list to float array.
  """
  return np.array([float(i) for i in s.strip('[').strip(']').split(', ')])


def get_checkpoint_scores(params, start_time):
  """
  Calculate and save checkpoint correlations.
  """
  corrs = []
  comparison_dir_path = os.path.join(params.comparison_dir_path, 'valid')
  comparison_paths = sorted(os.listdir(comparison_dir_path))
  
  for i, comparison_path in enumerate(comparison_paths):
    all_pred = None
    all_real = None
    df = pd.read_csv(os.path.join(comparison_dir_path, comparison_path))
    
    for _, row in df.iterrows():
      pred_rhc = get_float_array(row['pred_rhc'])
      real_rhc = get_float_array(row['real_rhc'])
      all_pred = pred_rhc if all_pred is None else np.concatenate((all_pred, pred_rhc))
      all_real = real_rhc if all_real is None else np.concatenate((all_real, real_rhc))
    
    pcc_r, pcc_ci95_lower, pcc_ci95_upper = get_pcc(all_real, all_pred)
    rmse, rmse_ci95_lower, rmse_ci95_upper = get_rmse(all_real, all_pred)

    checkpoint = f"{comparison_path.split('.')[0]}.checkpoint"
    corrs.append({
      'checkpoint': checkpoint,
      'pcc_r': pcc_r,
      'pcc_ci95_lower': pcc_ci95_lower,
      'pcc_ci95_upper': pcc_ci95_upper,
      'rmse': rmse,
      'rmse_ci95_lower': rmse_ci95_lower,
      'rmse_ci95_upper': rmse_ci95_upper
    })
    print(timelog(f'waveform_checkpoint | {params.dir_path} | {i}/{len(comparison_paths)} | {pcc_r:.3f} [{pcc_ci95_lower:.3f}, {pcc_ci95_upper:.3f}] | {rmse:.3f} | [{rmse_ci95_lower:.3f}, {rmse_ci95_upper:.3f}]', start_time))
  return corrs


def run(params):
  start_time = time()
  print(timelog(f'Run waveform_checkpoint for {params.dir_path}', start_time))
  scores = get_checkpoint_scores(params, start_time)
  scores_df = pd.DataFrame.from_dict(scores)
  scores_df.to_csv(os.path.join(params.dir_path, f'checkpoint_scores.csv'), index=False)
  best_score = scores_df.loc[scores_df['pcc_r'].idxmax()]
  with open(os.path.join(params.dir_path, 'checkpoint_best.txt'), 'w') as f:
    f.write(best_score.to_string())


if __name__ == '__main__':
  params = Params(os.path.join(sys.argv[1], 'params.json'))
  run(params)
