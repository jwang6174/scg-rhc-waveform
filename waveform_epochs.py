"""
Determine overall correlation for each epoch. There are two methods of
calculating overall correlation. One method is to calculate separation
correlation coefficients. Another method is to concatenate waveforms
and calculate a single coefficient.

Calculate correlation for each waveform and averaging. This approach gives you
a measure of how well your prediction performs on each individual waveform
(e.g., each trial or sample) rather than an overall global measure. By
computing separate correlations, you can see the spread or variability of
performance across waveforms, which may be useful for understanding consistency.
If different waveforms have different characteristics (e.g, amplitude scales,
noise levels, or underlying patterns), computing the correlation separately
respects these differences. However, the Pearson correlation coefficient is not
an additive or linear quantity. Simply averaging raw correlation coefficients
can be misleading. To circumvent this issue, Fisher z-transformation is applied
to each correlation coefficient, averaged the transformed values, and then
transformed back. In addition, shorter or noisier waveforms might yield 
unreliable estimates of the correlation coefficient.

Concatenating waveforms and calculating one overall correlation. This method
can be useful if you want to capture the global relationship between the
predicted and actual signals. When concatenated, the larger samle size may
reduce the impact of random noise or fluctuations in invidiual waveforms.
However, this approach ignores variability across different waveforms. If some
waveforms are predicted well and others poorly, a single overall coefficient
might hide this heterogeneity. This method also assumes that all waveforms are
part of one homogeneous process. If the waveforms have different baselines,
variances, or underlying dynamics, concatenation might mix incompatible data.
Lastly, if there are discontinuities or offsets at the boundaries between
concatenated waveforms, these might artificially affect the correlation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from paramutil import Params
from scipy.stats import pearsonr, norm
from time import time
from timelog import timelog


def fisher_z(r):
  """
  Apply Fisher Z-transformation.
  """
  return np.arctanh(r)


def inverse_fisher_z(z):
  """
  Inverse Fisher Z-transformation.
  """
  return np.tanh(z)


def get_avg_pcc(all_pred, all_real):
  """
  Calculate Pearson correlations for multiple pairs of waveforms using Fisher
  Z-transformation and calculate the p-value for testing the null hypothesis
  that the true correlation is zero.
  """
  if len(all_pred) != len(all_real):
    raise Exception('Number of predicted and actual waveforms must be the same')
  correlations = []
  m = len(all_pred)
  n = None
  for pred, real in zip(all_pred, all_real):
    if len(pred) != len(real):
      raise Exception('Each pair of waveforms must have the same number of samples')
    if n is None:
      n = len(pred)
    elif n != len(pred):
      raise Exception('All waveform pairs must have the same number of samples')
    r, _ = pearsonr(all_pred, all_real)
    r = np.clip(r, -0.9999, 0.9999)
    correlations.append(r)
  z_vals = np.array([fisher_z(r) for r in correlations])
  z_avg = np.mean(z_vals)
  z_std = np.std(z_vals)
  r_avg = inverse_fisher_z(z_avg)
  sem = 1 / np.sqrt(m * (n - 3))
  z_stat = z_avg / sem
  p_val = 2 * (1 - norm.cdf(np.abs(z_stat)))
  print(sem, z_stat, norm.cdf(np.abs(z_stat)))
  return r_avg, p_val


def get_rhc_float(rhc_str):
  """
  Convert RHC string to list of floats.
  """
  rhc_str = rhc_str.strip('[').strip(']')
  rhc_float = [float(s) for s in rhc_str.split()]
  return rhc_float


def get_checkpoint_scores(params, dataset, start_time):
  """
  Calculate and save checkpoint correlations.
  """
  corrs = []
  comparison_dir_path = os.path.join(params.comparison_dir_path, dataset)
  checkpoint_paths = sorted(os.listdir(comparison_dir_path))
  for i, comparison_path in enumerate(checkpoint_paths):
    all_pred = []
    all_real = []
    df = pd.read_csv(os.path.join(comparison_dir_path, comparison_path))
    for _, row in df.iterrows():
      pred_rhc = get_rhc_float(row['pred_rhc'])
      real_rhc = get_rhc_float(row['real_rhc'])
      all_pred.append(pred_rhc)
      all_real.append(real_rhc)
    avg, std = get_avg_pcc(all_pred, all_real)
    checkpoint = int(comparison_path.split('.')[0])
    corrs.append((checkpoint, avg, std))
    print(timelog(f'{dataset} | {i}/{len(checkpoint_paths)} | {avg:.3f} ({std:.3f})', start_time))
  return corrs


def run(params):
  start_time = time()
  print(timelog(f'Calculating optimal epoch for {params.dir_path}', start_time))
  
  scores = get_checkpoint_scores(params, 'valid', start_time)

  valid_x = [i[0] for i in correlations]
  valid_y = [i[1] for i in correlations]
  valid_e = [i[2] for i in correlations]

  plt.errorbar(valid_x, valid_y, valid_e, label='Valid')
  plt.title('Epoch vs Avg Score')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Score (SD)')
  plt.legend()
  plt.savefig(os.path.join(params.dir_path, 'checkpoint_scores.png'))
  plt.close()
  
  scores_df = pd.DataFrame(valid_scores, columns=['checkpoint', 'avg', 'std'])
  scores_df.to_csv(os.path.join(params.dir_path, f'checkpoint_scores.csv'), index=False)
  best_score = valid_scores_df.loc[valid_scores_df['score'].idxmax()]

  with open(os.path.join(params.dir_path, 'checkpoint_best.txt'), 'w') as f:
    f.write(best_score.to_string())


if __name__ == '__main__':
  with open('project_active.json', 'r') as f:
    data = json.load(f)
  run(Params(data['params_path']))
