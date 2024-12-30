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


def get_cross_correlation(x, y):
  """
  Definition: Cross-correlation measures how well one waveform matches another 
  as a function of a time-lag.

  Application: Ideal for detecting shifts, phase differences, or similarities
  in time-domain signals.

  Strengths: Identifies phase shifts or time alignment differences. Robust
  for signals with similar shapes but varying starting points.

  Limitations: Does not account for amplitude scaling differences.
  """
  return np.correlate(x, y, mode='full')


def get_mean_squared_error(x, y):
  """
  Definition: Measures the average squared difference between two waveforms.

  Application: Useful for quantifying dissimilarity; often used in regression
  and signal reconstruction contexts.

  Strengths: Simple to compute. Directly reflects pointwise differences.

  Limitations: Sensitive to amplitude scaling. Does not differentiate well
  between phase-shifted waveforms.
  """
  return np.mean((np.array(x) - np.array(y)) ** 2)


def get_dynamic_time_warping(x, y):
  """
  Definition: Aligns two sequences that may vary in speed or timing to find the
  optimal match.

  Application: Suitable for comparing signals with non-linear time shifts, such
  as in speech or motion analysis.

  Strengths: Accounts for temporal misalignments.

  Limitations: Computationally intensive for long signals.
  """
  x = np.array([x])
  y = np.array([y])
  distance, _ = fastdtw(x, y, dist=euclidean)
  return distance


def get_cosine_similarity(x, y):
  """
  Definition: Measures the cosine of the angle between two waveform vectors.

  Application: Suitable when you care about the overall shape rather than 
  magnitude differences.

  Strengths: Invariant to scaling.

  Limitations: Does not consider time shifts or phase differences.
  """
  return 1 - cosine(x, y)


def get_spectral_similarity(x, y):
  """
  Definition: Compares waveforms in the frequency domain by transforming them
  with a Fourier transform.

  Metrics: Magnitude spectrum compares the amplitude components. Phase spectrum
  compares the phase alignment. Spectral angle mapper (SAM) measures the angle
  between spectral vectors.

  Application: Ideal for periodic signals or signals with distinct frequency
  characteristics.

  Strengths: Useful for frequency-dominated signals.

  Limitations: Loses time-domain information.
  """
  fft_x = np.fft.fft(x)
  fft_y = np.fft.fft(y)
  return np.sum(np.abs(fft_x - fft_y))


def get_structural_similarity(x, y):
  """
  Definition: A perceptual metric that quantifies similarity in structure,
  luminance, and contrast.

  Application: Common in image and signal analysis for perceptual similarity.

  Strengths: Captures structural similarities rather than raw differences.

  Limitations: Less intuitive for time-domain waveform analysis. 
  """
  x = np.array(x)
  y = np.array(y)
  return ssim(x, y, data_range=1.0)


def get_correlation_coefficient(x, y):
  """
  Definition: Measures the linear relationship between two waveforms.

  Application: Indicates how well the shape of two signals aligns.

  Strengths: Invariant to shifts in mean. Simple to interpret. 

  Limitations: Sensitive to non-linear relationships.
  """
  return np.corrcoef(x, y)[0, 1]


def get_waveform_comparison(x, y):
  """
  Get comparison metrics for real and predicted RHC waveforms.
  """
  return {
    'real_rhc': x,
    'pred_rhc': y,
    'crosscor': get_cross_correlation(x, y),
    'mse': get_mean_squared_error(x, y),
    'dtw': get_dynamic_time_warping(x, y),
    'cosim': get_cosine_similarity(x, y),
    'specsim': get_spectral_similarity(x, y),
    'strucsim': get_structural_similarity(x, y),
    'corcoeff': get_correlation_coefficient(x, y)
  }


def get_waveform_comparisons(generator, loader):
  """
  Get all waveform comparions for real and predicted RHC waveforms.
  """
  comparisons = []
  for i, (scg, real_rhc, filename, start_idx, stop_idx) in enumerate(loader, start=1):
    real_rhc = real_rhc.detach().numpy()[0, 0, :]
    pred_rhc = generator(scg).detach().numpy()[0, 0, :]
    comparison = get_waveform_comparison(real_rhc, pred_rhc)
    filename = filename[0]
    comparison['filename'] = filename
    comparison['start_idx'] = start_idx
    comparison['stop_idx'] = stop_idx
    with open(os.path.join(PROCESSED_DATA_PATH, f'{filename}.json'), 'r') as f:
      comparison.update(json.load(f))
    comparisons.append(comparison)
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

  save_pred_plots(generator, train_loader, num_plots=5)
  save_pred_plots(generator, test_loader, num_plots=5)
  
  comparisons = get_waveform_comparisons(generator, test_loader)
  comparisons_df = pd.DataFrame(comparisons)
  comparisons_df.to_csv('comparisons.csv')


if __name__ == '__main__':
  params_path = '01_waveform_params.json'
  run(params_path)

