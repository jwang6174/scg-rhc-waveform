import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_flat_lines(waveform, threshold=1e-3, min_duration=0.1, sampling_rate=500):
  
  min_samples = int(min_duration * sampling_rate)
  waveform_series = pd.Series(waveform)
    
  # Compute rolling max-min difference
  rolling_diff = waveform_series.rolling(window=min_samples).max() - \
                 waveform_series.rolling(window=min_samples).min()
    
  # Identify flat lines
  flat_indices = rolling_diff[rolling_diff < threshold].index
  flat_segments = []
  start = None

  for i in range(len(flat_indices) - 1):
    if start is None:
      start = flat_indices[i]
    if flat_indices[i + 1] != flat_indices[i] + 1:  # Gap detected
      flat_segments.append((start, flat_indices[i]))
      start = None

    # Handle the last segment
    if start is not None:
      flat_segments.append((start, flat_indices[-1]))
  
  return flat_segments


def get_vertical_lines(waveform, threshold=1.0, sampling_rate=500):
    
  # Compute the derivative (rate of change)
  derivative = np.diff(waveform) * sampling_rate  # Derivative approximation
    
  # Identify points where the derivative exceeds the threshold
  vertical_indices = np.where(np.abs(derivative) > threshold)[0]
    
  # Classify direction of the change
  directions = np.sign(derivative[vertical_indices])
    
  # Combine indices and directions
  vertical_lines = [(index, direction) for index, direction in zip(vertical_indices, directions)]
    
  return vertical_lines


def is_straight_line(waveform):
  x = np.arange(len(waveform))
  y = np.array(waveform)
  model = LinearRegression().fit(x.reshape(-1, 1), y)
  r_squared = model.score(x.reshape(-1, 1), y)
  return r_squared > 0.8


def has_noise(waveform):
  return (
    len(get_flat_lines(waveform)) > 0 or
    # len(get_vertical_lines(waveform)) > 0 or
    is_straight_line(waveform)
  )
