import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import wfdb
from datetime import datetime
from paramutil import Params
from pathlib import Path
from pathutil import PROCESSED_DATA_PATH
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from waveform_noise import has_noise

SAMPLE_FREQ = 500


class SCGDataset(Dataset):
  """
  Container dataset class SCG and RHC segments.
  """
  def __init__(self, segments, segment_size, minmax_scg, minmax_rhc):
    self.segments = segments
    self.segment_size = int(segment_size * SAMPLE_FREQ)
    self.minmax_scg = minmax_scg
    self.minmax_rhc = minmax_rhc

  def pad(self, tensor):
    """
    Add padding to segment to match specified size.
    """
    if tensor.shape[-1] < self.segment_size:
        padding = self.segment_size - tensor.shape[-1]
        tensor = torch.nn.functional.pad(tensor, (0, padding))
    elif tensor.shape[-1] > self.segment_size:
        tensor = tensor[:, :, :self.segment_size]
    return tensor

  def minmax_norm(self, tensor, minmax_vals):
    """
    Perform min-max normalization on tensor.
    """
    min_val, max_val = minmax_vals
    tensor = (tensor - min_val) / (max_val - min_val + 0.0001)
    return tensor

  def invert(self, tensor):
    """
    Invert a tensor.
    """
    return torch.tensor(tensor.T, dtype=torch.float32)

  def __len__(self):
    """
    Get number of segments.
    """
    return len(self.segments)

  def __getitem__(self, index):
    """
    Iterate through segments.
    """
    segment = self.segments[index]
    scg = self.pad(self.invert(self.minmax_norm(segment[0], self.minmax_scg)))
    rhc = self.pad(self.invert(self.minmax_norm(segment[1], self.minmax_rhc)))
    record_name = segment[2]
    start_idx = segment[3]
    stop_idx = segment[4]
    return scg, rhc, record_name, start_idx, stop_idx


def get_record_names():
  """
  Get record names in a given directory.
  """
  filenames = set()
  for filename in os.listdir(PROCESSED_DATA_PATH):
    if filename.endswith('.dat') or filename.endswith('.hea'):
      filenames.add(Path(filename).stem)
  return list(filenames)


def get_chamber_intervals(record_name, chamber):
  """
  Get time intervals in seconds for when cath was in a particular chamber.
  """
  intervals = []
  with open(os.path.join(PROCESSED_DATA_PATH, f'{record_name}.json'), 'r') as f:
    data = json.load(f)
    macStTime = datetime.strptime(data['MacStTime'].split()[1], '%H:%M:%S')
    macEndTime = datetime.strptime(data['MacEndTime'].split()[1], '%H:%M:%S')
    chamEvents = data['ChamEvents_in_s']
    if isinstance(chamEvents, dict):
      chamEvents['END'] = (macEndTime - macStTime).total_seconds()
      chamEvents = sorted(chamEvents.items(), key=lambda x: x[1])
      intervals = []
      for i, event in enumerate(chamEvents[:-1]):
        if event[0].split('_')[0] == chamber:
          intervals.append((int(event[1] * SAMPLE_FREQ), int(chamEvents[i+1][1] * SAMPLE_FREQ)))
  return intervals


def get_channels(record, channel_names, start_idx, stop_idx):
  """
  Get specific channels from record by channel name.
  """
  indexes = [record.sig_name.index(name) for name in channel_names]
  channels = record.p_signal[start_idx:stop_idx, indexes]
  return channels

   
def get_segments(in_channels, segment_size, chamber, record_name=None):
  """
  Get segments of a given size with the specified SCG channels.
  """
  if record_name is None:
    segments = []
    for record_name in get_record_names():
      segments.extend(get_segments(in_channels, segment_size, chamber, record_name=record_name))
    return segments
  else:
    segments = []
    segment_size = int(segment_size * SAMPLE_FREQ)
    record = wfdb.rdrecord(os.path.join(PROCESSED_DATA_PATH, record_name))
    for interval in get_chamber_intervals(record_name, chamber):
      scg_signal = get_channels(record, in_channels, interval[0], interval[1])
      rhc_signal = get_channels(record, ['RHC_pressure'], interval[0], interval[1])
      num_segments = scg_signal.shape[0] // segment_size
      for i in range(num_segments):
        start_idx = i * segment_size
        stop_idx = start_idx + segment_size
        scg_segment = scg_signal[start_idx:stop_idx]
        rhc_segment = rhc_signal[start_idx:stop_idx]
        if not has_noise(rhc_segment[:, 0]):
          segments.append((scg_segment, rhc_segment, record_name, start_idx, stop_idx))
    return segments


def get_minmax_vals(segments):
  """
  Get min and max values for normalization.
  """
  scg_min = None
  scg_max = None
  rhc_min = None
  rhc_max = None
  for segment in segments:
    scg_min_seg = np.min(segment[0])
    scg_max_seg = np.max(segment[0])
    rhc_min_seg = np.min(segment[1])
    rhc_max_seg = np.max(segment[1])
    scg_min = scg_min_seg if scg_min is None else min(scg_min_seg, scg_min)
    scg_max = scg_max_seg if scg_max is None else max(scg_max_seg, scg_max)
    rhc_min = rhc_min_seg if rhc_min is None else min(rhc_min_seg, rhc_min)
    rhc_max = rhc_max_seg if rhc_max is None else max(rhc_max_seg, rhc_max)
  return (scg_min, scg_max), (rhc_min, rhc_max)


def save_dataloaders(params):
  """
  Get training and test segments, then save as torch DataLoader objects.
  """
  all_segments = get_segments(params.in_channels, params.segment_size, params.chamber)
  
  minmax_scg, minmax_rhc = get_minmax_vals(all_segments)

  train_segments, test_segments = train_test_split(all_segments, train_size=0.9)

  train_set = SCGDataset(train_segments, params.segment_size, minmax_scg, minmax_rhc)
  test_set = SCGDataset(test_segments, params.segment_size, minmax_scg, minmax_rhc)

  train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
  
  with open(params.train_path, 'wb') as f:
    pickle.dump(train_loader, f)
  
  with open(params.test_path, 'wb') as f:
    pickle.dump(test_loader, f)

def load_dataloader(path):
  """
  Load prior DataLoader object.
  """
  with open(path, 'rb') as f:
    return pickle.load(f)


if __name__ == '__main__':
  path = '04_waveform/params.json'
  print(f'Running recordutil for {path}')
  params = Params(path)
  save_dataloaders(params)
