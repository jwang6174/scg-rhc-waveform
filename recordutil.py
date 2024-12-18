import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import wfdb
from pathlib import Path
from pathutil import PROCESSED_DATA_PATH
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from waveform_noise import has_noise

class SCGDataset(Dataset):
  """
  Container dataset class SCG and RHC segments.
  """
  def __init__(self, segments, segment_size):
    self.segments = segments
    self.segment_size = segment_size

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

  def minmax_norm(self, tensor):
    """
    Perform min-max normalization on tensor.
    """
    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 0.0001)
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
    scg = self.pad(self.invert(self.minmax_norm(segment[0])))
    rhc = self.pad(self.invert(self.minmax_norm(segment[1])))
    filename = segment[2]
    start_idx = segment[3]
    stop_idx = segment[4]
    return scg, rhc, filename, start_idx, stop_idx


def get_record_names(dirname):
  """
  Get record names in a given directory.
  """
  filenames = set()
  for filename in os.listdir(dirname):
    if filename.endswith('.dat') or filename.endswith('.hea'):
      filenames.add(Path(filename).stem)
  return list(filenames)


def get_records(dirname):
  """
  Get WFDB record objects in a given directory.
  """
  records = []
  for filename in get_record_names(dirname):
    record_obj = wfdb.rdrecord(os.path.join(dirname, filename))
    records.append((filename, record_obj))
  return records


def get_channels(record, channel_names):
  """
  Get specific channels from record by channel name.
  """
  indexes = [record.sig_name.index(name) for name in channel_names]
  channels = record.p_signal[:, indexes]
  return channels
  

def get_segments(scg_channels, size, record=None):
  """
  Get segments of a given size with the specified SCG channels.
  """
  if record is None:
    segments = []
    for record in get_records(PROCESSED_DATA_PATH):
      segments.extend(get_segments(scg_channels, size, record=record))
    return segments
  else:
    try:
      segments = []
      filename = record[0]
      record_obj = record[1]
      scg_signal = get_channels(record_obj, scg_channels)
      rhc_signal = get_channels(record_obj, ['RHC_pressure'])
      num_segments = record_obj.p_signal.shape[0] // size
      for i in range(num_segments):
        start_idx = i * size
        stop_idx = start_idx + size
        scg_segment = scg_signal[start_idx:stop_idx]
        rhc_segment = rhc_signal[start_idx:stop_idx]
        if not has_noise(rhc_segment[:, 0]):
          segments.append((scg_segment, rhc_segment, filename, start_idx, stop_idx))
      return segments
    except ValueError:
      return []


def save_dataloaders(scg_channels, segment_size, batch_size, 
                     train_path, valid_path, test_path):
  """
  Get training and test segments, then save as torch DataLoader objects.
  """
  all_segments = get_segments(scg_channels, segment_size)
  train_segments, non_train_segments = train_test_split(all_segments, train_size=0.9)
  valid_segments, test_segments = train_test_split(train_segments, train_size=0.5)

  train_set = SCGDataset(train_segments, segment_size)
  valid_set = SCGDataset(valid_segments, segment_size)
  test_set = SCGDataset(test_segments, segment_size)

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True)
  test_loader = Datazloader(test_set, batch_size=1, shuffle=True)
  
  with open(train_path, 'wb') as f:
    pickle.dump(train_loader, f)

  with open(valid_path, 'wb') as f:
    pickle.dump(valid_loader, f)
  
  with open(test_path, 'wb') as f:
    pickle.dump(test_loader, f)


def load_dataloader(path):
  """
  Load prior DataLoader object.
  """
  with open(path, 'rb') as f:
    return pickle.load(f)


if __name__ == '__main__':
  save_dataloaders(
    scg_channels=['patch_ACC_lat', 'patch_ACC_hf'],
    segment_size=750,
    batch_size=128,
    train_path='waveform_loader_train.pickle',
    valid_path='waveform_loader_valid.pickle',
    test_path='waveform_loader_test.pickle'
  )


