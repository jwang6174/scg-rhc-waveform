import os
import wfdb
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = os.path.join('/', 'Users', 'jessewang', 'MyStuff', 'Projects', 'SCG-RHC', 'data')

def get_record_names(dirname):
  filenames = set()
  for filename in os.listdir(dirname):
    filenames.add(Path(filename).stem)
  return filenames


def get_records(dirname):
  records = []
  for record_name in get_record_names(dirname):
    record = wfdb.rdrecord(os.path.join(dirname, record_name))
    records.append(record)
  return records


def get_channels(record, channel_names):
  indexes = [record.sig_name.index(name) for name in channel_names]
  channels = record.p_signal[:, indexes]
  return channels


def get_segments_from_record(record_name, input_dir, segment_size):
  segments = []
  record = wfdb.rdrecord(os.path.join(input_dir, record_name))
  num_segments = record.p_signal.shape[0] // segment_size
  for i in range(num_segments):
    start_idx = i * segment_size
    end_idx = start_idx + segment_size
    segment_signals = record.p_signal[start_idx:end_idx]
    segments.append(segment_signals)
  if num_segments % segment_size != 0:
    remaining_signals = record.p_signal[num_segments * segment_size:, :]
    segments.append(remaining_signals)
  return segments


def get_segments(segment_size):
  segments = []
  input_dir = os.path.join(data_dir, 'processed_data')
  for record_name in get_record_names(input_dir):
    segments.extend(get_segments(record_name, input_dir, segment_size))
  return segments
