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


def write_segments(record_name, input_dir, output_dir, segment_size):  
  record = wfdb.rdrecord(os.path.join(input_dir, record_name))
  num_segments = record.p_signal.shape[0] // segment_size
  for i in range(num_segments):
    start_idx = i * segment_size
    end_idx = start_idx + segment_size
    segment_signals = record.p_signal[start_idx:end_idx]
    wfdb.wrsamp(
      record_name=os.path.join(f'{record_name}_split_{i}'),
      write_dir=output_dir,
      fs=record.fs,
      units=record.units,
      sig_name=record.sig_name,
      p_signal=segment_signals)


def write_all_segments(segment_size):
  input_dir = os.path.join(data_dir, 'processed_data')
  output_dir = os.path.join(data_dir, f'processed_data_size_{segment_size}')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  for record_name in get_record_names(input_dir):
    write_segments(record_name, input_dir, output_dir, segment_size)


def get_train_and_test_records(dirname):
  records = get_records(dirname)
  train_records, test_records = train_test_split(records, test_size=0.2)
