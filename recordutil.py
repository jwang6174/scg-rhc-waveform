import os
import wfdb
from pathlib import Path

data_dir = os.path.join('/', 'Users', 'jessewang', 'MyStuff', 'Projects', 'SCG-RHC', 'data')
processed_data_dir = os.path.join(data_dir, 'processed_data')


def get_record_names(dirname):
  filenames = set()
  for filename in os.listdir(dirname):
    filenames.add(Path(filename).stem)
  return list(filenames)


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


def get_scg_rhc_segments_from_record(record, segment_size):
  try:
    scg_signal = get_channels(record, channel_names=['patch_ACC_lat', 'patch_ACC_hf', 'patch_ACC_dv'])
    rhc_signal = get_channels(record, channel_names=['RHC_pressure'])
  except ValueError:
    return []

  segments = []
  num_segments = record.p_signal.shape[0] // segment_size
  for i in range(num_segments):
    start_idx = i * segment_size
    stop_idx = start_idx + segment_size
    scg_segment = scg_signal[start_idx:stop_idx]
    rhc_segment = rhc_signal[start_idx:stop_idx]
    segments.append((scg_segment, rhc_segment))
  return segments


def get_scg_rhc_segments(segment_size):
  segments = []
  for record in get_records(processed_data_dir):
    segments.extend(get_scg_rhc_segments_from_record(record, segment_size))
  return segments
