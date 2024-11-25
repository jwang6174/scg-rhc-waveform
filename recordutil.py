import matplotlib.pyplot as plt
import os
import wfdb
from pathlib import Path
from pathutil import PROCESSED_DATA_PATH


def get_record_names(dirname):
  filenames = set()
  for filename in os.listdir(dirname):
    if filename.endswith('.dat') or filename.endswith('.hea'):
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


def get_scg_rhc_segments(scg_channel_names, segment_size, record=None):
  if record is None:
    segments = []
    for record in get_records(PROCESSED_DATA_PATH):
      segments.extend(get_scg_rhc_segments(scg_channel_names, segment_size, record=record))
    return segments
  else:
    try:
      segments = []
      scg_signal = get_channels(record, scg_channel_names)
      rhc_signal = get_channels(record, channel_names=['RHC_pressure'])
      num_segments = record.p_signal.shape[0] // segment_size
      for i in range(num_segments):
        start_idx = i * segment_size
        stop_idx = start_idx + segment_size
        scg_segment = scg_signal[start_idx:stop_idx]
        rhc_segment = rhc_signal[start_idx:stop_idx]
        segments.append((scg_segment, rhc_segment))
      return segments
    except ValueError:
      return []  

