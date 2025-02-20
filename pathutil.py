import os
import shutil

DATA_PATH = os.path.join('/', 'home', 'jesse', 'physionet.org', 'files', 'scg-rhc-wearable-database', '1.0.0')

PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed_data')


def clear(paths):
  for path in paths:
    if os.path.exists(path):
      shutil.rmtree(path)
      os.makedirs(path)
      print(f'Cleared {path}')


def clear_comparisons_valid():
  paths = [os.path.join(p, 'comparisons', 'valid') for p in sorted(os.listdir(os.getcwd()))]
  clear(paths)

