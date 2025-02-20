import os
import shutil

def clear(paths):
  for path in paths:
    if os.path.exists(path):
      shutil.rmtree(path)
      os.makedirs(path)
      print(f'Cleared {path}')


def clear_comparisons_valid():
  paths = [os.path.join(p, 'comparisons', 'valid') for p in sorted(os.listdir(os.getcwd()))]
  clear(paths)

