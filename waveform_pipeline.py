import os
import sys
from paramutil import Params
from recordutil import run as recordutil, SCGDataset
from waveform_train import run as waveform_train
from waveform_test import run as waveform_test
from waveform_checkpoint import run as waveform_checkpoint


def run(params):
  
  try:
    recordutil(params)
  except Exception as e:
    print(e)
  
  waveform_train(params)
  
  try:
    waveform_test(params, 'valid', 'all')
  except Exception as e:
    print(e)
  
  waveform_checkpoint(params)
  
  with open(os.path.join(params.dir_path, 'checkpoint_best.txt'), 'r') as f:
    best_checkpoint = f.read().splitlines()[0].split()[1]
    waveform_test(params, 'test', best_checkpoint)


if __name__ == '__main__':
  dir_name = sys.argv[1]
  if dir_name == 'all':
    for i in range(6, 34):
      dir_name = f'waveform_{i:02d}'
      params = Params(os.path.join(dir_name, 'params.json'))
      run(params)
  else:
    params = Params(os.path.join(dir_name, 'params.json'))
    run(params)
