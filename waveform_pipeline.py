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


if __name__ == '__main__':
  dir_name = sys.argv[1]
  params = Params(os.path.join(dir_name, 'params.json'))
  run(params)
