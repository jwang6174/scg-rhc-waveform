import os
import sys
from paramutil import Params
from recordutil import run as recordutil, SCGDataset
from waveform_train import run as waveform_train
from waveform_test import run as waveform_test
from waveform_epochs import run as waveform_epochs


def run(params):
  recordutil(params)
  waveform_train(params)
  waveform_test(params, 'valid', 'all')
  waveform_epochs(params)


if __name__ == '__main__':
  dir_name = sys.argv[1]
  params = Params(os.path.join(dir_name, 'params.json'))
  run(params)
