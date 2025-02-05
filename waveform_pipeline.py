import os
from paramutil import Params
from recordutil import run as recordutil, SCGDataset
from waveform_train import run as waveform_train
from waveform_test import run as waveform_test
from waveform_epochs import run as waveform_epochs

for i in range(12, 34):
  dirname = f'{i}_waveform'
  params = Params(os.path.join(dirname, 'params.json'))
  try:
    recordutil(params)
  except Exception as e:
    print(e)
  waveform_train(params)
  waveform_test(params, 'valid', 'all')
  waveform_epochs(params)

