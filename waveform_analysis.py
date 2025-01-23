import pandas as pd
from paramutil import Params

def run(params):
  df = pd.read_csv(params.comparisons_path)
  
  dtw_avg = df['dtw'].mean().item()
  dtw_std = df['dtw'].std().item()
  print(f'{dtw_avg = :.2f}, {dtw_std = :.2f}')

  pcc_avg = df['pcc'].mean().item()
  pcc_std = df['pcc'].std().item()
  print(f'{pcc_avg = :.2f}, {pcc_std = :.2f}')

  rmse_avg = df['rmse'].mean().item()
  rmse_std = df['rmse'].std().item()
  print(f'{rmse_avg = :.2f}, {rmse_std = :.2f}')

  mae_avg = df['mae'].mean().item()
  mae_std = df['mae'].std().item()
  print(f'{mae_avg = :.2f}, {mae_std = :.2f}')


if __name__ == '__main__':
  params = Params('02_waveform/params.json')
  run(params)
