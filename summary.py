import json
import numpy as np
import os
import pandas as pd
import wfdb
from pathutil import PROCESSED_DATA_PATH
from recordutil import get_record_names, get_segments, get_chamber_intervals, SAMPLE_FREQ
    

record_names = get_record_names()

jsons = []
for record_name in record_names:
  path = os.path.join(PROCESSED_DATA_PATH, f'{record_name}.json')
  with open(path, 'r') as f:
    data = json.load(f)
    data['filename'] = record_name
    jsons.append(data)
df = pd.DataFrame(jsons)

records = []
for record_name in record_names:
  record = wfdb.rdrecord(os.path.join(PROCESSED_DATA_PATH, record_name))
  records.append(record)


def summarize(df):

  print(f"Record count: {len(df)}")

  print("Age (yr)")
  print(f"  Min {df['age'].min():.2f}")
  print(f"  Max {df['age'].max():.2f}")
  print(f"  Avg {df['age'].mean():.2f} ± {df['age'].std():.2f}")

  print("Height (cm)")
  print(f"  Min {df['height'].min():.2f}")
  print(f"  Max {df['height'].max():.2f}")
  print(f"  Avg {df['height'].mean():.2f} ± {df['height'].std():.2f}")

  print(f"Weight (kg)")
  print(f"  Min {df['weight'].min():.2f}")
  print(f"  Max {df['weight'].max():.2f}")
  print(f"  Avg {df['weight'].mean():.2f} ± {df['weight'].std():.2f}")

  print("SBP (mmHg)")
  print(f"  Min {df['sbp'].min():.2f}")
  print(f"  Max {df['sbp'].max():.2f}")
  print(f"  Avg {df['sbp'].mean():.2f} ± {df['sbp'].std():.2f}")

  print("DBP (mmHg)")
  print(f"  Min {df['dbp'].min():.2f}")
  print(f"  Max {df['dbp'].max():.2f}")
  print(f"  Avg {df['dbp'].mean():.2f} ± {df['dbp'].std():.2f}")

  print("Clinical status")
  print(f"  Comp {df['CDecomp'].value_counts().get(0)}")
  print(f"  Decomp {df['CDecomp'].value_counts().get(1)}")

  print(f"Physiologic status")
  print(f"  Comp {df['PDecomp'].value_counts().get(0)}")
  print(f"  Decomp {df['PDecomp'].value_counts().get(1)}")

  print("NYHAC")
  print(f"  1 {df['NYHAC'].value_counts().get(1)}")
  print(f"  2 {df['NYHAC'].value_counts().get(2)}")
  print(f"  3 {df['NYHAC'].value_counts().get(3)}")
  print(f"  4 {df['NYHAC'].value_counts().get(4)}")
  print(f"  Avg {df['NYHAC'].mean():.2f} ± {df['NYHAC'].std():.2f}")

  print("Physiologic challenge")
  print(f"  Y {df['IsChallenge'].value_counts().get(True)}")
  print(f"  N {df['IsChallenge'].value_counts().get(False)}")

  print("Outpatient")
  print(f"  Y {df['outpatient'].value_counts().get(True)}")
  print(f"  N {df['outpatient'].value_counts().get(False)}")

  print("Heart failure")
  print(f"  Y {df['heart failure'].value_counts().get(True)}")
  print(f"  N {df['heart failure'].value_counts().get(False)}")

  print("Missing Maclab RHC")
  print(f"  Y {df['Missing_MaclabRHC'].value_counts().get(True)}")
  print(f"  N {df['Missing_MaclabRHC'].value_counts().get(False)}")

  print("Fine alignment")
  print(f"  Y {df['fine_alignment'].value_counts().get(True)}")
  print(f"  N {df['fine_alignment'].value_counts().get(False)}")

  print("Diagnosed with HF")
  print(f"  Y {df['heart failure'].value_counts().get(True)}")
  print(f"  N {df['heart failure'].value_counts().get(False)}")

  print("Outpatient")
  print(f"  Y {df['outpatient'].value_counts().get(True)}")
  print(f"  N {df['outpatient'].value_counts().get(False)}")

  meas_dicts = []
  print('Missing maclabMeas')
  for _, row in df.iterrows():
    mod_dict = {}
    for key, val in row['maclabMeas'].items():
      if isinstance(val, str):
        print(f"  {row['filename']} {key} {val}")
        mod_dict[key.strip()] = np.nan
      else:
        mod_dict[key.strip()] = val
    meas_dicts.append(mod_dict)
  meas_df = pd.DataFrame.from_dict(meas_dicts)

  meas_vals = [
    'RAA Wave',
    'RAV Wave',
    'RAM',
    'RAHR',
    'RVS',
    'RVD',
    'RVEDP',
    'RVHR',
    'PAS',
    'PAD',
    'PAM',
    'PAHR',
    'PCWA Wave',
    'PCWV Wave',
    'PCWM',
    'PCWHR',
    'Fick COL/min',
    'TDCOL/min',
    'TDCIL/min/m^2',
    'Avg. COmL/min',
    'SVmL/beat',
  ]

  for val in meas_vals:
    print(val)
    print(f"  Min {meas_df[val].min():.2f}")
    print(f"  Max {meas_df[val].max():.2f}")
    print(f"  Avg {meas_df[val].mean():.2f} ± {meas_df[val].std():.2f}")

  chamber_times = {
    'RA': [],
    'RV': [],
    'PA': [],
    'PCW': [],
  }

  for chamber in chamber_times:
    for _, row in df.iterrows():
      time = 0
      for start_index, end_index in get_chamber_intervals(row['filename'], chamber):
        start_sec = start_index / SAMPLE_FREQ
        end_sec = end_index / SAMPLE_FREQ
        time += (end_sec - start_sec)
      chamber_times[chamber].append(time)

  for chamber, time in chamber_times.items():
    total = np.sum(time)
    avg = np.mean(time)
    std = np.std(time)
    print(f"{chamber} (mmHg)")
    print(f"  Avg {avg:.2f} ± {std:.2f}")
    print(f"  Sum {total}")

print('\n----- All -----\n')
summarize(df)

print('\n----- Male -----\n')
summarize(df[df['gender'] == 'Male'])

print('\n----- Female -----\n')
summarize(df[df['gender'] == 'Female'])
