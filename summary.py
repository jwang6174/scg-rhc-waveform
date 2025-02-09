import json
import numpy as np
import os
import pandas as pd
import wfdb
from pathutil import PROCESSED_DATA_PATH
from recordutil import get_record_names, get_chamber_intervals, SAMPLE_FREQ
from scipy.stats import ttest_ind, ranksums


def get_modified_maclab_meas(original):
  modified = {}
  for key, val in original.items():
    key = key.strip()
    if isinstance(val, str):
      modified[key] = np.nan
    else:
      modified[key] = val
  return modified


def get_df():
  df = {}
  for record_name in get_record_names():
    path = os.path.join(PROCESSED_DATA_PATH, f'{record_name}.json')
    with open(path, 'r') as f:
      data = json.load(f)
      data['record_name'] = record_name
      data['sbp'] = np.nan if data['sbp'] == -1 else data['sbp']
      data['dbp'] = np.nan if data['dbp'] == -1 else data['dbp']
      data.update(get_modified_maclab_meas(data['maclabMeas']))
      df[record_name] = data
  return df


def get_signal_names(records):
  names = set()
  for record in records:
    names.update(record.sig_name)
  return names


def add_signal_presence(df):
  record_names = df.keys()
  records = [wfdb.rdrecord(os.path.join(PROCESSED_DATA_PATH, x)) for x in record_names]
  signal_names = get_signal_names(records)
  for record_name, record in zip(record_names, records):
    for signal_name in signal_names:
      df[record_name][signal_name] = signal_name in record.sig_name


def add_chamber_durations(df):
  for record_name in df.keys():
    chamber_times = {
      'RA': 0,
      'RV': 0,
      'PA': 0,
      'PCW': 0
    }
    for chamber in chamber_times.keys():
      for start, end in get_chamber_intervals(record_name, chamber):
        time = (end - start) / SAMPLE_FREQ
        chamber_times[chamber] += time
    for chamber, time in chamber_times.items():
      df[record_name][chamber] = time


def summarize_continuous(df, var, gender_stratified):
  print(var)
  print(f"  Min {df[var].min():.2f}")
  print(f"  Max {df[var].max():.2f}")
  print(f"  Avg {df[var].mean():.2f} ± {df[var].std():.2f}")
  print(f"  Sum {df[var].sum():.2f}")
  if not gender_stratified:
    group1 = df[df['gender'] == 'Male'][var]
    group2 = df[df['gender'] == 'Female'][var]
    _, p_value = ttest_ind(group1, group2, nan_policy='omit')
    print(f"  Sig {p_value:.2f}")


def summarize_boolean(df, var):
  print(var)
  print(f"  Y {df[var].value_counts().get(True)}")
  print(f"  N {df[var].value_counts().get(False)}")


def show_missing_vals(df):
  print('Missing vals:')
  print(df.isna().sum()[df.isna().sum() > 0])


def summarize(df, gender_stratified):
  continuous_vars = [
    'age',
    'bmi',
    'sbp',
    'dbp',
    'RA',
    'RV',
    'PA',
    'PCW',
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
  
  boolean_vars = [
    'Missing_MaclabRHC',
    'fine_alignment',
    'outpatient',
    'patch_ECG',
    'patch_ACC_lat',
    'patch_ACC_hf',
    'patch_ACC_dv',
    'patch_Hum',
    'patch_Pre',
    'patch_Temp',
    'RHC_pressure',
    'ART',
    'ECG_lead_I',
    'ECG_lead_II',
    'ECG_lead_III',
    'aVR',
    'aVL',
    'aVF',
    'ECG_lead_V1',
    'ECG_lead_V2',
    'ECG_lead_V3',
    'ECG_lead_V4',
    'ECG_lead_V5',
    'ECG_lead_V6',
    'PLETH',
    'RESP',
  ]
  
  for var in continuous_vars:
    summarize_continuous(df, var, gender_stratified)
  
  for var in boolean_vars:
    summarize_boolean(df, var)
  
  print("NYHAC")
  print(f"  1 {df['NYHAC'].value_counts().get(1)}")
  print(f"  2 {df['NYHAC'].value_counts().get(2)}")
  print(f"  3 {df['NYHAC'].value_counts().get(3)}")
  print(f"  4 {df['NYHAC'].value_counts().get(4)}")
  if not gender_stratified:
    group1 = df[df['gender'] == 'Male']['NYHAC']
    group2 = df[df['gender'] == 'Female']['NYHAC']
    _, p_value = ranksums(group1, group2, nan_policy='omit')
    print(f"  Sig {p_value}")

  show_missing_vals(df)


if __name__ == '__main__':
  df = get_df()
  add_signal_presence(df)
  add_chamber_durations(df)
  df = [val for val in df.values()]
  df = pd.DataFrame.from_dict(df)
  df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2) 
 
  print('\n----- All -----')
  summarize(df, gender_stratified=False)

  print('\n----- Male -----')
  summarize(df[df['gender'] == 'Male'], gender_stratified=True)

  print('\n----- Female -----')
  summarize(df[df['gender'] == 'Female'], gender_stratified=True)

