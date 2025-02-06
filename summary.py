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
    jsons.append(json.load(f))
df = pd.DataFrame(jsons)

records = []
for record_name in record_names:
  record = wfdb.rdrecord(os.path.join(PROCESSED_DATA_PATH, record_name))
  records.append(record)

print("Age (yr)")
print(f"  {df['age'].mean():.1f} ± {df['age'].std():.1f}")

print("Height (cm)")
print(f"  {df['height'].mean():.1f} ± {df['height'].std():.1f}")

print(f"Weight (kg)")
print(f"  {df['weight'].mean():.1f} ± {df['weight'].std():.1f}")

print("Gender")
print(f"  {df['gender'].value_counts().get('Male')} male")
print(f"  {df['gender'].value_counts().get('Female')} female")

print("BP (mmHg)")
print(f"  SBPs {df['sbp'].mean():.1f} ± {df['sbp'].std():.1f}")
print(f"  DBP {df['dbp'].mean():.1f} ± {df['dbp'].std():.1f}")

print("Clinical status")
print(f"  Comp {df['CDecomp'].value_counts().get(0)}")
print(f"  Decomp {df['CDecomp'].value_counts().get(1)}")

print(f"Physiologic status")
print(f"  Comp {df['PDecomp'].value_counts().get(0)}")
print(f"  Decomp {df['PDecomp'].value_counts().get(1)}")

print("NYHAC")
print(f"  I {df['NYHAC'].value_counts().get(1)}")
print(f"  II {df['NYHAC'].value_counts().get(2)}")
print(f"  III {df['NYHAC'].value_counts().get(3)}")
print(f"  IV {df['NYHAC'].value_counts().get(4)}")

print("Physiologic challenge")
print(f"  Yes {df['IsChallenge'].value_counts().get(True)}")
print(f"  No {df['IsChallenge'].value_counts().get(False)}")

print("Outpatient")
print(f"  Yes {df['outpatient'].value_counts().get(True)}")
print(f"  No {df['outpatient'].value_counts().get(False)}")

print("Heart failure")
print(f"  Yes {df['heart failure'].value_counts().get(True)}")
print(f"  No {df['heart failure'].value_counts().get(False)}")

print("Missing Maclab RHC")
print(f"  Yes {df['Missing_MaclabRHC'].value_counts().get(True)}")
print(f"  No {df['Missing_MaclabRHC'].value_counts().get(False)}")

print("Fine alignment")
print(f"  Yes {df['fine_alignment'].value_counts().get(True)}")
print(f"  No {df['fine_alignment'].value_counts().get(False)}")

print("Diagnosed with HF")
print(f"  Yes {df['heart failure'].value_counts().get(True)}")
print(f"  No {df['heart failure'].value_counts().get(False)}")

print("Outpatient")
print(f"  Yes {df['outpatient'].value_counts().get(True)}")
print(f"  No {df['outpatient'].value_counts().get(False)}")

sigs = {}
for record in records:
  for sig in record.sig_name:
    sigs[sig] = sigs.get(sig, 0) + 1

for sig, count in sigs.items():
   print(f'{sig}: {count}')

chamber_times = {
  'RA': [],
  'RV': [],
  'PA': [],
  'PCW': [],
}
for chamber in chamber_times:
  for record_name in record_names:
    time = 0
    for start_index, end_index in get_chamber_intervals(record_name, chamber):
      start_sec = start_index / SAMPLE_FREQ
      end_sec = end_index / SAMPLE_FREQ
      time += (end_sec - start_sec)
    chamber_times[chamber].append(time)

for chamber, time in chamber_times.items():
  total = np.sum(time)
  avg = np.mean(time)
  std = np.std(time)
  print(f'{chamber}: avg {avg:.2f} ({std:.2f}), total {total:.2f}')
