import json
import os
import pandas as pd
from pathutil import PROCESSED_DATA_PATH
from recordutil import get_record_names, get_segments

jsons = []
for filename in get_record_names(PROCESSED_DATA_PATH):
  path = os.path.join(PROCESSED_DATA_PATH, f'{filename}.json')
  with open(path, 'r') as f:
    jsons.append(json.load(f))
df = pd.DataFrame(jsons)

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


