import matplotlib.pyplot as plt
import os
import wfdb
from pathutil import PROCESSED_DATA_PATH
from recordutil import get_record_names

rhc_vals = []
for name in get_record_names():
  record = wfdb.rdrecord(os.path.join(PROCESSED_DATA_PATH, name))
  if 'RHC_pressure' in record.sig_name:
    rhc_index = record.sig_name.index('RHC_pressure')
    rhc_channel = record.p_signal[:, rhc_index]
    rhc_vals.extend(rhc_channel)

plt.hist(rhc_vals, bins=20)
plt.savefig('rhc_histo.png')
plt.close()
