import matplotlib.pyplot as plt
from recordutil import get_scg_rhc_segments

def save_random_rhc_segment(segment_size):
  segments = get_scg_rhc_segments(['patch_ACC_lat', 'patch_ACC_hf'], segment_size)
  _, rhc = segments[0]
  plt.plot(rhc)
  plt.savefig('random_rhc.png')
  plt.close()

if __name__ == '__main__':
  save_random_rhc_segment(750)
