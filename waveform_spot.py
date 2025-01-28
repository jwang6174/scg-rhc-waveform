import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from paramutil import Params
from recordutil import SCGDataset

def run(params):
  with open(params.train_path, 'rb') as f:
    loader = pickle.load(f)

  for i, segment in enumerate(loader, start=1):
    scg = segment[0]
    rhc = segment[1]
    timestamp = str(datetime.now().strftime('%Y-%m-%d %H-%M-%S')).replace(' ', '_')
    rhc = rhc.detach().numpy()[0, 0, :]
    plt.plot(rhc)
    plt.xlabel('Sample')
    plt.ylabel('mmHg')
    plt.savefig(os.path.join(params.pred_rand_dir_path, f'random_pred_plot_{timestamp}_spot_{i}.png'))
    plt.close()
    if i == 5:
      break


if __name__ == '__main__':
  with open('active_project.txt', 'r') as f:
    path = f.readline().strip('\n')
  params = Params(path)
  run(params)
