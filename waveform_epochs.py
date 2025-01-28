import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import torch
from paramutil import Params
from recordutil import SCGDataset
from time import time
from timelog import timelog
from waveform_test import get_waveform_comparisons
from waveform_train import Generator

def run(params):
  start_time = time()
  print(timelog(f'Calculating optimal epoch for {params.dir_path}', start_time))

  checkpoint_scores = []

  with open(params.valid_path, 'rb') as f:
    valid_loader = pickle.load(f)
  
  checkpoint_paths = os.listdir(params.checkpoint_dir_path)
  checkpoint_paths.sort()

  for i, checkpoint_path in enumerate(checkpoint_paths):
    print(timelog(f'{i}/{len(checkpoint_paths)}', start_time))

    checkpoint = torch.load(os.path.join(params.checkpoint_dir_path, checkpoint_path), weights_only=False)
    generator = Generator(len(params.in_channels))
    generator.load_state_dict(checkpoint['g_state_dict'])
    generator.eval()
    
    comparisons = get_waveform_comparisons(generator, valid_loader)
    comparisons_df = pd.DataFrame(comparisons)

    dtw_avg = comparisons_df['dtw'].mean().item()
    dtw_std = comparisons_df['dtw'].std().item()

    checkpoint_scores.append((checkpoint_path, dtw_avg, dtw_std))
  
  checkpoint_scores.sort(key=lambda x: x[1])
  checkpoint_df = pd.DataFrame(checkpoint_scores)
  checkpoint_df.to_csv(params.checkpoint_scores_path, index=False)

  top_epoch = checkpoint_scores[0]
  print('Top epoch:', top_epoch)

  x = [int(i[0].split('.')[0]) for i in checkpoint_scores]
  y = [i[1] for i in checkpoint_scores]

  plt.scatter(x, y)
  plt.title('Average DTW Score by Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Mean DTW')
  plt.savefig('epoch_scores.png')
  plt.close()

if __name__ == '__main__':
  with open('active_project.txt', 'r') as f:
    path = f.readline().strip('\n')
  run(Params(path))
