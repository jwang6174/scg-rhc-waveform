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

def save_checkpoint_scores(params, loader, prefix, start_time):
  checkpoint_scores = []

  checkpoint_paths = os.listdir(params.checkpoint_dir_path)
  checkpoint_paths.sort()

  for i, checkpoint_path in enumerate(checkpoint_paths):

    checkpoint = torch.load(os.path.join(params.checkpoint_dir_path, checkpoint_path), weights_only=False)
    
    generator = Generator(len(params.in_channels))
    generator.load_state_dict(checkpoint['g_state_dict'])
    generator.eval()
    
    comparisons = get_waveform_comparisons(generator, loader)
    comparisons_df = pd.DataFrame(comparisons)
    
    avg = comparisons_df['pcc'].mean().item()
    std = comparisons_df['pcc'].std().item()
    
    checkpoint_num = int(checkpoint_path.split('.')[0])
    checkpoint_scores.append((checkpoint_num, avg, std))

    print(timelog(f'{prefix} | {i}/{len(checkpoint_paths)} | {avg:.2f} | {std:.2f}', start_time))
  
  return checkpoint_scores


def run(params):
  start_time = time()

  print(timelog(f'Calculating optimal epoch for {params.dir_path}', start_time))

  with open(params.valid_path, 'rb') as f:
    valid_loader = pickle.load(f)

  valid_scores = save_checkpoint_scores(params, valid_loader, 'valid', start_time)

  valid_x = [i[0] for i in valid_scores]
  valid_y = [i[1] for i in valid_scores]
  valid_e = [i[2] for i in valid_scores]

  plt.errorbar(valid_x, valid_y, valid_e, label='Valid')
  plt.title('Epoch vs Mean PCC')
  plt.xlabel('Epoch')
  plt.ylabel('Mean PCC (Std Dev)')
  plt.legend()
  plt.savefig(os.path.join(params.dir_path, 'checkpoint_scores.png'))
  plt.close()
  
  valid_scores_df = pd.DataFrame(valid_scores, columns=['checkpoint', 'score', 'std'])
  valid_scores_df.to_csv(os.path.join(params.dir_path, f'checkpoint_scores.csv'), index=False)

  max_score = valid_scores_df.loc[valid_scores_df['score'].idxmax()]

  with open(os.path.join(params.dir_path, 'checkpoint_score_max.txt'), 'w') as f:
    f.write(max_score.to_string())


if __name__ == '__main__':
  with open('active_project.txt', 'r') as f:
    path = f.readline().strip('\n')
  run(Params(path))
