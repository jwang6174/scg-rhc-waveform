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

def save_checkpoint_scores(params, loader, prefix):
  checkpoint_scores = []

  checkpoint_paths = os.listdir(params.checkpoint_dir_path)
  checkpoint_paths.sort()

  for i, checkpoint_path in enumerate(checkpoint_paths[:50]):
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
  
  checkpoint_scores_df = pd.DataFrame(checkpoint_scores)
  checkpoint_scores_df.to_csv(os.path.join(params.dir_path, f'checkpoint_scores_{prefix}.csv'))
  
  return checkpoint_scores


def run(params):
  start_time = time()

  print(timelog(f'Calculating optimal epoch for {params.dir_path}', start_time))
  
  with open(params.train_path, 'rb') as f:
    train_loader = pickle.load(f)

  with open(params.valid_path, 'rb') as f:
    valid_loader = pickle.load(f)

  train_scores = save_checkpoint_scores(params, train_loader, 'train')
  valid_scores = save_checkpoint_scores(params, valid_loader, 'valid')

  train_x = [int(i[0].split('.')[0]) for i in train_scores]
  train_y = [i[1] for i in train_scores]
  train_e = [i[2] for i in train_scores]

  valid_x = [int(i[0].split('.')[0]) for i in valid_scores]
  valid_y = [i[1] for i in valid_scores]
  valid_e = [i[2] for i in valid_scores]

  plt.errorbar(train_x, train_y, train_e, label='Train')
  plt.errorbar(valid_x, valid_y, valid_e, label='Valid')
  plt.title('Average DTW Score by Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Mean PCC (Std Dev)')
  plt.legend()
  plt.savefig('epoch_scores.png')
  plt.close()

if __name__ == '__main__':
  with open('active_project.txt', 'r') as f:
    path = f.readline().strip('\n')
  run(Params(path))
