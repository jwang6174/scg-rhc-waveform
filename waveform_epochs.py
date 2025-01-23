from paramutil import Params
from waveform_test import get_waveform_comparisons

def run(params):
  checkpoint_scores = []

  with open(params.valid_path, 'rb') as f:
    valid_loader = pickle.load(f)

  for checkpoint_path in os.listdir(dirpath):
    checkpoint = torch.load(os.path.join(params.checkpoint_dir_path, checkpoint_path), weights_only=False)
    generator = Generator(len(params.in_channels))
    generator.load_state_dict(checkpoint['g_state_dict'])
    generator.eval()
    
    comparisons = get_waveform_comparisons(generator, valid_loader)
    comparisons_df = pd.DataFrame(comparisons)

    dtw_avg = comparison_df['dtw'].mean().item()
    dtw_std = comparison_df['dtw'].std().item()

    checkpoint_scores.append((checkpoint_path, dtw_avg, dtw_std))
  
  checkpoint_scores.sort(key=lambda x: x[1])
  checkpoint_df = pd.DataFrame(checkpoint_scores)
  checkpoint_df.to_csv(params.checkpoint_scores_path, index=False)

  top_epoch = checkpoint_scores[0]
  print('Top epoch:', top_epoch)

  x = [int(i[0].split('.')[0]) for i in checkpoint_scores]
  y = [i[1] for i in checkpoint_scores]

  plt.scatter(x, y)
  plt.title('Average Dynamic Time Warping Score by Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Mean DTW')
  plt.savefig('epoch_scores.png')
  plt.close()
