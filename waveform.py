import os
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from recordutil import get_scg_rhc_segments

class SCGDataset(Dataset):

  def __init__(self, segments, segment_size):
    self.segments = segments
    self.segment_size = segment_size

  def pad(self, tensor):
    if tensor.shape[-1] < self.segment_size:
        padding = self.segment_size - tensor.shape[-1]
        tensor = torch.nn.functional.pad(tensor, (0, padding))
    elif tensor.shape[-1] > self.segment_size:
        tensor = tensor[:, :, :self.segment_size]
    return tensor

  def minmax_norm(self, tensor):
    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 0.0001)
    return tensor

  def invert(self, tensor):
    return torch.tensor(tensor.T, dtype=torch.float32)

  def __len__(self):
    return len(self.segments)

  def __getitem__(self, index):
    segments = self.segments[index]
    scg_tensor = self.pad(self.invert(self.minmax_norm(segments[0])))
    rhc_tensor = self.pad(self.invert(self.minmax_norm(segments[1])))
    return scg_tensor, rhc_tensor


class AttentionGate(nn.Module):
  """
  Enables the network to focus on the most relevant regions of the input image during
  segmentation. By introducing attention gates into the skip connections, Attention U-Net can
  dynamically decide which features to pass to the decoder, helping the model segment small or
  complex structures more accurately.
  """
  def __init__(self, F_g, F_l, F_int):
    super(AttentionGate, self).__init__()
    
    self.W_g = nn.Sequential(
        nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm1d(F_int))
    
    self.W_x = nn.Sequential(
        nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm1d(F_int))
    
    self.psi = nn.Sequential(
        nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm1d(1),
        nn.Sigmoid())
    
    self.relu = nn.ReLU(inplace=True)

  def forward(self, g, x):
    g1 = self.W_g(g)
    x1 = self.W_x(x)
    psi = self.psi(self.relu(g1 + x1))
    return x * psi


class AttentionUNetGenerator(nn.Module):
  """
  The Generator is an implementation of Attention U-Net with a condition for conditional GAN. The
  conditioning of the generator is implicit through the input.

  Attention U-Net is a fully convolutional network designed for image segmentation. It has an
  encoder-decoder structure with skip connections between corresponding layers in the encoder
  and decoder paths. The encoder extracts hierarchical features by progressively downsampling
  the input, and the decoder upscales the features to predict a segmentation mask with the same
  resolution as the input.

  1. Attention Gate:
  The attention gate compuates an attentipn map using both the encoder output (skip connection) and
  the decoder output. The encoder features are multiplied by this attention map before being
  concatenated with the decoder features.

  2. Skip Connections with Attention:
  Each up-convolution layer in the decoder path is followed by an attention block applied to the
  corresponding encoder feature map. This helps the network focus on the most relevant features
  passed through the skip connections.

  3. Up-Convolutions for Upsampling:
  The decoder path uses transposed convolutions (ConvTranspose1d) to upsample the feature maps.
  This ensures that the output has the same resolution as the input.

  Odd dimensions may occur in the data, which can be addressed by adding padding to ensure
  compatibility between encoder and decoder features. In U-Net models, dimensions are halved at
  each downsampling stage (via max pooling). When working with odd dimensions, this process may
  result in shapes that are incompatible for concatenation in skip connections. There are two 
  potential options to handle odd dimensions in a U-Net generator:

  1. Adjust Padding for Conv Layers
  - Use padding in the conv_block to maintain spatial dimensions at each layer.
  - At the downsampling (pooling stage), add padding as necessary to ensure each layer has even
  dimensions before being downsampled.

  2. Center Cropping in Decoder (Optional)
  - If padding is not desirable, can crop the upsampled feature maps to slightly match the
  encoder feature map dimensions in the skip connections.

  Another issue that may arise is discrepancy between the input and output lengths provided by the
  generator, which is likely due to the downsampling and upsampling operations. Specifically,
  pooling and convolutional layers can reduce the spatial dimensions, and when we upsample in the
  decoder path, it doesn't always perfectly match the original input length, especially if the
  input length is not a multiple of the downsampling factor.
  """
  def __init__(self, in_channels, out_channels):
    super(AttentionUNetGenerator, self).__init__()
    self.encoder1 = self.conv_block(in_channels, 64)
    self.encoder2 = self.conv_block(64, 128)
    self.encoder3 = self.conv_block(128, 256)
    self.encoder4 = self.conv_block(256, 512)
    self.bottleneck = self.conv_block(512, 1024)
    self.attention1 = AttentionGate(512, 512, 256)
    self.attention2 = AttentionGate(256, 256, 128)
    self.attention3 = AttentionGate(128, 128, 64)
    self.up1 = self.up_conv(1024, 512)
    self.up2 = self.up_conv(512, 256)
    self.up3 = self.up_conv(256, 128)
    self.up4 = self.up_conv(128, 64)
    self.decoder1 = self.conv_block(1024, 512)
    self.decoder2 = self.conv_block(512, 256)
    self.decoder3 = self.conv_block(256, 128)
    self.decoder4 = self.conv_block(128, 64)
    self.final = nn.Conv1d(64, out_channels, kernel_size=1)

  def conv_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm1d(out_channels),
      nn.ReLU(inplace=True))

  def up_conv(self, in_channels, out_channels):
      return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

  def max_pool(self, x):
    return F.max_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)

  def match_length(self, target, source):
    if source.size(2) < target.size(2):
      target = target[..., :source.size(-1)]
    elif source.size(2) > target.size(2):
      target = F.pad(target, (0, source.size(2) - target.size(2)))
    return target

  def forward(self, x):
    e1 = self.encoder1(x)
    e2 = self.encoder2(self.max_pool(e1))
    e3 = self.encoder3(self.max_pool(e2))
    e4 = self.encoder4(self.max_pool(e3))

    b = self.bottleneck(self.max_pool(e4))

    d1 = self.up1(b)
    d1 = self.match_length(d1, e4)
    e4 = self.attention1(d1, e4)
    d1 = self.decoder1(torch.cat((d1, e4), dim=1))

    d2 = self.up2(d1)
    d2 = self.match_length(d2, e3)
    e3 = self.attention2(d2, e3)
    d2 = self.decoder2(torch.cat((d2, e3), dim=1))

    d3 = self.up3(d2)
    d3 = self.match_length(d3, e2)
    d2 = self.attention3(d3, e2)
    d3 = self.decoder3(torch.cat((d3, e2), dim=1))

    d4 = self.up4(d3)
    d4 = self.match_length(d4, e1)
    d4 = self.decoder4(torch.cat((d4, e1), dim=1))

    f = self.final(d4)
    f = self.match_length(f, x)
    return f


class PatchGANDiscriminator(nn.Module):
  """
  The PatchGAN Discriminator is a type of discriminator used in Generative Adversarial Networks
  (GANs), particularly for image processing tasks. It was introduced in the pix2pix paper for
  image-to-image translation. Instead of evaluating the entire image as a whole to decide whether
  it's real or fake, the PatchGAN discriminator breaks the image down into smaller patches and
  classifies each patch as real or fake. This local-level discrimination allows the PatchGAN
  to focus on texture-level or fine-grained details.

  The discriminator should be conditioned on the same external information as the generator (e.g.,
  additional signals or features) to ensure the generated output is not only realistic but also
  aligned with the conditioning data. The simplest way to condition the discriminator is by
  concatenating the conditioning information with the input signals along the channel dimension.
  More advanced techniques like embedding the conditioning signal or using adaptive normalization
  can also be applied, depending on the complexity of the problem.
  """
  def __init__(self, in_channels, condition_channels, n_filters):
    super(PatchGANDiscriminator, self).__init__()

    # The discriminator will take both the input and the condition concatenated along the channel
    # dimension.
    self.model = nn.Sequential(
        nn.Conv1d(in_channels + condition_channels, n_filters, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(n_filters, n_filters * 2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(n_filters * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(n_filters * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(n_filters * 4, n_filters * 8, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm1d(n_filters * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(n_filters * 8, 1, kernel_size=4, stride=1, padding=1)
    )

  def forward(self, x):
    return self.model(x)


def compute_gradient_penalty(discriminator, scg, real_pap, fake_pap):
  """
  Adding a gradient penalty is a common technique used in training Wasserstein GANs with a
  discriminator. The gradient penalty is used to enforce the Lipschitz continuity of the
  discriminator by penalizing gradients with norms that deviate from 1. This is known as the
  WGAN-GP (Wasserstein GAN with Gradient Penalty). The gradient penalty is computed by sampling 
  random points along the straight lines between real and generated samples. The penalty term 
  encourages the gradient norm of the critic (discriminator) output with respect to the input to 
  be 1.
  """

  batch_size = real_pap.size(0)

  # Random weight term for interpolation between real and fake samples. Interpolation is a
  # technique to estimate the values of unknown data points that fall in btween existing, known
  # data points.
  alpha = torch.rand(real_pap.size(0), 1, 1).to(real_pap.device)

  # Get random interpolation btween real and fake samples.
  interpolates = (alpha * real_pap + ((1 - alpha) * fake_pap)).requires_grad_(True)

  # Forward pass through the discriminator.
  d_interpolates = discriminator(torch.cat((scg, interpolates), dim=1))

  # Gradient penalty computation
  fake = torch.ones(d_interpolates.size(), requires_grad=False).to(real_pap.device)

  # Compute gradients of the discriminator w.r.t interpolates.
  gradients = autograd.grad(
    outputs=d_interpolates,
    inputs=interpolates,
    grad_outputs=fake,
    create_graph=True,
    retain_graph=True,
    only_inputs=True
  )[0]

  # Compute the gradient norm.
  gradients = gradients.view(batch_size, -1)
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

  return gradient_penalty


def run_conditional_GAN():
  """
  A Conditional Generative Adversarial Network (cGAN) is a type of GAN where both the generator
  and discriminator are conditioned on additional information. This extra information can be
  anything that guides the generation process, such as class labels, images, waveforms, or any
  auxiliary input. In the case of waveforms, this could be a separate input waveform or label that
  guides the generation process.

  In the provided code, the conditional aspect is implemented by concatenating the input data
  (which can be the waveform) with the label or condition at different points:

  1. In the Generator: The input data (waveform) is passed to the generator, which learns to
  generate output based on it. However, in the code below, the conditioning of the generator is
  implicit through the input. A more explicit conditioning could involve concatenating the
  input waveform with the condition/label and passing it through the generator.

  2. In the Discriminator: The generator's output is concatenated with the input waveform before
  passing it through the discriminator. This allows the discriminator to evaluate whether the
  generated waveform matches the input condition.

  The original loss function used in the paper by Meng et al incorporated the Wasserstein distance
  with a gradient penalty, as well as the Euclidean distance between the reconstructed signals and
  the ground truth.

  Wasserstein loss (WGAN) is used to stabilize GAN training by improving gradient flow and avoiding 
  issues like vanishing gradients that can occur with binary cross-entropy loss (BCE).

  - Discriminator loss: The discriminator (also called a critic in WGAN) aims to output a real
  score (not bounded between 0 and 1). Instead of classifying real/fake as in traditional GAN, it
  outputs a continuous score, and we compute the difference between the critic's scores for real
  and fake samples.

  - Generator loss: The generator tries to maximize the critic's score for the fake samples
  (increasing the "realness" of the generated data).

  Euclidean loss (L2 loss) is often used for reconstruction tasks to measure the difference between
  the generated output and the ground truth. It computes the squared difference between the
  predicted output and the target.

  A gradient penalty is also used to enforce the Liipschitz continuity of the discriminator by
  penalizing gradients with norms that deviate from 1.
  """
  segment_size = 375
  segments = get_scg_rhc_segments(segment_size)
  train_and_validation_segments, test_segments = train_test_split(segments, train_size=0.9)
  train_segments, validation_segments = train_test_split(train_and_validation_segments, train_size=0.9)
  train_set = SCGDataset(train_segments, segment_size)

  lambda_gp = 10
  num_epochs = 10
  train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
  generator = AttentionUNetGenerator(in_channels=3, out_channels=1)
  discriminator = PatchGANDiscriminator(in_channels=3, condition_channels=1, n_filters=64)
  optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
  optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
  criterion_L2 = nn.MSELoss()

  G_losses = []
  D_losses = []

  for epoch in range(num_epochs):

    for i, (scg, pap) in enumerate(train_loader):

      # Train generator
      optimizer_G.zero_grad()
      fake_pap = generator(scg)
      fake_validity = discriminator(torch.cat((scg, fake_pap), dim=1))

      # Wasserstein loss for generator (maximize critic's score for fake)
      g_loss_wasserstein = -torch.mean(fake_validity)

      # Euclidean loss (L2) between generated and real PAP waveforms
      g_loss_l2 = criterion_L2(fake_pap, pap)

      # Combined generator loss
      g_loss = g_loss_wasserstein + 100 * g_loss_l2
      G_losses.append(g_loss.item())
      g_loss.backward()

      # Apply gradient clipping to limit the magnitude of gradients
      torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)

      optimizer_G.step()

      # Train discriminator (critic)
      optimizer_D.zero_grad()
      real_validity = discriminator(torch.cat((scg, pap), dim=1))
      fake_pap = generator(scg)
      fake_validity = discriminator(torch.cat((scg, fake_pap.detach()), dim=1))
      d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
      gradient_penalty = compute_gradient_penalty(discriminator, scg, pap, fake_pap)
      d_loss_total = d_loss + lambda_gp * gradient_penalty
      D_losses.append(d_loss_total.item())
      d_loss_total.backward()

      # Apply gradient clipping to limit the magnitude of gradients
      torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)

      optimizer_D.step()

      # Print progress
      print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}]')
      print(f'   G Loss: {g_loss.item():.4f}')
      print(f'   D Loss Total: {d_loss_total.item():.4f}')

    # Save model checkpoints every epoch
    checkpoint = {
      'epoch': epoch,
      'generator_state_dict': generator.state_dict(),
      'discriminator_state_dict': discriminator.state_dict(),
      'optimizer_G_state_dict': optimizer_G.state_dict(),
      'optimizer_D_state_dict': optimizer_D.state_dict(),
      'g_loss': g_loss,
      'd_loss_total': d_loss_total
    }
    torch.save(checkpoint, 'checkpoint.pth')
    print('Saved checkpoint')

  # Plot the losses after training
  plt.plot(G_losses, label='Generator Loss')
  plt.plot(D_losses, label='Discriminator Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Generator and Discriminator Loss During Training')
  plt.legend()
  plt.show()
