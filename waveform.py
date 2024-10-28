import os
import wfdb
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import numpy as np

from explore import get_rhc_records

torch.autograd.set_detect_anomaly(True)


class SCGDataset(Dataset):
  def __init__(self, records, length):
    self.records = records
    self.length = length

  def _pad(self, tensor):
    if tensor.shape[-1] < self.length:
        padding = self.length - tensor.shape[-1]
        tensor = torch.nn.functional.pad(tensor, (0, padding))
    elif tensor.shape[-1] > self.length:
        tensor = tensor[:, :, :self.length]
    return tensor

  def _minmax_norm(self, signal):
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 0.0001)
    return signal

  def _get_channels(self, record, channel_names):
    indexes = [record.sig_name.index(name) for name in channel_names]
    channels = record.p_signal[:, indexes]
    return channels

  def __len__(self):
    return len(self.records)

  def __getitem__(self, index):
    record = wfdb.rdrecord(self.records[index])

    # Select patch ACG signals as input for neural net.
    scg_signal = self._get_channels(record, channel_names=['patch_ACC_lat', 'patch_ACC_hf', 'patch_ACC_dv'])
    scg_signal = self._minmax_norm(scg_signal)
    scg_tensor = torch.tensor(scg_signal.T, dtype=torch.float32).unsqueeze(0)
    scg_tensor = self._pad(scg_tensor)

    # Select patch RHC signals as labels for neural net.
    rhc_signal = self._get_channels(record, channel_names=['RHC_pressure'])
    rhc_signal = self._minmax_norm(rhc_signal)
    rhc_tensor = torch.tensor(rhc_signal.T, dtype=torch.float32).unsqueeze(0)
    rhc_tensor = self._pad(rhc_tensor)

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
    psi = self.relu(g1 + x1)
    psi = self.psi(psi)
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
    return nn.functional.max_pool1d(x, kernel_size=2, stride=2)

  def forward(self, x):
    e1 = self.encoder1(x)
    e2 = self.encoder2(self.max_pool(e1))
    e3 = self.encoder3(self.max_pool(e2))
    e4 = self.encoder4(self.max_pool(e3))

    b = self.bottleneck(self.max_pool(e4))

    d1 = self.up1(b)
    e4 = self.attention1(d1, e4)
    d1 = self.decoder1(torch.cat((d1, e4), dim=1))

    d2 = self.up2(d1)
    e3 = self.attention2(d2, e3)
    d2 = self.decoder2(torch.cat((d2, e3), dim=1))

    d3 = self.up3(d2)
    d2 = self.attention3(d3, e2)
    d3 = self.decoder3(torch.cat((d3, e2), dim=1))

    d4 = self.up4(d3)
    d4 = self.decoder4(torch.cat((d4, e1), dim=1))

    return self.final(d4)


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


def run_conditional_GAN(dataloader):
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
  generator = AttentionUNetGenerator(in_channels=3, out_channels=1)
  discriminator = PatchGANDiscriminator(in_channels=3, condition_channels=1, n_filters=64)
  optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
  optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
  criterion_L2 = nn.MSELoss()
  lambda_gp = 10
  num_epochs = 500

  for epoch in range(num_epochs):

    for i, (scg, pap) in enumerate(dataloader):

      # ---------------
      # Train generator
      # ---------------
      optimizer_G.zero_grad()
      fake_pap = generator(scg)
      fake_validity = discriminator(torch.cat((scg, fake_pap), dim=1))

      # Wasserstein loss for generator (maximize critic's score for fake)
      g_loss_wasserstein = -torch.mean(fake_validity)

      # Euclidean loss (L2) between generated and real PAP waveforms
      g_loss_l2 = criterion_L2(fake_pap, pap)

      # GAN loss
      # g_loss_gan = criterion_GAN(fake_validity, torch.ones_like(fake_validity))

      # L1 loss for better generation quality
      # g_loss_l1 = criterion_L1(fake_pap, pap)
      # g_loss = g_loss_gan + 100 * g_loss_l1
      # g_loss.backward(retain_graph=True)

      # Combined generator loss
      g_loss = g_loss_wasserstein + 100 * g_loss_l2
      g_loss.backward()

      # Apply gradient clipping to limit the magnitude of gradients
      torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)

      optimizer_G.step()

      # ----------------------------
      # Train discriminator (critic)
      # ----------------------------
      optimizer_D.zero_grad()
      real_validity = discriminator(torch.cat((scg, pap), dim=1))
      fake_pap = generator(scg)
      fake_validity = discriminator(torch.cat((scg, fake_pap.detach()), dim=1))
      d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
      gradient_penalty = compute_gradient_penalty(discriminator, scg, pap, fake_pap)
      d_loss_total = d_loss + lambda_gp * gradient_penalty
      d_loss_total.backward()

      # Apply gradient clipping to limit the magnitude of gradients
      torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)

      optimizer_D.step()

      # Print progress every 10 batches
      if i % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}]')
        print(f'   Generator Loss: {g_loss.item():.4f}')
        print(f'   Discriminator Loss: {d_loss.item():.4f}')

    # Save model checkpoints every 1 epoch
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


if __name__ == '__main__':
  records = get_rhc_records()
  dataset = SCGDataset(records, length=1024)
  run_conditional_GAN(dataset)
