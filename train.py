"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

This file contains the training script for the Raw-JPEG Adapter model.
"""

import argparse
import logging
import os
import io
import sys

import matplotlib.pyplot as plt
import tensorboard.summary
from tabulate import tabulate
import shutil
import random

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Union, Optional, List
from file_utils import write_json_file

from dataset import Data
from raw_jpeg_adapter_model import JPEGAdapter
from jpeg_utils import Quantization, DiffJPEG
from loss_utils import compute_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from constants import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None




def visualize_map(x):
  """Improves visualization of gamma map."""
  with torch.no_grad():
    b, _, _, _ = x.shape
    maps_flat = x.reshape(b, -1)
    mean = maps_flat.mean(dim=1, keepdim=True)
    std = maps_flat.std(dim=1, keepdim=True) + EPS
    norm_maps = (maps_flat - mean) / std
    norm_maps = norm_maps.view_as(x)
    norm_maps *= mean.view(b, 1, 1, 1)
    batch_min = norm_maps.min()
    batch_max = norm_maps.max()
    vis_maps = (norm_maps - batch_min) / (batch_max - batch_min + EPS)
    vis_maps = torch.nan_to_num(vis_maps, nan=0.0, posinf=1.0, neginf=0.0)
    return vis_maps

def visualize_image(x):
  """Applies 2.2 gamma to improve visualization."""
  return torch.nan_to_num(x ** (1/2.2), nan=0.0, posinf=1.0, neginf=0.0)

def visualize_lut(writer: SummaryWriter, luts: torch.Tensor, global_step: int, tag: Optional[str]='LUT'):
  """Tensorboard visualization of LuT"""
  b, c, k1 = luts.shape
  num_samples = min(b, 3)  # Show up to 3 samples
  indices = random.sample(range(b), num_samples)
  labels = ['Red', 'Green', 'Blue']
  for i, idx in enumerate(indices):
    fig, ax = plt.subplots()
    for c_j in range(c):
      y_vals = luts[idx, c_j, :].detach().cpu().numpy()
      ax.plot(y_vals, label=labels[c_j])
    ax.set_title(f'{tag} Sample {i}')
    ax.set_xlabel('Input Bin')
    ax.set_ylabel('Output Value')
    ax.set_ylim(0, 1)
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    img = Image.open(buf).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    writer.add_image(f'{tag}/sample_{i}', img_tensor, global_step)

def print_line(end: Optional[bool]=False, length: Optional[int]=30):
  """Prints a separator line."""
  line = '-' * length
  if end:
    print(f"{line}\n")
  else:
    print(f"\n{line}")

def training(model: JPEGAdapter, epochs: int, lr: float, l2_reg: float, tr_device: torch.device,
             train_loader: DataLoader, val_loader: DataLoader, train: Data, global_step: List[int],
             validation_frequency: int, exp_name: str, val_quantization: DiffJPEG,
             quantization: Union[Quantization, DiffJPEG], batch_size: int, l1_loss_weight: float,
             fft_loss_weight: float, ssim_loss_weight: float, writer: tensorboard.summary.Writer,
             log: Dict, debugging: Optional[bool]=False):
  """Performs training on the given dataloaders."""

  optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2_reg)
  scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

  for epoch in range(epochs):
    model.train()

    epoch_loss = 0
    with tqdm(total=len(train) * batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
      for i, batch in enumerate(train_loader):
        if debugging and i > 10:
          break
        images = batch['images'].squeeze(0)
        in_images = images.to(device=tr_device, non_blocking=True)
        gt_images = images.to(device=tr_device, non_blocking=True)
        proc_images, operator_params = model(in_images)
        jpeg_images = quantization(proc_images)
        rec_images = model.reconstruct_raw(
          raw_jpeg=jpeg_images, gamma_map=operator_params['gamma_map'],
          scale_dct=operator_params['scale_dct'],
          lut=operator_params['lut'])
        loss, detailed_loss = compute_loss(reconstructed_img=rec_images, gt_img=gt_images,
                                           l1_loss_weight=l1_loss_weight,
                                           fft_loss_weight=fft_loss_weight,
                                           ssim_loss_weight=ssim_loss_weight)
        epoch_loss += loss.item()

        if writer:
          writer.add_scalar(f'Loss (x{JPEG_ADAPT_LOSS_SCALE})/train', JPEG_ADAPT_LOSS_SCALE * loss.item(),
                            global_step[0])
          writer.add_scalar(f'L1 (x{JPEG_ADAPT_LOSS_SCALE})/train', JPEG_ADAPT_LOSS_SCALE * detailed_loss['l1'],
                            global_step[0])
          writer.add_scalar(f'PSNR/train', detailed_loss['psnr'], global_step[0])
          writer.add_scalar(f'SSIM (x{JPEG_ADAPT_LOSS_SCALE})/train', JPEG_ADAPT_LOSS_SCALE * detailed_loss['ssim'],
                            global_step[0])
          writer.add_scalar(f'FFT (x{JPEG_ADAPT_LOSS_SCALE})/train', JPEG_ADAPT_LOSS_SCALE * detailed_loss['fft'],
                            global_step[0])

        pbar.set_postfix({
          f'Batch-loss (x{JPEG_ADAPT_LOSS_SCALE})': f'{JPEG_ADAPT_LOSS_SCALE * detailed_loss["total"]:.4f} '
                                                    f'- L1 (x{JPEG_ADAPT_LOSS_SCALE}) = '
                                                    f'{JPEG_ADAPT_LOSS_SCALE * detailed_loss["l1"]:.4f}, '
                                                    f'PSNR = {detailed_loss["psnr"]:.4f}, '
                                                    f'SSIM (x{JPEG_ADAPT_LOSS_SCALE}) = '
                                                    f'{JPEG_ADAPT_LOSS_SCALE * detailed_loss["ssim"]:.4f}, '
                                                    f'FFT (x{JPEG_ADAPT_LOSS_SCALE}) = '
                                                    f'{JPEG_ADAPT_LOSS_SCALE * detailed_loss["fft"]:.4f}'
        })

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update(np.ceil(images.shape[0]))
        global_step[0] += 1

    if (epoch + 1) % validation_frequency == 0:
      val_loss = validate(model=model, loader=val_loader, val_device=tr_device, l1_loss_scale=l1_loss_weight,
                          fft_loss_scale=fft_loss_weight, ssim_loss_scale=ssim_loss_weight,
                          val_quantization=val_quantization, writer=writer, global_step=global_step[0],
                          debugging=debugging)
      print_line()
      logging.info(
        f'Validation loss (x{JPEG_ADAPT_LOSS_SCALE}): {JPEG_ADAPT_LOSS_SCALE * val_loss["total"]:.4f} - '
        f'L1 (x{JPEG_ADAPT_LOSS_SCALE}) = {JPEG_ADAPT_LOSS_SCALE * val_loss["l1"]:.4f}, '
        f'SSIM (x{JPEG_ADAPT_LOSS_SCALE}) = {JPEG_ADAPT_LOSS_SCALE * val_loss["ssim"]:.4f}, '
        f'FFT (x{JPEG_ADAPT_LOSS_SCALE}) = {JPEG_ADAPT_LOSS_SCALE * val_loss["fft"]:.4f}, '
        f'PSNR = {val_loss["psnr"]:.4f}\n'
      )

      checkpoint_model_name = os.path.join('checkpoints', f'{exp_name}_{epoch + 1}.pth')
      torch.save(model.state_dict(), checkpoint_model_name)
      logging.info(f'Checkpoint {epoch + 1} saved!')
      print_line(end=True)

      log['checkpoint_model_name'].append(checkpoint_model_name)
      log['val_psnr'].append(
        val_loss['psnr'].item() if isinstance(val_loss['psnr'], torch.Tensor) else val_loss['psnr'])
      write_json_file(log, os.path.join('logs', f'{exp_name}'))

      if writer:
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step[0])
        writer.add_scalar(f'Loss (x{JPEG_ADAPT_LOSS_SCALE})/val', JPEG_ADAPT_LOSS_SCALE * val_loss["total"],
                          global_step[0])
        writer.add_scalar(f'L1 (x{JPEG_ADAPT_LOSS_SCALE})/val', JPEG_ADAPT_LOSS_SCALE * val_loss['l1'],
                          global_step[0])
        writer.add_scalar(f'SSIM (x{JPEG_ADAPT_LOSS_SCALE})/val', JPEG_ADAPT_LOSS_SCALE * val_loss['ssim'],
                          global_step[0])
        writer.add_scalar(f'FFT (x{JPEG_ADAPT_LOSS_SCALE})/val', JPEG_ADAPT_LOSS_SCALE * val_loss['fft'],
                          global_step[0])
        writer.add_scalar(f'PSNR/val', val_loss['psnr'], global_step[0])

        writer.add_images('Images (2.2 gamma)/train', visualize_image(images), global_step[0])
        writer.add_images('Rec images (2.2 gamma)/train', visualize_image(rec_images), global_step[0])
        writer.add_images('Proc images (2.2 gamma)/train', visualize_image(proc_images), global_step[0])
        if operator_params['gamma_map'] is not None:
          writer.add_images('Gamma map/train', visualize_map(operator_params['gamma_map']), global_step[0])
        if operator_params['scale_dct'] is not None:
          writer.add_images('DCT scale/train', visualize_map(operator_params['scale_dct'].squeeze(2)), global_step[0])
        if operator_params['lut'] is not None:
          visualize_lut(writer=writer, luts=model.normalize_lut(operator_params['lut']),
                        global_step=global_step[0], tag='LuT/train')

    scheduler.step()

  torch.save(model.state_dict(), os.path.join('models', f'{exp_name}.pth'))
  logging.info('Saved trained model!')
  best_model_idx = log['val_psnr'].index(max(log['val_psnr']))
  best_model_name = log['checkpoint_model_name'][best_model_idx]
  shutil.copy(best_model_name, os.path.join('models', f'{exp_name}-best.pth'))

def train_net(model: JPEGAdapter, tr_device: torch.device, tr_dir: str, val_dir: str, epochs: int, batch_size: int,
              lr: float, l2_reg: float, patch_sz: int, validation_frequency: int, exp_name: str,
              num_patches_per_image: int, temp_folder: str, overwrite_temp_folder: bool, delete_temp_folder: bool,
              l1_loss_weight: float, fft_loss_weight: float, ssim_loss_weight: float, color_aug: bool, scale_aug: bool,
              use_jpeg_block: bool, jpeg_quality: int, debugging: bool):
  """Trains network."""

  print_line()
  print(f'Training on {patch_sz}x{patch_sz} patches...')
  print_line(end=True)
  if use_jpeg_block:
    quantization = DiffJPEG(quality=jpeg_quality).to(device=device)
    val_quantization = quantization.to(device=device)
  else:
    quantization = Quantization().to(device=device)
    val_quantization = DiffJPEG(quality=jpeg_quality).to(device=device)

  writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}') if SummaryWriter else None
  global_step = [0]

  train = Data(tr_dir, patch_size=patch_sz, num_patches_per_image=num_patches_per_image, temp_folder=temp_folder,
               overwrite_temp_folder=overwrite_temp_folder, color_aug=color_aug, intensity_aug=scale_aug,
               batch_size=batch_size, shuffle=True)
  val = Data(val_dir, patch_size=patch_sz, num_patches_per_image=num_patches_per_image, temp_folder=temp_folder,
             overwrite_temp_folder=overwrite_temp_folder, geometric_aug=False, batch_size=batch_size)
  train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
  val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

  log = {'checkpoint_model_name': [], 'val_psnr': []}

  training(model=model, epochs=epochs, lr=lr, l2_reg=l2_reg, tr_device=tr_device,
           train_loader=train_loader, val_loader=val_loader, train=train, global_step=global_step,
           validation_frequency=validation_frequency, exp_name=exp_name, val_quantization=val_quantization,
           quantization=quantization, batch_size=batch_size, l1_loss_weight=l1_loss_weight,
           fft_loss_weight=fft_loss_weight, ssim_loss_weight=ssim_loss_weight,
           writer=writer, log=log, debugging=debugging)

  if writer:
    writer.close()
  logging.info('End of training')

  if delete_temp_folder:
    logging.info('Deleting temp folders')
    tr_temp_dir = os.path.join(os.path.dirname(tr_dir), f'{temp_folder}_bs{batch_size}')
    val_temp_dir = os.path.join(os.path.dirname(val_dir), f'{temp_folder}_bs{batch_size}')
    shutil.rmtree(tr_temp_dir)
    if tr_temp_dir != val_temp_dir:
      shutil.rmtree(val_temp_dir)
    logging.info('Done!')


def validate(model: JPEGAdapter, loader: DataLoader, val_device: torch.device, ssim_loss_scale: float,
             l1_loss_scale: float, fft_loss_scale: float, val_quantization: Union[DiffJPEG, Quantization],
             writer: SummaryWriter, global_step: int, debugging: Optional[bool]=False) -> Dict[str, float]:
  """Network validation."""
  model.eval()
  n_val = len(loader) + 1
  val_loss = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'fft': 0.0, 'psnr': 0.0}
  with torch.no_grad():
    for i, batch in enumerate(loader):
      if debugging and i > 10:
        break
      images = batch['images'].squeeze(0)
      in_images = images.to(device=val_device, non_blocking=True)
      gt_images = images.to(device=val_device, non_blocking=True)
      proc_images, operator_params = model(in_images)
      rec_images = model.reconstruct_raw(val_quantization(proc_images),
                                         gamma_map=operator_params['gamma_map'],
                                         scale_dct=operator_params['scale_dct'], lut=operator_params['lut'])
      _, detailed_b_loss= compute_loss(rec_images, gt_images, l1_loss_weight=l1_loss_scale,
                                       fft_loss_weight=fft_loss_scale, ssim_loss_weight=ssim_loss_scale)
      for key in val_loss:
        val_loss[key] += detailed_b_loss[key]
  model.train()

  writer.add_images('Images (2.2 gamma)/val', visualize_image(images), global_step)
  writer.add_images('Rec images (2.2 gamma)/val', visualize_image(rec_images), global_step)
  writer.add_images('Proc images (2.2 gamma)/val', visualize_image(proc_images), global_step)
  if operator_params['gamma_map'] is not None:
    writer.add_images('Gamma map/val', visualize_map(operator_params['gamma_map']), global_step)
  if operator_params['scale_dct'] is not None:
    writer.add_images('DCT scale/val', visualize_map(operator_params['scale_dct'].squeeze(2)), global_step)
  if operator_params['lut'] is not None:
    visualize_lut(writer=writer, luts=model.normalize_lut(operator_params['lut']),
                  global_step=global_step, tag='LuT/val')

  for key in val_loss:
    val_loss[key] /= 10 if debugging else n_val
  return val_loss

def get_args():
  parser = argparse.ArgumentParser(description='Train Raw-JPEG-Adapter Net.')
  parser.add_argument('--training-dir', dest='tr_dir', help='Training raw image directory')
  parser.add_argument('--validation-dir', dest='vl_dir', help='Validation raw image directory')
  parser.add_argument('--epochs', type=int, default=100, dest='epochs')
  parser.add_argument('--batch-size', type=int, default=16, dest='batch_size')
  parser.add_argument('--learning-rate', type=float, default=0.001, dest='lr')
  parser.add_argument('--l2reg', type=float, default=0.0001, help='L2 Regularization factor', dest='l2_r')
  parser.add_argument( '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
  parser.add_argument('--validation-frequency', dest='val_frq', type=int, default=1)
  parser.add_argument('--patch-size', dest='patch_sz', type=int, default=512,
                      help='Size of training patches.')
  parser.add_argument('--jpeg-quality', dest='jpeg_quality', type=int, default=75,
                      help='JPEG quality to use during training. Range: 0 (low) to 100 (high). Default is 75.')
  parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=24,
                      help='Dimensionality of the latent features produced by the encoder.')
  parser.add_argument('--discard-eca', dest='discard_eca', action='store_true',
                      help='Do not use the ECA block.')
  parser.add_argument('--use-gamma', dest='use_gamma', action='store_true',
                      help='To use gamma map.')
  parser.add_argument('--use-scale-dct', dest='use_scale_dct', action='store_true',
                      help='To use 8x8 DCT scale.')
  parser.add_argument('--use-lut', dest='use_lut', action='store_true', help='To use LuT.')
  parser.add_argument('--debugging', dest='debugging', action='store_true', help='For debugging.')
  parser.add_argument('--lut-size', dest='lut_size', type=int, default=JPEG_ADAPT_LUT_BINS,
                      help='Number of bins in the LuT.')
  parser.add_argument('--lut-channels', dest='lut_channels', type=int, default=JPEG_ADAPT_LUT_CHS,
                      help='Number of channels in LuT.')
  parser.add_argument('--discard-jpeg-block', dest='discard_jpeg_block', action='store_true',
                      help='Do not use JPEG simulation block.')
  parser.add_argument('--map-size', dest='map_size', type=int, nargs=2,
                      default=[JPEG_ADAPT_MAP_SZ, JPEG_ADAPT_MAP_SZ],
                      help='Size of coefficients map (spatial domain) that is produced by the network. '
                           'Specify two integers: height and width.')
  parser.add_argument('--num-patches-per-image', dest='num_patches_per_image', type=int, default=0,
                      help='Number of patches to extract per image. If set to 0, all non-overlapping patches are used.')
  parser.add_argument('--temp-folder', dest='temp_folder', type=str, default='temp_patches',
                      help='Name of temporary folder to save extracted patches.')
  parser.add_argument('--overwrite-temp-folder', dest='overwrite_temp_folder', action='store_true',
                      help='Overwrite temp_folder if it exists.')
  parser.add_argument('--delete-temp-folder', dest='delete_temp_folder', action='store_true',
                      help='To delete temp_folder after training.')
  parser.add_argument('--l1-loss-weight', type=float, default=1.0,
                      help='Weight for L1 loss (set to 0 to disable).')
  parser.add_argument('--fft-loss-weight', type=float, default=0.1,
                      help='Weight for FFT loss (set to 0 to disable).')
  parser.add_argument('--ssim-loss-weight', type=float, default=0.1,
                      help='Weight for SSIM loss (set to 0 to disable).')
  parser.add_argument('--color-aug', action='store_true', help='Enable color augmentation.')
  parser.add_argument('--scale-aug', action='store_true', help='Enable scale augmentation.')
  return parser.parse_args()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  print(tabulate([(key, value) for key, value in vars(args).items()], headers=["Argument", "Value"], tablefmt="grid"))
  if not (args.use_gamma or args.use_scale_dct or args.use_lut):
    raise ValueError(f'At least one of the flags [use-gamma, use-scale-dct, use-lut] must be True.')

  if args.l1_loss_weight + args.fft_loss_weight + args.ssim_loss_weight == 0:
    raise ValueError(f'At least the weight(s) of one of the following loses should be > 0: [l1, fft, ssim].')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info(f'Using device {device}')

  model_name = 'raw_jpeg_adapter'
  model_name += (f'_q_{args.jpeg_quality}_{args.map_size[0]}x{args.map_size[1]}_l_{args.latent_dim}_'
                 f'l1_{args.l1_loss_weight}_fft_{args.fft_loss_weight}_ssim_{args.ssim_loss_weight}')
  if args.discard_eca:
    model_name += '_wo_eca'
  if args.use_gamma:
    model_name += '_w_gamma'
  if args.use_scale_dct:
    model_name += '_w_scale_dct'
  if args.use_lut:
    model_name += f'_w_lut_{args.lut_size}_{args.lut_channels}'
  if args.discard_jpeg_block:
      model_name += '_wo_jpeg_block'
  if args.color_aug:
    model_name += '_col_aug'
  if args.scale_aug:
    model_name += '_scale_aug'

  logging.info(f'Training of Raw-JPEG Net -- model name: {model_name} ...')

  os.makedirs('models', exist_ok=True)
  os.makedirs('config', exist_ok=True)
  os.makedirs('checkpoints', exist_ok=True)
  os.makedirs('logs', exist_ok=True)


  config = {'latent_dim': args.latent_dim,
            'map_size': args.map_size,
            'eca': not args.discard_eca,
            'gamma': args.use_gamma,
            'scale_dct': args.use_scale_dct,
            'lut': args.use_lut,
            'lut_size': args.lut_size,
            'lut_channels': args.lut_channels,
            'quality': args.jpeg_quality}

  write_json_file(config, os.path.join('config', f'{model_name}.json'))
  write_json_file(config, os.path.join('config', f'{model_name}-best.json'))

  net = JPEGAdapter(latent_dim=args.latent_dim, target_img_size=args.map_size, use_eca=not args.discard_eca,
                    quality=args.jpeg_quality, use_gamma=args.use_gamma,
                    use_scale_dct=args.use_scale_dct, use_lut=args.use_lut,
                    lut_size=args.lut_size, lut_channels=args.lut_channels)
  if not net.test_embedding_jpeg_comment():
    print('Current network configuration produces too large data for JPEG metadata.')
    sys.exit(1)
  if args.load:
    net.load_state_dict(torch.load(args.load, map_location=device, weights_only=True))
    logging.info(f'Model loaded from {args.load}')

  net.print_num_of_params()

  net.to(device=device)
  try:
    train_net(
      model=net,
      tr_device=device,
      tr_dir=args.tr_dir,
      val_dir=args.vl_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      lr=args.lr,
      l2_reg=args.l2_r,
      exp_name=model_name,
      validation_frequency=args.val_frq,
      patch_sz=args.patch_sz,
      num_patches_per_image=args.num_patches_per_image,
      temp_folder=args.temp_folder,
      l1_loss_weight=args.l1_loss_weight,
      fft_loss_weight=args.fft_loss_weight,
      ssim_loss_weight=args.ssim_loss_weight,
      use_jpeg_block=not args.discard_jpeg_block,
      jpeg_quality=args.jpeg_quality,
      color_aug=args.color_aug,
      scale_aug=args.scale_aug,
      overwrite_temp_folder=args.overwrite_temp_folder,
      delete_temp_folder=args.delete_temp_folder,
      debugging=args.debugging,
    )
  except KeyboardInterrupt:
    torch.save(net.state_dict(), 'interrupted_checkpoint.pth')
    logging.info('Saved interrupt checkpoint backup')
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
