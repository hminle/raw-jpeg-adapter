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

This file contains loss functions.
"""

from typing import Optional, Dict, Tuple, Union
import torch.nn.functional as F
import torch
from constants import *


def fft_loss(pred_img: torch.Tensor, gt_img: torch.Tensor) -> torch.Tensor:
  """FFT loss."""
  pred_fft = torch.fft.fft2(pred_img, norm='ortho')
  gt_fft = torch.fft.fft2(gt_img, norm='ortho')

  loss_real = F.l1_loss(pred_fft.real, gt_fft.real)
  loss_imag = F.l1_loss(pred_fft.imag, gt_fft.imag)

  return loss_real + loss_imag

def ssim_loss(pred_img: torch.Tensor, gt_img: torch.Tensor, window_size: Optional[int]=11,
              c1: Optional[float]=0.01 ** 2, c2: Optional[float]=0.03 ** 2) -> torch.Tensor:
  """Differentiable SSIM loss."""

  def create_window(window_sz: int, chs: int):
    coords = torch.arange(window_sz).float() - window_sz // 2
    gauss = torch.exp(-(coords ** 2) / 2.0)
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] @ gauss[None, :]
    window = window_2d.expand(chs, 1, window_sz, window_sz).contiguous()
    return window

  channel = pred_img.shape[1]
  window = create_window(window_size, channel).to(pred_img.device)

  mu1 = F.conv2d(pred_img, window, padding=window_size // 2, groups=channel)
  mu2 = F.conv2d(gt_img, window, padding=window_size // 2, groups=channel)

  mu1_sq = mu1.pow(2)
  mu2_sq = mu2.pow(2)
  mu1_mu2 = mu1 * mu2

  sigma1_sq = F.conv2d(pred_img * pred_img, window, padding=window_size // 2, groups=channel) - mu1_sq
  sigma2_sq = F.conv2d(gt_img * gt_img, window, padding=window_size // 2, groups=channel) - mu2_sq
  sigma12 = F.conv2d(pred_img * gt_img, window, padding=window_size // 2, groups=channel) - mu1_mu2

  ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
             ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

  return 1 - ssim_map.mean()

def compute_loss(reconstructed_img: torch.Tensor, gt_img: torch.Tensor,
                 l1_loss_weight: Optional[float]=0,
                 fft_loss_weight: Optional[float]=0, ssim_loss_weight: Optional[float]=0,
                 ) -> Tuple[Union[torch.Tensor, float], Dict[str, float]]:
  """Computes loss."""
  detailed_loss = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'fft': 0.0, 'psnr': 0.0}
  total_loss = 0.0

  if l1_loss_weight > 0:
    l1 = l1_loss_weight * F.l1_loss(reconstructed_img, gt_img)
    total_loss += l1
    detailed_loss['l1'] = l1.item()

  if fft_loss_weight > 0:
    fft_loss_value = fft_loss_weight * fft_loss(reconstructed_img, gt_img)
    total_loss += fft_loss_value
    detailed_loss['fft'] = fft_loss_value.item()

  if ssim_loss_weight > 0:
    ssim_loss_value = ssim_loss_weight * ssim_loss(reconstructed_img, gt_img)
    total_loss += ssim_loss_value
    detailed_loss['ssim'] = ssim_loss_value.item()

  detailed_loss['total'] = total_loss.item()

  mse = torch.mean((reconstructed_img - gt_img) ** 2)
  if mse == 0:
    mse += EPS
  detailed_loss['psnr'] = - 10 * torch.log10(mse)

  return total_loss, detailed_loss
