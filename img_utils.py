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

This file contains image utility functions.
"""

from typing import List, Union, Dict, Optional, Tuple, Any

import torch
import numpy as np
import os
import cv2
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
from pytorch_msssim import ms_ssim as ms_ssim_metric
import rawpy
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007


def clip(x: np.ndarray, min_v: Optional[float] = 0.0,
         max_v: Optional[float] = 1.0) -> np.ndarray:
  """Limits the values in x by min_v and max_v."""
  return np.clip(x, a_min=min_v, a_max=max_v).astype(np.float32)


def imwrite(image: np.ndarray, output_path: str, format: str, quality: Optional[int]=95, comment: Optional[
  Union[str, bytes]]=None
            ) -> str:
  """Saves an image to a file in the specified format.

  Args:
    image: An array representing the image (height x width x channel).
    output_path: File path without extension where the image will be saved.
    format: The desired file format: 'PNG-16', 'PNG-8', or 'JPEG'.
    quality: JPEG quality level (0-100), default is 95.
    comment: Optional JPEG comment to embed (must be ASCII, max ~64KB). Can be str or bytes.

  Raises:
    ValueError: If the specified format is invalid.
  """
  format = format.lower()
  image = clip(image)
  ext_map = {
    'png-16': ('.png', np.uint16, 16),
    'png-8': ('.png', np.uint8, 8),
    'jpeg': ('.jpg', np.uint8, 8),
    'jpg': ('.jpg', np.uint8, 8),
  }

  if format in ext_map:
    ext, dtype, bit_depth = ext_map[format]
    image = (image * (2 ** bit_depth - 1)).astype(dtype)
  else:
    raise ValueError(f"Format '{format}' is not supported.")

  output_file = f"{os.path.splitext(output_path)[0]}{ext}"
  if format in ['jpeg', 'jpg']:
    if comment is None:
      cv2.imwrite(output_file, convert_bgr_rgb(image), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
      pil_img = Image.fromarray(image)
      if isinstance(comment, str):
        comment_bytes = comment.encode('ascii', errors='ignore')
      else:
        comment_bytes = comment
      pil_img.save(output_file, format='JPEG', quality=quality, comment=comment_bytes)

  else:
    cv2.imwrite(output_file, convert_bgr_rgb(image))

  return output_file

def im2double(img: np.ndarray) -> np.ndarray:
  """ Converts image to floating-point format [0-1]."""
  if img[0].dtype == 'uint8':
    max_value = 255
  elif img[0].dtype == 'uint16':
    max_value = 65535
  else:
    raise ValueError
  return img.astype('float') / max_value


def imread(img_file: str, single_channel: Optional[bool] = False, normalize: Optional[bool] = True,
           load_comment: Optional[bool] = False) -> Union[np.ndarray, Tuple[np.ndarray, bytes]]:
  """Reads RGB image file with optional normalization and comment extraction (JPEG only)."""
  if not os.path.exists(img_file):
    raise FileNotFoundError(f'Image not found: {img_file}')
  ext = os.path.splitext(img_file)[-1].lower()
  if load_comment and ext in ['.jpg', '.jpeg']:
    img = Image.open(img_file)
    comment = img.info.get('comment', b"")
    img = np.array(img)
    if img.ndim == 2 and not single_channel:
      img = np.stack([img] * 3, axis=-1)
  else:
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    comment = None
    if img is None:
      raise FileNotFoundError(f'Cannot load image: {img_file}')
    if not single_channel:
      img = convert_bgr_rgb(img)
  if normalize:
    img = im2double(img)
  return (img, comment) if load_comment else img

def img_to_tensor(img):
  """Converts a given ndarray image to torch tensor image."""
  dims = len(img.shape)
  assert (dims == 3 or dims == 4)
  if dims == 3:
    img = img.transpose((2, 0, 1))
  elif dims == 4:
    img = img.transpose((3, 2, 0, 1))
  else:
    raise NotImplementedError
  return torch.from_numpy(img)

def tensor_to_img(img):
  """Converts a given torch tensor image to an ndarray image."""
  dims = len(img.shape)
  assert (dims == 3 or dims == 4)
  if dims == 3:
    img = img.permute(1, 2, 0)
  elif dims == 4:
    img = img.permute(2, 3, 1, 0).squeeze(dim=-1)
  else:
    raise NotImplementedError
  return img.detach().cpu().numpy()

def convert_bgr_rgb(img: np.ndarray) -> np.ndarray:
  """Converts BGR/RGB image to RGB/BGR image."""
  return img[..., ::-1]

def shift_image(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
  """Shifts an image by a given amount along the x and y axes.

  Args:
    img: Input image as a NumPy array (H x W x C).
    shift_x: Number of pixels to shift along the x-axis. Positive values shift right, negative left.
    shift_y: Number of pixels to shift along the y-axis. Positive values shift down, negative up.

  Returns:
    A shifted version of the input image.
  """
  h, w, c = img.shape
  translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
  shifted_img = cv2.warpAffine(img, translation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
  return shifted_img

def augment_img(img: np.ndarray) -> np.ndarray:
  """Applies simple augmentations: horizontal/vertical flip and random shift."""
  if random.random() < 0.5:
    img = np.flip(img, axis=0)
  if random.random() < 0.5:
    img = np.flip(img, axis=1)
  max_shift = 5
  shift_x = random.randint(-max_shift, max_shift)
  shift_y = random.randint(-max_shift, max_shift)
  img = shift_image(img, shift_x, shift_y)
  return img


def get_ssim(source: np.ndarray, reference: np.ndarray) -> float:
  """Computes the SSIM between two color images."""
  return structural_similarity(source, reference, multichannel=True,
                               channel_axis=2, data_range=1)

def get_psnr(source: np.ndarray, reference: np.ndarray) -> float:
  """Computes PSNR between two color images."""
  return peak_signal_noise_ratio(source, reference)

def get_ms_ssim(x: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]):
  """Computes multi-scale SSIM."""
  if isinstance(x, np.ndarray):
    x = img_to_tensor(x).unsqueeze(0)
  if isinstance(y, np.ndarray):
    y = img_to_tensor(y).unsqueeze(0)
  ms_ssim_score = ms_ssim_metric(x, y, data_range=1., size_average=True).item()
  return ms_ssim_score

def extract_non_overlapping_patches(img: np.ndarray, gt_img: Optional[np.ndarray] = None,
                                    patch_size: Optional[int] = 228, num_patches: Optional[int] = 0,
                                    allow_overlap: Optional[bool] = False) -> Dict[str, List[np.ndarray]]:
  """
  Extracts non-overlapping patches from an input image and optionally from a ground-truth image.
  If allow_overlap is True, extra patches are extracted near the borders to cover the full image.

  Args:
    img: Input image of shape (height, width, [channels]).
    gt_img: Optional ground-truth image with the same dimensions as 'img'.
    patch_size: Size of square patches (width = height).
    num_patches: Number of patches to extract. If set to 0, all valid patches are returned.
    allow_overlap: If True, adds extra overlapping patches to cover remaining border regions.

  Returns:
    A dictionary containing:
      - 'img': A list of extracted patches from 'img'.
      - 'xy': A list of (x, y) coordinates for the top-left corner of each extracted patch.
      - 'gt': A list of extracted patches from 'gt_img' (if provided).
  """
  h, w = img.shape[:2]

  if gt_img is not None:
    assert gt_img.shape[:2] == (h, w), 'Input image and ground-truth image must have the same dimensions.'
  y_steps = list(range(0, h - patch_size + 1, patch_size))
  x_steps = list(range(0, w - patch_size + 1, patch_size))
  if allow_overlap:
    if h % patch_size != 0:
      y_steps.append(h - patch_size)
    if w % patch_size != 0:
      x_steps.append(w - patch_size)
  patch_positions = [(y, x) for y in y_steps for x in x_steps]
  if num_patches == 0 or num_patches >= len(patch_positions):
    chosen_positions = patch_positions
  else:
    chosen_positions = random.sample(patch_positions, num_patches)
  img_patches = [img[y:y + patch_size, x:x + patch_size, ...] for y, x in chosen_positions]
  patch_coords = [(x, y) for y, x in chosen_positions]
  patches = {'img': img_patches, 'xy': patch_coords}
  if gt_img is not None:
    gt_patches = [gt_img[y:y + patch_size, x:x + patch_size, ...] for y, x in chosen_positions]
    patches['gt'] = gt_patches
  return patches

def extract_image_from_dng(file: str) -> np.ndarray:
  """Extracts raw image from a DNG file."""
  raw = rawpy.imread(file)
  return raw.raw_image_visible.astype(np.float32)


def demosaice(img: np.ndarray, cfa_pattern: str) -> np.ndarray:
  """Performs image demosaicing to input image.

  Args:
    img: Input Bayer image.
    cfa_pattern: Color filter array (CFA) pattern (e.g., 'RGGB')

  Returns:
    RGB image after demosaicing.
  """
  cfa_pattern = cfa_pattern.upper()
  img_rgb = demosaicing_CFA_Bayer_Menon2007(img, cfa_pattern)
  return clip(img_rgb)

def extract_raw_metadata(file: str) -> Dict[str, Any]:
  """Extracts DNG metadata."""
  def get_pattern(
          pattern: str,
          raw_pattern: Optional[Union[List, np.ndarray]]=None) -> str:
    if raw_pattern is None:
      return pattern
    return ''.join([pattern[i] for i in np.array(raw_pattern).flatten()])

  with rawpy.imread(file) as raw:
    return {'black_level': raw.black_level_per_channel,
            'white_level': float(raw.white_level),
            'pattern': get_pattern(raw.color_desc.decode('utf-8'),  raw.raw_pattern),
            'raw_pattern': raw.raw_pattern,
            }


def normalize_raw(img: np.ndarray, black_level: Union[List, np.ndarray],
                  white_level: float) -> np.ndarray:
  """Normalizes raw image using black level and white level values.

  Args:
    img: Demosaiced/mosaiced RGB raw image in the format (height x width) or (height x width x 4).
    black_level: 4D vector of black levels.
    white_level: A scalar value of white level.

  Returns:
    Normalized image after black level correction (BLC).
  """
  raw = img.astype(np.float32)
  if raw.shape[-1] == 4:
      raw = (raw - black_level) / white_level
  else:
    height, width = raw.shape
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(len(black_level)):
        raw[idx[i][0]:height:2, idx[i][1]:width:2] = (raw[idx[i][0]:height:2,
                                                      idx[i][1]:width:2] - black_level[i]
                                                      ) / white_level
  return raw

def s_curve(x: np.ndarray, contrast: Optional[int]=6, midpoint: Optional[float]=0.5) -> np.ndarray:
  """S-shaped tone mapping curve using a sigmoid function."""
  x = clip(x)
  y = 1 / (1 + np.exp(-contrast * (x - midpoint)))
  y_min = 1 / (1 + np.exp(contrast * midpoint))
  y_max = 1 / (1 + np.exp(-contrast * (1 - midpoint)))
  return (y - y_min) / (y_max - y_min)

def visualization(x: np.ndarray) -> np.ndarray:
  return s_curve(x) ** (1/2.2)