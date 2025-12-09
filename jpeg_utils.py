"""
Adapted from: https://github.com/yzxing87/Invertible-ISP

Original code under MIT license. Minor modifications have been made to fit local requirements by:
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

This file contains utility functions for differentiable JPEG-like compression.
"""

import itertools
import math
import torch
import torch.nn as nn
from constants import *

y_table = nn.Parameter(torch.from_numpy(np.array(LUMINANCE_QUANTIZATION_TABLE, dtype=np.float32).T))
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array(CHROMA_QUANTIZATION_TABLE).T
c_table = nn.Parameter(torch.from_numpy(c_table))

class Quantization(nn.Module):
  def __init__(self):
    super(Quantization, self).__init__()

  @staticmethod
  def differentiable_quantize(x, rounding):
    """Simulate 8-bit quantization in a differentiable way."""
    x = torch.clamp(x, 0.0, 1.0)
    x_scaled = x * 255.0
    x_rounded = rounding(x_scaled)
    x_quantized = x_rounded / 255.0
    return x_quantized

  @staticmethod
  def diff_round(input_tensor):
    test = 0
    for n in range(1, 10):
      test += math.pow(-1, n + 1) / n * torch.sin(2 * math.pi * n * input_tensor)
    final_tensor = input_tensor - 1 / math.pi * test
    return final_tensor

  def forward(self, x):
    return self.differentiable_quantize(x, self.diff_round)

class DiffJPEG(nn.Module):
  def __init__(self, differentiable=True, quality=95, dtype=torch.float32):
    super(DiffJPEG, self).__init__()
    if differentiable:
      rounding = Quantization.diff_round
    else:
      rounding = torch.round
    factor = self._quality_to_factor(quality)
    self._compress = CompressJPEG(rounding=rounding, factor=factor, dtype=dtype)
    self._decompress = DecompressJPEG(factor=factor, dtype=dtype)

  def forward(self, x):
    org_height = x.shape[2]
    org_width = x.shape[3]
    y, cb, cr = self._compress(x)
    recovered = self._decompress(y, cb, cr, org_height, org_width)
    return recovered

  @staticmethod
  def _quality_to_factor(quality):
    """Calculate factor corresponding to quality."""
    if quality < 50:
      quality = 5000.0 / quality
    else:
      quality = 200.0 - quality * 2
    return quality / 100.0

class RGBToYCbCr(nn.Module):
  """Converts RGB image to YCbCr"""
  def __init__(self, dtype=torch.float32):
    super(RGBToYCbCr, self).__init__()
    matrix = np.array(RGB_TO_YCBCR, dtype=np.float32).T
    self._shift = nn.Parameter(torch.tensor([0., 128., 128.]).to(dtype=dtype))
    self._matrix = nn.Parameter(torch.from_numpy(matrix).to(dtype=dtype))

  def forward(self, image):
    image = image.permute(0, 2, 3, 1)
    result = torch.tensordot(image, self._matrix, dims=1) + self._shift
    result.view(image.shape)
    return result


class ChromaSubsampling(nn.Module):
  """Chroma subsampling on CbCv channels."""
  def __init__(self):
    super(ChromaSubsampling, self).__init__()

  def forward(self, image):
    image_2 = image.permute(0, 3, 1, 2).clone()
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                            count_include_pad=False)
    cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
    cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
    cb = cb.permute(0, 2, 3, 1)
    cr = cr.permute(0, 2, 3, 1)
    return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class BlockSplitting(nn.Module):
  """Splitting image into patches."""
  def __init__(self):
    super(BlockSplitting, self).__init__()
    self._k = 8

  def forward(self, image):
    height, width = image.shape[1:3]
    batch_size = image.shape[0]
    image_reshaped = image.view(batch_size, height // self._k, self._k, -1, self._k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, -1, self._k, self._k)


class DCT8x8(nn.Module):
  """Discrete Cosine Transformation."""
  def __init__(self):
    super(DCT8x8, self).__init__()
    tensor = np.zeros((8, 8, 8, 8))
    for x, y, u, v in itertools.product(range(8), repeat=4):
      tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
        (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    self._tensor = nn.Parameter(torch.from_numpy(tensor).float())
    self._scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

  def forward(self, image):
    image = image - 128
    result = self._scale * torch.tensordot(image, self._tensor, dims=2)
    result.view(image.shape)
    return result


class YQuantize(nn.Module):
  """JPEG Quantization for Y channel."""
  def __init__(self, rounding, factor=1):
    super(YQuantize, self).__init__()
    self._rounding = rounding
    self._factor = factor
    self._y_table = y_table

  def forward(self, image):
    image = image.float() / (self._y_table * self._factor)
    image = self._rounding(image)
    return image


class CQuantize(nn.Module):
  """JPEG Quantization for CrCb channels."""
  def __init__(self, rounding, factor=1):
    super(CQuantize, self).__init__()
    self._rounding = rounding
    self._factor = factor
    self._c_table = c_table

  def forward(self, image):
    image = image.float() / (self._c_table * self._factor)
    image = self._rounding(image)
    return image


class CompressJPEG(nn.Module):
  """JPEG compression."""

  def __init__(self, rounding=torch.round, factor=1, dtype=torch.float32):
    super(CompressJPEG, self).__init__()
    self._l1 = nn.Sequential(RGBToYCbCr(dtype), ChromaSubsampling())
    self._l2 = nn.Sequential(BlockSplitting(), DCT8x8())
    self._c_quantize = CQuantize(rounding=rounding, factor=factor)
    self._y_quantize = YQuantize(rounding=rounding, factor=factor)

  def forward(self, image):
    y, cb, cr = self._l1(image * 255)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
      comp = self._l2(components[k])
      if k in ('cb', 'cr'):
        comp = self._c_quantize(comp)
      else:
        comp = self._y_quantize(comp)
      components[k] = comp
    return components['y'], components['cb'], components['cr']


class LuminanceDequantize(nn.Module):
  """De-quantizes Y channel."""

  def __init__(self, factor=1):
    super(LuminanceDequantize, self).__init__()
    self._y_table = y_table
    self._factor = factor

  def forward(self, image):
    return image * (self._y_table * self._factor)


class ChromaDequantize(nn.Module):
  """De-quantizes CbCr channel."""

  def __init__(self, factor=1):
    super(ChromaDequantize, self).__init__()
    self._factor = factor
    self._c_table = c_table

  def forward(self, image):
    return image * (self._c_table * self._factor)


class IDCT8x8(nn.Module):
  """Inverse discrete Cosine Transformation."""

  def __init__(self, dtype):
    super(IDCT8x8, self).__init__()
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    self._alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).to(dtype=dtype))
    tensor = np.zeros((8, 8, 8, 8))
    for x, y, u, v in itertools.product(range(8), repeat=4):
      tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
        (2 * v + 1) * y * np.pi / 16)
    self._tensor = nn.Parameter(torch.from_numpy(tensor).to(dtype=dtype))

  def forward(self, image):
    image = image * self._alpha
    result = 0.25 * torch.tensordot(image, self._tensor.to(dtype=image.dtype), dims=2) + 128
    result.view(image.shape)
    return result


class BlockMerging(nn.Module):
  """Merges pathces into image."""

  def __init__(self):
    super(BlockMerging, self).__init__()

  def forward(self, patches, height, width):
    k = 8
    batch_size = patches.shape[0]
    image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.contiguous().view(batch_size, height, width)


class ChromaUpsampling(nn.Module):
  """Upsamples chroma layers."""

  def __init__(self):
    super(ChromaUpsampling, self).__init__()

  def forward(self, y, cb, cr):
    def repeat(x, k=2):
      height, width = x.shape[1:3]
      x = x.unsqueeze(-1)
      x = x.repeat(1, 1, k, k)
      x = x.view(-1, height * k, width * k)
      return x
    cb = repeat(cb)
    cr = repeat(cr)

    return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class YCbCrToRGB(nn.Module):
  """Converts YCbCr image to RGB JPEG."""

  def __init__(self, dtype):
    super(YCbCrToRGB, self).__init__()

    matrix = np.array(YCBCR_TO_RGB).T
    self._shift = nn.Parameter(torch.tensor([0, -128., -128.], dtype=dtype))
    self._matrix = nn.Parameter(torch.from_numpy(matrix).to(dtype=dtype))

  def forward(self, image):
    result = torch.tensordot(image + self._shift.to(dtype=image.dtype), self._matrix.to(dtype=image.dtype), dims=1)
    result.view(image.shape)
    return result.permute(0, 3, 1, 2)


class DecompressJPEG(nn.Module):
  """JPEG decompression."""
  def __init__(self, factor=1, dtype=torch.float32):
    super(DecompressJPEG, self).__init__()
    self._c_dequantize = ChromaDequantize(factor=factor)
    self._y_dequantize = LuminanceDequantize(factor=factor)
    self._idct = IDCT8x8(dtype)
    self._merging = BlockMerging()
    self._chroma = ChromaUpsampling()
    self._colors = YCbCrToRGB(dtype)

  def forward(self, y, cb, cr, height, width):
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
      if k in ('cb', 'cr'):
        comp = self._c_dequantize(components[k])
        height_k, width_k = int(height / 2), int(width / 2)

      else:
        comp = self._y_dequantize(components[k])
        height_k, width_k = height, width
      comp = self._idct(comp)
      components[k] = self._merging(comp, height_k, width_k)
    image = self._chroma(components['y'], components['cb'], components['cr'])
    image = self._colors(image)
    image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
    return image / 255