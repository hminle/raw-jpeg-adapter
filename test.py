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

This file contains the testing script for the Raw-JPEG Adapter model.
"""

import argparse
import logging
import os
import shutil
import string
import secrets
import time

from tabulate import tabulate
from typing import Dict, List, Tuple
from file_utils import read_json_file

import torch
from raw_jpeg_adapter_model import JPEGAdapter
from img_utils import imread, img_to_tensor, tensor_to_img, imwrite, get_psnr, get_ssim, get_ms_ssim
from constants import *

def error_metrics(errors: np.ndarray) -> Dict[str, float]:
  """Computes error statistics: mean and quartiles."""
  if errors is None:
    return {}

  errors = np.squeeze(errors)
  q1, q2, q3 = np.percentile(errors, [25, 50, 75])
  mean = float(np.mean(errors))

  return {
    'mean': mean,
    'Q1': float(q1),
    'Q2': float(q2),
    'Q3': float(q3),
  }

def compute_compression_metrics(original_file: str, compressed_file: str, image_shape: Tuple[int, int]
                                ) -> Tuple[float, float]:
  """Computes compression ratio and bits per pixel (bpp).

  Args:
    original_file: Path to the original (reference) file (e.g., PNG-16).
    compressed_file: Path to the compressed file (e.g., JPEG).
    image_shape: (height, width) of the image.

  Returns:
    compression_ratio: Ratio of original size to compressed size.
    bpp: Bits per pixel used by the compressed file.
  """
  original_size = os.path.getsize(original_file)
  compressed_size = os.path.getsize(compressed_file)
  compression_ratio = original_size / compressed_size
  total_pixels = image_shape[0] * image_shape[1]
  bpp = (compressed_size * 8) / total_pixels
  return compression_ratio, bpp

def get_secure_random_string(length=12):
  """Gets random string."""
  characters = string.ascii_letters + string.digits
  return ''.join(secrets.choice(characters) for _ in range(length))


def test_net(model: JPEGAdapter, te_device: torch.device, te_dir: List[str], jpeg_quality: int) -> str:
  """Tests a given trained model."""

  temp_dir = f'temp-{jpeg_quality}-{get_secure_random_string()}'
  os.makedirs(temp_dir, exist_ok=True)

  filenames = [os.path.join(te_dir_i, f) for te_dir_i in te_dir for f in os.listdir(te_dir_i) if f.endswith('.png')]
  psnr = np.zeros((len(filenames), 1))
  ssim = np.zeros((len(filenames), 1))
  ms_ssim = np.zeros((len(filenames), 1))
  bpp = np.zeros((len(filenames), 1))
  comp_ratio = np.zeros((len(filenames), 1))
  total_time = 0
  for i, file in enumerate(filenames):
    print(f'Processing {i+1}/{len(filenames)}...')
    raw_img = imread(file).astype(np.float32)

    shape = raw_img.shape
    raw_img_tensor = img_to_tensor(raw_img).to(device=te_device).unsqueeze(dim=0)
    with torch.no_grad():
      start = time.time()
      _, operator_params = model(raw_img_tensor)

      raw_img_proc = model.preprocess_raw(raw=raw_img_tensor,
                                          gamma_map=operator_params['gamma_map'],
                                          scale_dct=operator_params['scale_dct'],
                                          lut=operator_params['lut'])
    end = time.time()
    total_time += (end - start)
    if operator_params['gamma_map'] is not None:
      imwrite(tensor_to_img(operator_params['gamma_map'].squeeze(0)) ** (1/2.2), os.path.join(temp_dir, 'gamma'),
              format='jpg')

    raw_img_proc = tensor_to_img(raw_img_proc.squeeze(0))
    start = time.time()
    comment = model.encode_params(operator_params)
    end = time.time()
    total_time += (end - start)
    imwrite(image=raw_img_proc, output_path=os.path.join(temp_dir, 'temp-img'), format='jpg',
            quality=jpeg_quality, comment=comment)

    proc_raw_img, comment = imread(os.path.join(temp_dir, 'temp-img.jpg'), load_comment=True)

    start = time.time()
    comment_params = model.decode_params(comment.decode('utf-8'))

    rec_raw_img = model.reconstruct_raw(
      raw_jpeg=img_to_tensor(proc_raw_img).to(device=te_device, dtype=torch.float32).unsqueeze(0),
      gamma_map=comment_params['gamma_map'], scale_dct=comment_params['scale_dct'], lut=comment_params['lut'])
    end = time.time()
    total_time += (end - start)
    rec_raw_img = tensor_to_img(rec_raw_img.squeeze(0)).astype(dtype=np.float32)
    psnr[i] = get_psnr(raw_img, rec_raw_img)
    ssim[i] = get_ssim(raw_img, rec_raw_img)
    ms_ssim[i] = get_ms_ssim(raw_img, rec_raw_img)
    comp_ratio[i], bpp[i] = compute_compression_metrics(file, os.path.join(temp_dir, 'temp-img.jpg'), shape[:2])
  psnr_results = error_metrics(psnr)
  ssim_results = error_metrics(ssim)
  ms_ssim_results = error_metrics(ms_ssim)
  bpp_results = error_metrics(bpp)
  comp_ratio_results = error_metrics(comp_ratio)
  results = ''
  results += (f'PSNR:\n AVG = {psnr_results["mean"]} - Q1 = {psnr_results["Q1"]} - Q2 (median) = {psnr_results["Q2"]} -'
              f' Q3 = {psnr_results["Q3"]}\n')
  results += (f'SSIM:\n AVG = {ssim_results["mean"]} - Q1 = {ssim_results["Q1"]} - Q2 (median) = {ssim_results["Q2"]} -'
              f' Q3 = {ssim_results["Q3"]}\n')
  results += (f'MS-SSIM:\n AVG = {ms_ssim_results["mean"]} - Q1 = {ms_ssim_results["Q1"]} - '
              f'Q2 (median) = {ms_ssim_results["Q2"]} - Q3 = {ms_ssim_results["Q3"]}\n')
  results += (f'BPP:\n AVG = {bpp_results["mean"]} - Q1 = {bpp_results["Q1"]} - Q2 (median) = {bpp_results["Q2"]} -'
              f' Q3 = {bpp_results["Q3"]}\n')
  results += (f'Compression ratio:\n AVG = {comp_ratio_results["mean"]} - Q1 = {comp_ratio_results["Q1"]} - '
              f'Q2 (median) = {comp_ratio_results["Q2"]} - Q3 = {comp_ratio_results["Q3"]}\n')
  results += '----------------------------------------------------\n'
  results += (f'Summary (avg): PSNR = {psnr_results["mean"]} - SSIM = {ssim_results["mean"]} -'
              f' MS-SSIM = {ms_ssim_results["mean"]} - BPP = {bpp_results["mean"]} - '
              f'Compression ratio: {comp_ratio_results["mean"]}\n')
  results += f'Time: total = {total_time} - AVG = {total_time / len(filenames)}'
  shutil.rmtree(temp_dir)
  return results

def get_args():
  parser = argparse.ArgumentParser(description='Test the Raw-JPEG-Adapter network.')
  parser.add_argument('--testing-dir', dest='te_dir', nargs='+',
                      help='Directory or directories containing raw images for testing.')
  parser.add_argument('--config-dir', dest='config_dir', default='config',
                      help='Directory containing config JSON files.')
  parser.add_argument('--result-dir', dest='result_dir', default='results',
                      help='Directory to save the results report (.txt).')
  parser.add_argument('--model-path', dest='model_path', help='Path to the trained model.')
  return parser.parse_args()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  args = get_args()
  os.makedirs(args.result_dir, exist_ok=True)
  print(tabulate([(key, value) for key, value in vars(args).items()], headers=["Argument", "Value"], tablefmt="grid"))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  assert os.path.exists(args.model_path)
  logging.info(f'Using device {device}')
  logging.info(f'Testing of Raw-JPEG Net -- model name: {os.path.basename(args.model_path)} ...')
  config_base = args.model_path.replace('-best', '') if args.model_path.endswith('-best.pth') else args.model_path
  config_base = os.path.splitext(os.path.basename(config_base))[0]
  config = read_json_file(os.path.join(args.config_dir, f'{config_base}.json'))
  net = JPEGAdapter(latent_dim= config['latent_dim'], target_img_size=config['map_size'], use_eca=config['eca'],
                    quality=config['quality'], use_scale_dct=config['scale_dct'],
                    use_gamma=config['gamma'], use_lut=config['lut'], lut_size=config['lut_size'],
                    lut_channels=config['lut_channels']).to(device=device)

  net.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
  net.eval()
  logging.info(f'Model loaded from {args.model_path}')

  results = test_net(model=net, te_device=device, te_dir=args.te_dir, jpeg_quality=config['quality'])
  print(results)
  with open(os.path.join(args.result_dir, os.path.splitext(os.path.basename(args.model_path))[0] + '.txt'), 'w') as f:
    f.write(results)

