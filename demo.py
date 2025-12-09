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

This demo illustrates the use of trained Raw-JPEG Adapter models. It takes a raw image as input (in DNG or 16-bit PNG
format) and outputs a JPEG image with pre-processing applied and metadata embedded. The script also reconstructs the
raw image from the saved JPEG and reports PSNR and SSIM metrics.
"""
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from img_utils import (extract_image_from_dng, extract_raw_metadata, normalize_raw, demosaice, imwrite, imread,
                       img_to_tensor, tensor_to_img, get_psnr, get_ssim, get_ms_ssim, visualization)

from file_utils import read_json_file
from raw_jpeg_adapter_model import JPEGAdapter
from constants import *
import torch


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(description='Raw-JPEG Adapter Demo')

  parser.add_argument('--input-file', type=str, required=True,
                      help='Path to a DNG or PNG-16 raw image file.', dest='input_file')

  parser.add_argument('--output-dir', type=str, default='.',
                      help='Directory to save the output image.', dest='output_dir')

  parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'],
                      help=f'Device to run the model on: {DEVICES}.')

  parser.add_argument('--jpeg-quality', type=int, default=75, choices=JPEG_QUALITY_LEVELS,
                      help=f'JPEG quality level, should be one of: {JPEG_QUALITY_LEVELS}.')

  parser.add_argument(
    '--dct-component', action='store_true',
    help='Enable DCT component in the Raw-JPEG Adapter. Recommended only for Samsung S24 Ultra main camera.')

  parser.add_argument('--save-png-16', action='store_true',
                      help='Save the post-processed raw image in 16-bit PNG format.')

  parser.add_argument('--save-gamma-map', action='store_true',
                      help="Save the gamma map as 'gamma_{output_file}'.")

  parser.add_argument('--save-raw-jpg-wo-pre-processing', action='store_true',
                      help='Save the raw image as a JPEG without any pre-processing.')

  parser.add_argument('--save-original-raw', action='store_true',
                      help='Save the original raw image as a 16-bit PNG file for reference purposes.')

  parser.add_argument('--visualization', action='store_true',
                      help='Apply global tone mapping to raw output for better visualization. Only effective '
                           'if save-raw-jpg-wo-pre-processing, save-png-16, or save-original-raw is True.')

  return parser.parse_args()

pypass = lambda x, *args, **kwargs: x

if __name__ == '__main__':
  args = get_args()

  assert os.path.exists(args.input_file), 'File does not exist.'
  assert args.input_file.lower().endswith('.dng') or args.input_file.lower().endswith('.png'), 'Invalid file type.'

  dir_path = args.output_dir
  base_name = 'temp.jpg'
  output_file = os.path.join(dir_path, base_name)

  assert args.jpeg_quality in JPEG_QUALITY_LEVELS, (f'Invalid JPEG quality: {args.jpeg_quality}. '
                                                    f'Allowed values are: {JPEG_QUALITY_LEVELS}.')

  if args.dct_component:
    model_name = JPEG_RAW_W_DCT_MODELS[args.jpeg_quality]
  else:
    model_name = JPEG_RAW_WO_DCT_MODELS[args.jpeg_quality]

  assert args.device in DEVICES, 'Invalid device.'

  if args.visualization:
    vis_func = visualization
  else:
    vis_func = pypass


  if args.device == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  else:
    device = torch.device('cpu')

  model_path = os.path.join('models', model_name)

  config_path = os.path.join('config', model_name.replace('.pth', '.json'))
  config = read_json_file(config_path)
  net = JPEGAdapter(latent_dim=config['latent_dim'], target_img_size=config['map_size'], use_eca=config['eca'],
                    quality=config['quality'], use_scale_dct=config['scale_dct'],
                    use_gamma=config['gamma'], use_lut=config['lut'], lut_size=config['lut_size'],
                    lut_channels=config['lut_channels']).to(device=device)

  net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
  net.eval()

  if args.input_file.lower().endswith('.dng'):
    metadata = extract_raw_metadata(args.input_file)
    raw_img = extract_image_from_dng(args.input_file)
    raw_img = normalize_raw(img=raw_img, black_level=metadata['black_level'],
                            white_level=metadata['white_level']).astype(np.float32)
    if not (raw_img.shape[-1] == 4 and len(raw_img.shape) == 3):
      try:
        print('demosaicing...')
        raw_img = demosaice(raw_img, metadata['pattern'])
      except:
        raise NotImplementedError('Unsupported bayer pattern.')
    else:
      raw_img = raw_img[..., :3]
  else:
    raw_img = imread(args.input_file).astype(np.float32)

  if args.save_original_raw:
    filename = os.path.join(dir_path, f'original_{base_name}')
    imwrite(vis_func(raw_img), filename, format='png-16')

  # Encoding:
  print('encoding...')
  raw_img_tensor = img_to_tensor(raw_img).to(device=device, dtype=torch.float32).unsqueeze(dim=0)
  with torch.no_grad():
    raw_img_proc, operator_params = net(raw_img_tensor)
  raw_img_proc = tensor_to_img(raw_img_proc.squeeze(0))
  comment = net.encode_params(operator_params)
  imwrite(image=raw_img_proc, output_path=output_file, format='jpg', quality=args.jpeg_quality, comment=comment)

  if args.save_gamma_map:
    filename = os.path.join(dir_path, f'gamma_{base_name}')
    imwrite(tensor_to_img(operator_params['gamma_map'].squeeze(0)) ** (1 / 2.2), filename, format='png-8')

  if args.save_raw_jpg_wo_pre_processing:
    filename = os.path.join(dir_path, f'wo_pre_processing_{base_name}')
    imwrite(image=vis_func(raw_img), output_path=filename, format='jpg', quality=args.jpeg_quality)

  # Decoding:
  print('decoding...')
  proc_raw_img, comment = imread(output_file, load_comment=True)

  comment_params = net.decode_params(comment.decode('utf-8'), device=device)

  rec_raw_img = net.reconstruct_raw(
    raw_jpeg=img_to_tensor(proc_raw_img).to(device=device, dtype=torch.float32).unsqueeze(0),
    gamma_map=comment_params['gamma_map'], scale_dct=comment_params['scale_dct'], lut=comment_params['lut'])
  rec_raw_img = tensor_to_img(rec_raw_img.squeeze(0)).astype(dtype=np.float32)

  psnr = get_psnr(raw_img, rec_raw_img)
  ssim = get_ssim(raw_img, rec_raw_img)
  ms_ssim = get_ms_ssim(raw_img, rec_raw_img)

  if args.save_png_16:
    filename = os.path.join(dir_path, f'png-16-{base_name}')
    imwrite(vis_func(rec_raw_img), filename, 'PNG-16')

  print(f'PSNR = {psnr} - SSIM = {ssim} - MS-SSIM = {ms_ssim}')
  print('Done!')
