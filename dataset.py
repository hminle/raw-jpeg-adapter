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

This file defines the data loading pipeline for training the Raw-JPEG Adapter.
"""

import os
from os.path import join, exists, dirname
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from typing import Optional, Dict, List
import shutil
import random
import collections
import h5py

from img_utils import extract_non_overlapping_patches, imread, augment_img, img_to_tensor

class Data(Dataset):
  def __init__(self, img_dir: str, patch_size: Optional[int]=512, num_patches_per_image: Optional[int]=0,
               temp_folder: Optional[str]='temp_patches_h5', overwrite_temp_folder: Optional[bool]=False,
               geometric_aug: Optional[bool]=True, color_aug: Optional[bool]=False,
               intensity_aug: Optional[bool]=False, patches_per_file: Optional[int]=1000,
               batch_size: Optional[int]=8, shuffle: Optional[bool]=False):
    self._img_dir = img_dir
    self._patch_size = patch_size
    self._geo_aug = geometric_aug
    self._col_aug = color_aug
    self._scale_aug = intensity_aug
    self._patches_per_file = patches_per_file
    self._batch_size = batch_size
    self._shuffle = shuffle
    self._max_open_files = 64

    if self._patch_size > 0:
      self._temp_dir = join(dirname(img_dir), f'{temp_folder}_bs{batch_size}')

      re_create = True
      if exists(self._temp_dir) and overwrite_temp_folder:
        logging.info('Temporary directory exists. Removing it and re-preprocessing images...')
        shutil.rmtree(self._temp_dir)
      elif exists(self._temp_dir):
        re_create = False
      else:
        os.makedirs(self._temp_dir)

      if re_create:
        logging.info(f'Preprocessing images to extract non-overlapping patches with batch_size={batch_size}...')
        self._create_hdf5_files(num_patches_per_image)
      else:
        logging.info(f'Found pre-extracted batches in {self._temp_dir}; skipping reprocessing.')

      self._h5_file_paths = sorted([
        join(self._temp_dir, f) for f in os.listdir(self._temp_dir) if f.endswith('.h5')
      ])
      self._h5_cache: 'collections.OrderedDict[str, h5py.File]' = collections.OrderedDict()
    else:
      self._img_file_paths = sorted([
        join(self._img_dir, f) for f in os.listdir(self._img_dir) if f.endswith('.png')
      ])

  def __len__(self):
    if self._patch_size > 0:
      return len(self._h5_file_paths)
    else:
      return len(self._img_file_paths)

  def _open_h5_file(self, h5_path: str) -> h5py.File:
    """Opens an HDF5 file with caching to avoid too many open files."""
    if h5_path in self._h5_cache:
      self._h5_cache.move_to_end(h5_path)
      return self._h5_cache[h5_path]

    if len(self._h5_cache) >= self._max_open_files:
      old_path, old_file = self._h5_cache.popitem(last=False)
      old_file.close()

    f = h5py.File(h5_path, 'r')
    self._h5_cache[h5_path] = f
    return f

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Return a full batch (of size batch_size) from one HDF5 file (or a single image if patch-size = 0)."""
    if self._patch_size > 0:
      h5_path = self._h5_file_paths[idx]
      with self._open_h5_file(h5_path) as f:
        patches = f['patches'][()]
      if self._shuffle:
        np.random.shuffle(patches)
      batch = []
      for patch in patches:
        if self._geo_aug:
          patch = augment_img(patch)
        if self._col_aug:
          patch = self._apply_col_aug(patch)
        if self._scale_aug:
          patch = self._apply_scale_aug(patch)
        batch.append(img_to_tensor(patch))

      return {'images': torch.stack(batch).to(dtype=torch.float32)}
    else:
      raw_img = imread(self._img_file_paths[idx])
      raw_img = self._crop_image_to_multiple(raw_img, k=16)

      if self._geo_aug:
        raw_img = augment_img(raw_img)
      if self._col_aug:
        raw_img = self._apply_col_aug(raw_img)
      if self._scale_aug:
        raw_img = self._apply_scale_aug(raw_img)

      return {'images': img_to_tensor(raw_img).to(dtype=torch.float32)}

  def _create_hdf5_files(self, num_patches_per_img: int):
    """Creates HDF5 files where each file contains 'batch_size' patches."""
    all_patches: List[np.ndarray] = []
    file_counter = 0
    img_files = sorted(os.listdir(self._img_dir))
    for i, img_file in enumerate(img_files):
      if img_file.startswith('.'):
        continue
      print(f'Processing {i}/{len(img_files)} ...')
      img_path = join(self._img_dir, img_file)
      in_img = imread(img_path)
      patches = extract_non_overlapping_patches(
        img=in_img, num_patches=num_patches_per_img,
        patch_size=self._patch_size, allow_overlap=True)

      for patch in patches['img']:
        all_patches.append(patch.astype(np.float32))
        if len(all_patches) == self._batch_size:
          self._write_hdf5(file_counter, all_patches)
          all_patches = []
          file_counter += 1

    if all_patches:
      self._write_hdf5(file_counter, all_patches)

  def _write_hdf5(self, file_id: int, patches: List[np.ndarray]):
    fname = join(self._temp_dir, f"batch_{file_id:05d}.h5")
    logging.info(f'Writing batch of {len(patches)} patches to {fname}...')
    with h5py.File(fname, 'w') as f:
      f.create_dataset('patches', data=np.stack(patches), compression='gzip')

  @staticmethod
  def _apply_col_aug(img: np.ndarray) -> np.ndarray:
    """Applies simple color augmentation by randomly applying a 3x3 matrix."""
    if random.random() < 0.5:
      m = np.eye(3) + np.random.randn(3, 3) * 0.1
      m[1] += np.array([0.0, 0.05, 0.0])  # bias green response
      m = m / m.sum(axis=0, keepdims=True)  # normalizes cols
      img = (img.reshape(-1, 3) @ m).reshape(img.shape)  # applies and reshapes back
    return img

  @staticmethod
  def _apply_scale_aug(img: np.ndarray) -> np.ndarray:
    """Applies simple intensity augmentation by randomly applying a scale factor."""
    if random.random() < 0.5:
      scale = np.random.uniform(0.85, 1.15)
      img *= scale
    return img

  @staticmethod
  def _crop_image_to_multiple(img: np.ndarray, k: int) -> np.ndarray:
    """Crops the image so that height and width are divisible by k."""
    h, w = img.shape[:2]
    new_h = (h // k) * k
    new_w = (w // k) * k
    return img[:new_h, :new_w]

