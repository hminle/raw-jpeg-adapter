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

This file contains all constant values used in this project.
"""
import numpy as np

EPS = 0.00000001
DEVICES = ['cpu', 'gpu']
LUMINANCE_QUANTIZATION_TABLE = [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]]
CHROMA_QUANTIZATION_TABLE = [[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]
RGB_TO_YCBCR = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]
YCBCR_TO_RGB = [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]]
JPEG_ADAPT_LOSS_SCALE = 100
JPEG_ADAPT_OUT_DTYPE = np.float32
JPEG_ADAPT_LUT_BINS = 128
JPEG_ADAPT_LUT_CHS = 3
JPEG_ADAPT_MAP_SZ = 100
JPEG_QUALITY_LEVELS = [25, 50, 75, 95, 100]
JPEG_RAW_W_DCT_MODELS = {25: 'raw_jpeg_adapter_q_25_w_dct.pth',
                         50: 'raw_jpeg_adapter_q_50_w_dct.pth',
                         75: 'raw_jpeg_adapter_q_75_w_dct.pth',
                         95: 'raw_jpeg_adapter_q_95_w_dct.pth',
                         100: 'raw_jpeg_adapter_q_100_w_dct.pth'}

JPEG_RAW_WO_DCT_MODELS = {25: 'raw_jpeg_adapter_q_25_wo_dct.pth',
                          50: 'raw_jpeg_adapter_q_50_wo_dct.pth',
                          75: 'raw_jpeg_adapter_q_75_wo_dct.pth',
                          95: 'raw_jpeg_adapter_q_95_wo_dct.pth',
                          100: 'raw_jpeg_adapter_q_100_wo_dct.pth'}