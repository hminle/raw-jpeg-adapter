# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raw-JPEG Adapter is a research project implementing a lightweight neural network (~37K parameters) that preprocesses raw images before JPEG compression. Transformation parameters are embedded in the JPEG comment field (<64KB), enabling accurate raw reconstruction after decoding.

**License**: CC BY-NC 4.0 (non-commercial use only)

## Common Commands

### Environment Setup
```bash
conda create -n raw_jpeg_adapter_env python=3.9 -y
conda activate raw_jpeg_adapter_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Training
```bash
python train.py --training-dir /path/to/train/png16/ --validation-dir /path/to/val/png16/ --jpeg-quality 75 --use-gamma --use-lut --color-aug --scale-aug
```

With DCT component (for Samsung S24 Ultra data):
```bash
python train.py --training-dir /path/to/train/ --validation-dir /path/to/val/ --jpeg-quality 75 --use-gamma --use-lut --use-scale-dct --color-aug --scale-aug
```

### Testing
```bash
python test.py --testing-dir /path/to/test/png16/ --model-path path/to/model.pth
```

### Demo (Single Image Inference)
```bash
python demo.py --input-file /path/to/image.dng --jpeg-quality 75
python demo.py --input-file /path/to/image.dng --jpeg-quality 75 --dct-component  # Samsung S24 Ultra only
```

## Architecture

### Core Pipeline
1. **Encoder** (`raw_jpeg_adapter_model.py`): Processes 100×100 thumbnail through conv layers + optional ECA attention → latent features (24-dim default)
2. **Decoder** (`raw_jpeg_adapter_model.py`): Predicts three transformation parameters:
   - **Gamma Map** (100×100): Spatial pixel-wise gamma correction
   - **Scale DCT** (8×8): Optional frequency domain scaling
   - **LUT** (128×3): Per-channel tone mapping curves
3. **Pre-processing**: Apply LUT → DCT scale → Gamma to raw before JPEG encoding
4. **Post-processing**: Inverse Gamma → Inverse DCT → Inverse LUT after JPEG decoding

### Key Files
- `raw_jpeg_adapter_model.py`: Neural network (JPEGAdapter, Encoder, Decoder, LuTModule, ECABlock)
- `jpeg_utils.py`: Differentiable JPEG simulator for training (RGBToYCbCr, DCT, quantization)
- `dataset.py`: Data loading with HDF5 caching and augmentations
- `loss_utils.py`: L1, FFT, and SSIM loss functions
- `img_utils.py`: DNG/raw handling, demosaicing, image I/O, metrics
- `constants.py`: Global constants, JPEG quality levels, model paths

### Pre-trained Models
Located in `models/` with configs in `config/`. Models follow naming: `raw_jpeg_adapter_q_{quality}_{w|wo}_dct.pth`

Supported JPEG quality levels: 25, 50, 75, 95, 100

### Parameter Embedding
Parameters are zlib-compressed and base64-encoded into JPEG COM segment via `encode_params()` / `decode_params()` methods in JPEGAdapter class.

## Key Implementation Details

- Training uses differentiable JPEG simulation (`jpeg_utils.py`) instead of actual JPEG codec
- Data pipeline extracts patches, caches to HDF5, applies geometric/color/intensity augmentations
- Input raw images must be PNG-16 (training) or DNG format (demo)
- DCT component (`--use-scale-dct` / `--dct-component`) trained specifically on Samsung S24 Ultra data
