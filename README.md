# Raw-JPEG Adapter





This repository contains the code for the paper:

**Raw-JPEG Adapter: Efficient Raw Image Compression with JPEG**

*[Mahmoud Afifi](https://www.mafifi.info/)*, *[Ran Zhang](https://www.linkedin.com/in/ran-zhang-48b85021/)*, and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*

*AI Center-Toronto, Samsung Electronics*


## 💡 Overview

<p align="center">
  <img src="/figures/fig1.jpg" alt="Teaser" width="100%">
</p>

Raw-JPEG Adapter is a lightweight, learnable pre-processing pipeline that adapts raw images before standard JPEG compression using spatial and optionally frequency domain transforms. The operations are fully invertible, with parameters fitting in the JPEG comment field (<64 KB), enabling accurate raw reconstruction after JPEG decoding and significantly reducing file size. In this figure, (A) shows the original raw (DNG), stored as JPEG with high compression (quality 25) without our method in (B), and with our method in (C). Error maps for (B) and (C) are shown on the right. 


<p align="center">
  <img src="/figures/fig2.jpg" alt="Method" width="100%">
</p>

Our Raw-JPEG Adapter uses a lightweight network (~37K parameters) to process a thumbnail of the raw image and produce parameters for a pixel-wise gamma operator, RGB 1D tone-mapping lookup tables, and an optional DCT-based component applied globally in the frequency domain over 8x8 blocks. These transformations are applied before saving the image as a JPEG, while the associated parameters are compressed and embedded in the JPEG file’s comment (COM) segment (<64 KB). At decoding time, the parameters are retrieved, inverted, and applied to the stored image to reconstruct the original raw content. During training, the JPEG step is replaced with a differentiable simulator, and the network is optimized in a self-supervised manner.


## 🛠 Installation

You can set up the environment using **Conda** or **venv**:

### Option 1: Using Conda (recommended)

```bash
# Create and activate a new conda environment
conda create -n raw_jpeg_adapter_env python=3.9 -y
conda activate raw_jpeg_adapter_env

# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

### Option 2: Using venv
```bash
# Create and activate a virtual environment
python -m venv raw_jpeg_adapter_env
source raw_jpeg_adapter_env/bin/activate        # macOS/Linux
# On Windows:
# .\raw_jpeg_adapter_env\Scripts\activate

# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

## ⚙️ Training

To train the model that predicts a 100×100 spatial gamma map and a 128×3 LUT with color and scale augmentation, use the following command:

```bash
python train.py --training-dir /path/to/training/png-16/raw-image/folder --validation-dir /path/to/validation/png-16/raw-image/folder --jpeg-quality target-jpeg-quality-level --use-gamma --use-lut --color-aug --scale-aug
```


### Additional Arguments:
- `--epochs N`: Number of training epochs (default: 100; e.g., --epochs 50).
- `--batch-size N`: Mini-batch size (default: 16; e.g., --batch-size 8).
- `--patch-size N`: Training patch size (default: 512; e.g., --patch-size 256).
- `--latent-dim N`: Dimensionality of latent features produced by the encoder (default: 24; e.g., --latent-dim 12).
- `--discard-eca`: Disable the ECA block.
- `--use-scale-dct`: Enable prediction of an 8×8 global DCT scale matrix.
- `--lut-size N`: Number of bins in the LUT (default: 128; e.g., --lut-size 64).
- `--lut-channels N`: Number of channels in the LUT (default: 3; should be 1 or 3).
- `--discard-jpeg-block`: Disable JPEG simulation and use 8-bit quantization instead.
- `--map-size H W`: Size of the gamma map output (default: 100×100; e.g., --map-size 48 64).
- `--num-patches-per-image N`: Number of patches per image (0 uses all non-overlapping patches; default: 0).
- `--temp-folder S`: Name of the temporary folder used to save patches (default: "temp_patches"; e.g., --temp-folder extracted_patches).
- `--overwrite-temp-folder`: Overwrite existing temp folder, if it exists.
- `--delete-temp-folder`: Delete temp folder after training.
- `--l1-loss-weight F`: L1 loss weight (default: 1.0; set to 0 to disable).
- `--fft-loss-weight F`: FFT loss weight (default: 0.1; set to 0 to disable).
- `--ssim-loss-weight F`: SSIM loss weight (default: 0.1; set to 0 to disable).


## 🔬 Testing 
To evaluate the model, use:


```bash
python test.py --testing-dir /path/to/png-16/raw-image/folder --model-path path/to/trained/model
```

## 💻 Demo

To test the Raw-JPEG Adapter on a single DNG or PNG-16 raw image, use the `demo.py` script:
```bash
python demo.py --input-file /path/to/your/image.dng
```

This will:

Run the model on the given raw image.

Save the JPEG output (default quality = 75).

Use GPU for inference (default).

**Common Options**:

`--input-file`: (required) Path to the DNG or PNG-16 raw input image.

`--output-dir`: Directory to save output images (default: current directory).

`--device`: Choose inference device: gpu or cpu (default: gpu).

`--jpeg-quality`: JPEG quality level: one of [25, 50, 75, 95, 100] (default: 75).

`--dct-component`: Enable DCT component (**recommended only for Samsung S24 Ultra**).

`--save-png-16`: Save the decoded raw as a 16-bit PNG.

`--save-gamma-map`: Save the gamma map generated by our model.

`--save-raw-jpg-wo-pre-processing`: Save the raw JPEG without our pre-processing.

`--save-original-raw`: Save the original raw as PNG-16.

`--visualization`: Apply global tone mapping to improve raw image visualization.

Example with DCT, custom quality, and saving all optional output files.
```bash
python demo.py --input-file /path/to/image.dng --jpeg-quality 25 --dct-component --save-png-16 --save-raw-jpg-wo-pre-processing --save-original-raw --save-gamma-map
```
⚠️ Use `--dct-component` only if the input raw image was captured using the Samsung S24 Ultra main camera (the camera used during training).


## 📄 Citation
If you use this code in your research, please cite our paper:
```
@article{afifi2025raw,
  title={Raw-JPEG Adapter: Efficient Raw Image Compression with JPEG},
  author={Afifi, Mahmoud and Zhang, Ran and Brown, Michael S},
  journal={arXiv preprint arXiv:2509.19624},
  year={2025}
}
```


