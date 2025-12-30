# Raw-JPEG Adapter Documentation

## Overview

**Raw-JPEG Adapter** is a research project that implements a lightweight, learnable preprocessing pipeline for efficient raw image compression using standard JPEG. The key innovation is embedding transformation parameters in the JPEG comment field, enabling accurate raw reconstruction after JPEG decoding.

### Key Features

- **Lightweight Neural Network**: ~37K parameters
- **Embedded Parameters**: Transformation parameters stored in JPEG comment (<64 KB)
- **Multiple Quality Levels**: Support for JPEG quality 25, 50, 75, 95, and 100
- **Optional DCT Processing**: Frequency-domain scaling for improved quality
- **End-to-End Training**: Differentiable JPEG simulation enables gradient-based optimization

### Authors

- Mahmoud Afifi, Ran Zhang, Michael S. Brown
- Samsung Electronics (AI Center-Toronto)
- [arXiv:2509.19624](https://arxiv.org/abs/2509.19624)

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph Input
        RAW[Raw Image<br/>Full Resolution]
    end

    subgraph Encoder["Encoder Network"]
        THUMB[Thumbnail<br/>100x100]
        CONV1[Conv Layer 1]
        CONV2[Conv Layer 2]
        CONV3[Conv Layer 3]
        ECA[ECA Block<br/>Channel Attention]
        LATENT[Latent Features<br/>24-dim]
    end

    subgraph Decoder["Decoder Network"]
        DEC[Decoder Layers]
        GAMMA[Gamma Map<br/>100x100]
        DCT[Scale DCT<br/>8x8]
        LUT[LUT<br/>128 bins x 3 ch]
    end

    subgraph PreProcess["Pre-Processing"]
        PLUT[Apply LUT]
        PDCT[Apply DCT Scale]
        PGAMMA[Apply Gamma]
    end

    subgraph JPEG["JPEG Compression"]
        ENCODE[JPEG Encode]
        COMMENT[Embed Params<br/>in Comment]
        JPEGFILE[JPEG File]
    end

    subgraph PostProcess["Post-Processing"]
        DECODE[JPEG Decode]
        EXTRACT[Extract Params]
        IGAMMA[Inverse Gamma]
        IDCT[Inverse DCT]
        ILUT[Inverse LUT]
        RECON[Reconstructed Raw]
    end

    RAW --> THUMB
    THUMB --> CONV1 --> CONV2 --> CONV3 --> ECA --> LATENT
    LATENT --> DEC
    DEC --> GAMMA
    DEC --> DCT
    DEC --> LUT

    RAW --> PLUT
    LUT --> PLUT
    PLUT --> PDCT
    DCT --> PDCT
    PDCT --> PGAMMA
    GAMMA --> PGAMMA

    PGAMMA --> ENCODE
    GAMMA --> COMMENT
    DCT --> COMMENT
    LUT --> COMMENT
    ENCODE --> COMMENT
    COMMENT --> JPEGFILE

    JPEGFILE --> DECODE
    JPEGFILE --> EXTRACT
    DECODE --> IGAMMA
    EXTRACT --> IGAMMA
    IGAMMA --> IDCT
    IDCT --> ILUT
    ILUT --> RECON
```

---

## Code Structure

```mermaid
graph TB
    subgraph Core["Core Implementation"]
        MODEL[raw_jpeg_adapter_model.py<br/>Neural Network ~536 lines]
        DATASET[dataset.py<br/>Data Pipeline ~183 lines]
        LOSS[loss_utils.py<br/>Loss Functions ~95 lines]
        JPEG[jpeg_utils.py<br/>Differentiable JPEG ~306 lines]
        IMG[img_utils.py<br/>Image Utilities ~307 lines]
        FILE[file_utils.py<br/>File I/O ~43 lines]
        CONST[constants.py<br/>Global Constants ~43 lines]
    end

    subgraph Scripts["Training & Inference"]
        TRAIN[train.py<br/>Training Script ~465 lines]
        TEST[test.py<br/>Evaluation Script ~194 lines]
        DEMO[demo.py<br/>Single Image Demo ~174 lines]
    end

    subgraph Resources["Pre-trained Models"]
        MODELS[models/<br/>10 pre-trained .pth files]
        CONFIG[config/<br/>10 configuration .json files]
    end

    TRAIN --> MODEL
    TRAIN --> DATASET
    TRAIN --> LOSS
    TRAIN --> JPEG
    TEST --> MODEL
    TEST --> IMG
    DEMO --> MODEL
    DEMO --> IMG
    MODEL --> FILE
    DATASET --> IMG
    MODEL --> CONST
    JPEG --> CONST
```

### File Descriptions

| File | Purpose | Lines |
|------|---------|-------|
| `raw_jpeg_adapter_model.py` | Main neural network (JPEGAdapter, Encoder, Decoder, LuTModule, ECABlock) | ~536 |
| `dataset.py` | PyTorch Dataset with HDF5 caching and augmentations | ~183 |
| `loss_utils.py` | L1, FFT, and SSIM loss functions | ~95 |
| `jpeg_utils.py` | Differentiable JPEG encoder/decoder for training | ~306 |
| `img_utils.py` | Image I/O, DNG handling, demosaicing, metrics | ~307 |
| `file_utils.py` | File/directory utilities | ~43 |
| `constants.py` | Global constants, JPEG tables, model paths | ~43 |
| `train.py` | Training loop with TensorBoard logging | ~465 |
| `test.py` | Evaluation with comprehensive metrics | ~194 |
| `demo.py` | Single-image inference demonstration | ~174 |

---

## Neural Network Components

```mermaid
classDiagram
    class JPEGAdapter {
        +encoder: Encoder
        +decoder: Decoder
        +lut_module: LuTModule
        +forward(img) tuple
        +apply_pre_processing(img, params)
        +apply_post_processing(img, params)
        +encode_params(params) str
        +decode_params(comment) tuple
    }

    class Encoder {
        +conv_layers: Sequential
        +eca_block: ECABlock
        +forward(x) tensor
    }

    class Decoder {
        +fc_layers: Sequential
        +gamma_head: Sequential
        +dct_head: Sequential
        +lut_head: Sequential
        +forward(x) tuple
    }

    class LuTModule {
        +lut_size: int
        +lut_channels: int
        +apply_lut(img, lut) tensor
        +apply_inverse_lut(img, lut) tensor
    }

    class ECABlock {
        +conv: Conv1d
        +sigmoid: Sigmoid
        +forward(x) tensor
    }

    JPEGAdapter --> Encoder
    JPEGAdapter --> Decoder
    JPEGAdapter --> LuTModule
    Encoder --> ECABlock
```

### Transformation Parameters

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| **Gamma Map** | 100×100 | Spatial pixel-wise gamma correction |
| **Scale DCT** | 8×8 | Global frequency domain scaling |
| **LUT** | 128×3 | Per-channel tone mapping curves |

---

## Data Flow

```mermaid
sequenceDiagram
    participant RAW as Raw Image
    participant NET as Neural Network
    participant PRE as Pre-Processing
    participant JPEG as JPEG Codec
    participant POST as Post-Processing
    participant OUT as Output

    RAW->>NET: Thumbnail (100x100)
    NET->>NET: Encode → Latent
    NET->>NET: Decode → Parameters
    NET-->>PRE: γ map, DCT scale, LUT

    RAW->>PRE: Full resolution
    PRE->>PRE: Apply LUT
    PRE->>PRE: Apply DCT scaling
    PRE->>PRE: Apply gamma
    PRE->>JPEG: Processed image

    JPEG->>JPEG: Compress
    Note over JPEG: Embed params<br/>in comment field

    JPEG->>POST: Decode JPEG
    Note over POST: Extract params<br/>from comment

    POST->>POST: Inverse gamma
    POST->>POST: Inverse DCT
    POST->>POST: Inverse LUT
    POST->>OUT: Reconstructed Raw
```

---

## Training Pipeline

```mermaid
flowchart LR
    subgraph DataLoader
        D1[Load Raw Patches]
        D2[Apply Augmentations]
        D3[Batch Formation]
    end

    subgraph Forward["Forward Pass"]
        F1[Network → Params]
        F2[Pre-process Raw]
        F3[Differentiable JPEG]
        F4[Post-process]
    end

    subgraph Loss["Loss Computation"]
        L1[L1 Loss]
        L2[FFT Loss]
        L3[SSIM Loss]
        L4[Combined Loss]
    end

    subgraph Backward["Optimization"]
        B1[Gradients]
        B2[Adam Update]
        B3[LR Schedule]
    end

    D1 --> D2 --> D3
    D3 --> F1 --> F2 --> F3 --> F4
    F4 --> L1
    F4 --> L2
    F4 --> L3
    L1 --> L4
    L2 --> L4
    L3 --> L4
    L4 --> B1 --> B2 --> B3
```

---

## Loss Functions

### Combined Loss Formula

```
L_total = w_1 × L_1(x, x̂) + w_fft × L_fft(x, x̂) + w_ssim × (1 - SSIM(x, x̂))
```

Where:
- `L_1`: Pixel-wise absolute difference
- `L_fft`: Frequency domain matching (FFT real + imaginary)
- `SSIM`: Structural Similarity Index

### Default Weights
- `w_1 = 1.0`
- `w_fft = 0.1`
- `w_ssim = 0.1`

---

## Pre-trained Models

The project provides 10 pre-trained models for different configurations:

| Quality | With DCT | Without DCT |
|---------|----------|-------------|
| Q=25 | `raw_jpeg_adapter_q_25_w_dct.pth` | `raw_jpeg_adapter_q_25_wo_dct.pth` |
| Q=50 | `raw_jpeg_adapter_q_50_w_dct.pth` | `raw_jpeg_adapter_q_50_wo_dct.pth` |
| Q=75 | `raw_jpeg_adapter_q_75_w_dct.pth` | `raw_jpeg_adapter_q_75_wo_dct.pth` |
| Q=95 | `raw_jpeg_adapter_q_95_w_dct.pth` | `raw_jpeg_adapter_q_95_wo_dct.pth` |
| Q=100 | `raw_jpeg_adapter_q_100_w_dct.pth` | `raw_jpeg_adapter_q_100_wo_dct.pth` |

---

## Configuration Format

Each model has a corresponding JSON configuration:

```json
{
    "latent_dim": 24,
    "map_size": [100, 100],
    "eca": true,
    "gamma": true,
    "scale_dct": true,
    "lut": true,
    "lut_size": 128,
    "lut_channels": 3,
    "quality": 75
}
```

---

## Directory Structure

```
raw-jpeg-adapter/
├── README.md                    # Project overview
├── LICENSE.md                   # CC BY-NC-SA 4.0
├── requirements.txt             # Python dependencies
├── constants.py                 # Global constants
│
├── models/                      # Pre-trained weights
│   └── *.pth                    # 10 model files
│
├── config/                      # Model configurations
│   └── *.json                   # 10 config files
│
├── docs/                        # Documentation
│   ├── README.md                # This file
│   ├── USAGE.md                 # Usage guide
│   └── RESEARCH_IDEAS.md        # Future research
│
├── raw_jpeg_adapter_model.py    # Neural network
├── dataset.py                   # Data loading
├── loss_utils.py                # Loss functions
├── jpeg_utils.py                # Differentiable JPEG
├── img_utils.py                 # Image utilities
├── file_utils.py                # File utilities
│
├── train.py                     # Training script
├── test.py                      # Evaluation script
└── demo.py                      # Inference demo
```

---

## References

- **Paper**: [Raw-JPEG Adapter: Efficient Raw Image Compression with JPEG](https://arxiv.org/abs/2509.19624)
- **License**: CC BY-NC-SA 4.0 (Non-commercial use only)

```bibtex
@article{afifi2025raw,
  title={Raw-JPEG Adapter: Efficient Raw Image Compression with JPEG},
  author={Afifi, Mahmoud and Zhang, Ran and Brown, Michael S},
  journal={arXiv preprint arXiv:2509.19624},
  year={2025}
}
```
