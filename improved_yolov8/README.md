# Improved YOLOv8s Architecture

Implementation of Improved YOLOv8s architecture based on paper sensors-24-02896.

## Architecture Overview

The Improved YOLOv8s includes:

### Backbone
- **RFCBAMConv**: Residual Feature Channel-wise and Bottleneck Attention Module Convolution
- **C2f_RFCBAM**: C2f module with RFCBAM bottleneck blocks
- **Mix-SPPF**: Mixed Spatial Pyramid Pooling Fast (combines MaxPool and AvgPool)

### Neck
- **RepGFPN**: Re-parameterized Generalized Feature Pyramid Network
- **AKConv**: Adaptive Kernel Convolution
- **C2f**: Faster CSP Bottleneck with 2 convolutions

### Head
- **DynamicHead**: Dynamic detection head with three attention mechanisms:
  - Scale-aware Attention (π_L)
  - Spatial-aware Attention (π_S)
  - Task-aware Attention (π_C)

### Loss Function
- **Inner-IoU Loss**: Focuses on inner region of bounding boxes

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Training

```bash
python train.py --data data.yaml --epochs 150 --batch 16 --imgsz 640
```

## Model Configuration

The model configuration is in `cfg/yolov8-improved.yaml`.

## Files Structure

```
improved_yolov8/
├── models/
│   ├── modules/
│   │   ├── conv.py          # Base convolution modules
│   │   └── block.py          # Base block modules (C2f, SPPF, Mix-SPPF)
│   ├── extra_modules/
│   │   ├── RFAConv.py       # RFCBAMConv and related modules
│   │   ├── block.py         # C2f_RFCBAM, AKConv, RepGFPN
│   │   └── head.py          # DynamicHead
│   └── utils/
│       └── loss.py          # Inner-IoU loss
├── cfg/
│   └── yolov8-improved.yaml # Model configuration
├── train.py                 # Training script
├── data.yaml                # Dataset configuration
└── requirements.txt         # Dependencies
```

## Notes

- All modules from TDDet have been copied to this folder
- The architecture follows the paper's specifications
- Inner-IoU loss helps focus on central regions of objects

