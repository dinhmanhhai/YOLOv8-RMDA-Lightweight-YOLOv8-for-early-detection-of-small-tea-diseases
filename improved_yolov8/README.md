# Improved YOLOv8 with RFCBAM, MixSPPF, RepGFPN, and Dynamic Head

Kiến trúc detection cải tiến dựa trên YOLOv8 với các module mới:

- **RFCBAMConv**: Receptive Field Concentration-Based Attention Module
- **C2f_RFCBAM**: C2f module với RFCBAM_Neck
- **MixSPPF**: Mix Spatial Pyramid Pooling Fast (kết hợp MaxPool và AvgPool)
- **RepGFPN**: Reparameterized Generalized Feature Pyramid Network
- **AKConv**: Adaptive Kernel Convolution
- **Inner-IoU Loss**: Improved bounding box regression loss

## Cài đặt

```bash
pip install ultralytics
pip install einops
```

## Cấu trúc

```
improved_yolov8/
├── models/
│   ├── __init__.py
│   ├── blocks.py          # MixSPPF, RFCBAM_Neck, C2f_RFCBAM
│   ├── rfcbam.py          # RFCBAMConv
│   ├── repgfpn.py         # RepGFPN
│   └── akconv.py          # AKConv
├── configs/
│   └── yolov8-rfcbam-dynamic.yaml  # YAML config file
├── losses.py              # Inner-IoU loss
├── utils.py                # Module registration
└── README.md
```

## Sử dụng

### 1. Import và đăng ký modules

```python
from improved_yolov8 import utils  # Auto-registers modules
from ultralytics import YOLO
```

### 2. Training

**Lưu ý**: Đảm bảo file `dataset/data.yaml` có cấu hình đúng với dataset của bạn:
- `train`: Đường dẫn đến thư mục train images
- `val`: Đường dẫn đến thư mục validation images  
- `nc`: Số lượng classes (ví dụ: 6 cho tea diseases dataset)
- `names`: Danh sách tên classes

```bash
# Import modules trước khi load model
python -c "from improved_yolov8 import utils"

# Training với pretrained weights
yolo train \
  model=improved_yolov8/configs/yolov8-rfcbam-dynamic.yaml \
  data=dataset/data.yaml \
  pretrained=yolov8n.pt \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0
```

### 3. Validation

```bash
yolo val \
  model=runs/train/exp/weights/best.pt \
  data=dataset/data.yaml \
  imgsz=640
```

### 4. Prediction

```bash
yolo predict \
  model=runs/train/exp/weights/best.pt \
  source=dataset/test/images \
  imgsz=640
```

## Kiến trúc

### Backbone
- RFCBAMConv layers
- C2f_RFCBAM blocks
- MixSPPF at the end

### Neck
- RepGFPN for feature fusion
- AKConv for adaptive convolution
- Multi-scale feature processing

### Head
- Standard Detect head (có thể thay bằng Dynamic Head nếu có)

## Lưu ý

1. Đảm bảo import `improved_yolov8.utils` trước khi load YAML config
2. Nếu sử dụng Dynamic Head, cần import từ ultralytics.nn.extra_modules
3. Inner-IoU loss có thể được tích hợp vào training loop nếu cần
4. **Dataset Configuration**: 
   - File `dataset/data.yaml` phải có cấu hình đúng với dataset của bạn
   - Số classes (`nc`) trong `data.yaml` sẽ override giá trị trong YAML config
   - Đảm bảo đường dẫn `train` và `val` trong `data.yaml` đúng với hệ thống của bạn

## Requirements

- ultralytics >= 8.0.0
- torch >= 1.8.0
- einops
- numpy

