# TDDet - Tea Disease Detector

TDDet là một framework phát hiện bệnh trên lá trà nhẹ và hiệu quả, được xây dựng dựa trên YOLO.

## Framework

Framework sử dụng MobileNetV4 làm backbone và được tối ưu hóa cho việc phát hiện bệnh trên lá trà.

## Yêu cầu

- Python 3.9
- CUDA 12.2 (nếu dùng GPU)
- PyTorch 1.8+

## Cài đặt

### 1. Cài đặt Ultralytics
```bash
pip install ultralytics
```

### 2. Cài đặt TDDet
```bash
cd TDDet/codes
pip install -e .
```

## Cách sử dụng

### Cách 1: Sử dụng script Python (Khuyến nghị)

#### Training
```bash
cd TDDet
python run_training.py train \
    --model cfg/models/v8/yolov8-mobilenetv4.yaml \
    --data ../dataset/data.yaml \
    --epochs 150 \
    --batch 16 \
    --device 0
```

#### Validation
```bash
python run_training.py val \
    --model runs/train/exp/weights/best.pt \
    --data ../dataset/data.yaml \
    --split test
```

#### Prediction
```bash
python run_training.py predict \
    --model runs/train/exp/weights/best.pt \
    --source ../dataset/test/images
```

### Cách 2: Sử dụng lệnh YOLO trực tiếp

Sau khi cài đặt, bạn có thể sử dụng lệnh `yolo`:

#### Training
```bash
cd TDDet/codes
python -m ultralytics.cfg train \
    model=cfg/models/v8/yolov8-mobilenetv4.yaml \
    data=../../dataset/data.yaml \
    epochs=150 \
    batch=16 \
    device=0
```

#### Validation
```bash
python -m ultralytics.cfg val \
    model=runs/train/exp/weights/best.pt \
    data=../../dataset/data.yaml \
    split=test
```

#### Prediction
```bash
python -m ultralytics.cfg predict \
    model=runs/train/exp/weights/best.pt \
    source=../../dataset/test/images
```

### Cách 3: Sử dụng Python API

```python
import sys
from pathlib import Path

# Thêm TDDet vào path
sys.path.insert(0, str(Path("TDDet/codes").absolute()))

from ultralytics import YOLO

# Training
model = YOLO('cfg/models/v8/yolov8-mobilenetv4.yaml')
model.train(
    data='../../dataset/data.yaml',
    epochs=150,
    batch=16,
    device=0
)

# Validation
model = YOLO('runs/train/exp/weights/best.pt')
model.val(data='../../dataset/data.yaml', split='test')

# Prediction
results = model.predict(source='../../dataset/test/images')
```

## Cấu hình Dataset

File `data.yaml` cần có cấu trúc:
```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images
nc: 6
names: ['Tea algae leaf spot', 'Tea cake', 'Tea cloud leaf blight', 'Tea exobasidium blight', 'Tea red rust', 'Tea red scab']
```

**Lưu ý**: Cần cập nhật đường dẫn trong `dataset/data.yaml` cho đúng với hệ thống của bạn.

## Các tham số quan trọng

- `epochs`: Số epoch training (mặc định: 150)
- `batch`: Batch size (mặc định: 16)
- `imgsz`: Kích thước ảnh (mặc định: 640)
- `device`: GPU device (0, 1, ...) hoặc 'cpu'
- `optimizer`: Optimizer ('SGD', 'Adam', ...)
- `patience`: Early stopping patience (mặc định: 30)

## Kết quả

Sau khi training, kết quả sẽ được lưu trong:
```
runs/train/exp/
├── weights/
│   ├── best.pt      # Model tốt nhất
│   └── last.pt      # Checkpoint cuối
├── results.png      # Đồ thị training
└── confusion_matrix.png
```

## Xem thêm

Xem file `HUONG_DAN_CHAY.md` để biết hướng dẫn chi tiết hơn.

