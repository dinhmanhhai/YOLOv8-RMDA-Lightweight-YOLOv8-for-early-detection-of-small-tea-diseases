# Hướng dẫn chạy TDDet với YOLO

## Tổng quan
TDDet là một framework YOLO được fork từ Ultralytics YOLO, được tối ưu hóa để phát hiện bệnh trên lá trà. Framework này sử dụng MobileNetV4 làm backbone.

## Yêu cầu hệ thống

- Python 3.9
- CUDA 12.2 (nếu dùng GPU)
- PyTorch 1.8+

## Cài đặt

### 1. Cài đặt Ultralytics (nếu chưa có)
```bash
pip install ultralytics
```

### 2. Cài đặt TDDet package
Có 2 cách:

#### Cách 1: Cài đặt như một package (khuyến nghị)
```bash
cd TDDet/codes
pip install -e .
```

#### Cách 2: Thêm vào PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/haidm/Desktop/AI/CODE/YoloV8 detector/TDDet/codes"
```

Hoặc trong Python:
```python
import sys
sys.path.append('/Users/haidm/Desktop/AI/CODE/YoloV8 detector/TDDet/codes')
```

## Cấu trúc lệnh YOLO

Cú pháp chung:
```bash
yolo TASK MODE ARGS
```

Trong đó:
- **TASK** (tùy chọn): `detect`, `segment`, `classify`, `pose`, `obb`
- **MODE** (bắt buộc): `train`, `val`, `predict`, `export`, `track`, `benchmark`
- **ARGS**: Các tham số tùy chỉnh dạng `key=value`

## Các lệnh chính

### 1. Training (Huấn luyện)

#### Cú pháp từ README:
```bash
yolo train model=ultralytics/cfg/models/v8/TDDet.yaml data=ultralytics/dataset/chayev11/data.yaml device=0 cache=False imgsz=640 epochs=150 batch=16 close_mosaic=10 workers=1 optimizer=SGD patience=30 project=runs/train name=exp
```

#### Cú pháp cho dataset của bạn:
```bash
# Từ thư mục gốc project
cd TDDet/codes

# Chạy training
python -m ultralytics.cfg train \
    model=cfg/models/v8/yolov8-mobilenetv4.yaml \
    data=../../dataset/data.yaml \
    device=0 \
    cache=False \
    imgsz=640 \
    epochs=150 \
    batch=16 \
    close_mosaic=10 \
    workers=1 \
    optimizer=SGD \
    patience=30 \
    project=runs/train \
    name=exp
```

#### Hoặc sử dụng Python script:
```python
from ultralytics import YOLO

# Load model
model = YOLO('cfg/models/v8/yolov8-mobilenetv4.yaml')

# Train
model.train(
    data='../../dataset/data.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    optimizer='SGD',
    patience=30,
    project='runs/train',
    name='exp'
)
```

### 2. Validation (Đánh giá)

#### Từ README:
```bash
yolo val model=runs/train/exp/weights/best.pt data=ultralytics/dataset/chayev11/data.yaml split=test imgsz=640 batch=16 project=runs/val name=exp
```

#### Cho dataset của bạn:
```bash
cd TDDet/codes

python -m ultralytics.cfg val \
    model=runs/train/exp/weights/best.pt \
    data=../../dataset/data.yaml \
    split=test \
    imgsz=640 \
    batch=16 \
    project=runs/val \
    name=exp
```

#### Hoặc Python:
```python
from ultralytics import YOLO

model = YOLO('runs/train/exp/weights/best.pt')
model.val(data='../../dataset/data.yaml', split='test')
```

### 3. Prediction (Dự đoán)

#### Command line:
```bash
cd TDDet/codes

python -m ultralytics.cfg predict \
    model=runs/train/exp/weights/best.pt \
    source=../../dataset/test/images \
    imgsz=640 \
    conf=0.25 \
    save=True
```

#### Python:
```python
from ultralytics import YOLO

model = YOLO('runs/train/exp/weights/best.pt')
results = model.predict(
    source='../../dataset/test/images',
    imgsz=640,
    conf=0.25,
    save=True
)
```

### 4. Export model

```bash
python -m ultralytics.cfg export \
    model=runs/train/exp/weights/best.pt \
    format=onnx \
    imgsz=640
```

## Cấu hình dataset

File `data.yaml` cần có cấu trúc:
```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images  # optional
nc: 6  # number of classes
names: ['Tea algae leaf spot', 'Tea cake', 'Tea cloud leaf blight', 'Tea exobasidium blight', 'Tea red rust', 'Tea red scab']
```

**Lưu ý**: Cần cập nhật đường dẫn trong `dataset/data.yaml` cho đúng với hệ thống của bạn.

## Các tham số quan trọng

- `model`: Đường dẫn đến file model config (.yaml) hoặc weights (.pt)
- `data`: Đường dẫn đến file data.yaml
- `epochs`: Số epoch training
- `batch`: Batch size
- `imgsz`: Kích thước ảnh (640, 1280, ...)
- `device`: GPU device (0, 1, ...) hoặc 'cpu'
- `workers`: Số worker cho DataLoader
- `optimizer`: Optimizer ('SGD', 'Adam', 'AdamW', ...)
- `lr0`: Learning rate ban đầu
- `patience`: Số epoch không cải thiện trước khi early stopping
- `project`: Thư mục lưu kết quả
- `name`: Tên experiment

## Cấu trúc thư mục sau training

```
runs/
├── train/
│   └── exp/
│       ├── weights/
│       │   ├── best.pt      # Best model
│       │   └── last.pt      # Last checkpoint
│       ├── results.png      # Training curves
│       ├── confusion_matrix.png
│       └── ...
└── val/
    └── exp/
        └── ...
```

## Troubleshooting

### Lỗi: ModuleNotFoundError: No module named 'ultralytics'
**Giải pháp**: 
```bash
cd TDDet/codes
pip install -e .
```

### Lỗi: Cannot find data.yaml
**Giải pháp**: Kiểm tra đường dẫn trong file data.yaml và đảm bảo đường dẫn đúng

### Lỗi: CUDA out of memory
**Giải pháp**: Giảm batch size hoặc imgsz
```bash
batch=8  # thay vì 16
imgsz=416  # thay vì 640
```

### Lỗi: FileNotFoundError khi load model
**Giải pháp**: Kiểm tra đường dẫn model, có thể cần dùng đường dẫn tuyệt đối

## Ví dụ script Python hoàn chỉnh

```python
import sys
from pathlib import Path

# Thêm TDDet vào path
tddet_path = Path(__file__).parent / "TDDet" / "codes"
sys.path.insert(0, str(tddet_path))

from ultralytics import YOLO

# 1. Training
print("Bắt đầu training...")
model = YOLO('cfg/models/v8/yolov8-mobilenetv4.yaml')
model.train(
    data='../../dataset/data.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    project='runs/train',
    name='tea_disease_detector'
)

# 2. Validation
print("Bắt đầu validation...")
model = YOLO('runs/train/tea_disease_detector/weights/best.pt')
metrics = model.val(data='../../dataset/data.yaml', split='test')
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# 3. Prediction
print("Bắt đầu prediction...")
results = model.predict(
    source='../../dataset/test/images',
    save=True,
    conf=0.25
)
```

## Lưu ý quan trọng

1. **Đường dẫn**: Tất cả đường dẫn trong README là ví dụ, cần điều chỉnh cho phù hợp với hệ thống của bạn
2. **Model config**: File `yolov8-mobilenetv4.yaml` có thể cần điều chỉnh số class (nc) cho phù hợp
3. **Data format**: Dataset phải ở định dạng YOLO (images + labels)
4. **GPU**: Nếu không có GPU, đặt `device=cpu`

