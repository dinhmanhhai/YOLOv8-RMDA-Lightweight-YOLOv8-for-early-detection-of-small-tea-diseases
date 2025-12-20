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
├── train.sh               # Training script
├── val.sh                 # Validation script
├── predict.sh             # Prediction script
└── README.md
```

## Sử dụng

### Cách 1: Sử dụng Shell Scripts (Khuyến nghị)

Đã có sẵn 3 shell scripts để chạy dễ dàng:

#### Training
```bash
# Chạy với cấu hình mặc định
./improved_yolov8/train.sh

# Tùy chỉnh tham số
./improved_yolov8/train.sh -e 200 -b 32 --device 0

# Resume training
./improved_yolov8/train.sh --resume runs/train/exp/weights/last.pt

# Xem tất cả tùy chọn
./improved_yolov8/train.sh --help
```

#### Validation
```bash
# Validate với model mặc định
./improved_yolov8/val.sh

# Validate với model cụ thể
./improved_yolov8/val.sh runs/train/exp/weights/best.pt
```

#### Prediction
```bash
# Predict với mặc định
./improved_yolov8/predict.sh

# Predict với model và source cụ thể
./improved_yolov8/predict.sh runs/train/exp/weights/best.pt dataset/test/images

# Predict từ webcam
./improved_yolov8/predict.sh --source 0
```

### Cách 2: Sử dụng YOLO CLI trực tiếp

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

**Lưu và Checkpoint tự động:**
YOLO CLI tự động lưu tất cả kết quả training vào thư mục `runs/train/exp/`:
- **Checkpoints**: 
  - `runs/train/exp/weights/best.pt` - Model tốt nhất (dựa trên mAP)
  - `runs/train/exp/weights/last.pt` - Model cuối cùng của epoch cuối
- **Logs và Visualizations**:
  - `results.csv` - Bảng kết quả training (loss, mAP, precision, recall)
  - `results.png` - Biểu đồ training curves
  - `confusion_matrix.png` - Confusion matrix
  - `train_batch*.jpg` - Sample training images với labels
  - `val_batch*.jpg` - Sample validation images với predictions
  - `args.yaml` - Tất cả các tham số training đã sử dụng
- **TensorBoard logs**: Tự động lưu nếu TensorBoard được cài đặt

**Tùy chỉnh tên project và experiment:**
```bash
yolo train \
  model=improved_yolov8/configs/yolov8-rfcbam-dynamic.yaml \
  data=dataset/data.yaml \
  project=my_project \
  name=experiment_1 \
  epochs=150
# Kết quả sẽ lưu tại: my_project/experiment_1/
```

**Resume training từ checkpoint:**
```bash
yolo train \
  resume=runs/train/exp/weights/last.pt \
  epochs=200
```

**Tích hợp WandB (Weights & Biases) để log metrics:**

1. **Cài đặt WandB:**
```bash
pip install wandb
```

2. **Login vào WandB (lần đầu tiên):**
```bash
wandb login
# Nhập API key từ https://wandb.ai/authorize
```

3. **Training với WandB logging:**
```bash
# Import modules trước khi load model
python -c "from improved_yolov8 import utils"

# Training với WandB (tự động detect nếu đã login)
yolo train \
  model=improved_yolov8/configs/yolov8-rfcbam-dynamic.yaml \
  data=dataset/data.yaml \
  pretrained=yolov8n.pt \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0
```

WandB sẽ tự động log:
- **Metrics**: Loss (box, cls, dfl), mAP50, mAP50-95, Precision, Recall
- **Images**: Training batches, validation predictions, confusion matrix
- **System**: GPU/CPU usage, memory consumption
- **Hyperparameters**: Tất cả training arguments
- **Model**: Model architecture và weights (nếu enable)

4. **Tùy chỉnh WandB project và run name:**
```bash
yolo train \
  model=improved_yolov8/configs/yolov8-rfcbam-dynamic.yaml \
  data=dataset/data.yaml \
  project=my_wandb_project \
  name=experiment_rfcbam \
  epochs=150
```

5. **Disable WandB nếu không muốn sử dụng:**
```bash
yolo train \
  model=improved_yolov8/configs/yolov8-rfcbam-dynamic.yaml \
  data=dataset/data.yaml \
  wandb=False \
  epochs=150
```

**Xem logs trên WandB:**
- Truy cập https://wandb.ai để xem dashboard
- Logs sẽ tự động sync trong quá trình training
- Có thể so sánh nhiều experiments trên cùng một dashboard

### 3. Validation

```bash
yolo val \
  model=runs/train/exp/weights/best.pt \
  data=dataset/data.yaml \
  imgsz=640
```

**Kết quả validation** sẽ được lưu tại `runs/val/exp/`:
- `results.csv` - Kết quả validation metrics
- `confusion_matrix.png` - Confusion matrix
- `val_batch*.jpg` - Sample validation images với predictions

### 4. Prediction

```bash
yolo predict \
  model=runs/train/exp/weights/best.pt \
  source=dataset/test/images \
  imgsz=640
```

**Kết quả prediction** sẽ được lưu tại `runs/predict/exp/`:
- Các ảnh đã được vẽ bounding boxes và labels
- `predictions.csv` - File CSV chứa tất cả predictions (nếu có)

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
5. **Logs và Checkpoints**: 
   - Tất cả logs và checkpoints được tự động lưu trong thư mục `runs/`
   - Có thể xem training progress bằng TensorBoard: `tensorboard --logdir runs/train`
   - Checkpoint `best.pt` là model tốt nhất dựa trên validation mAP
   - Checkpoint `last.pt` có thể dùng để resume training

## Requirements

**Bắt buộc:**
- ultralytics >= 8.0.0
- torch >= 1.8.0
- einops >= 0.6.0
- numpy >= 1.24.0

**Tùy chọn (cho logging và visualization):**
- wandb >= 0.15.0 - Để log metrics lên Weights & Biases
- tensorboard - Để xem logs với TensorBoard (thường đi kèm với torch)

