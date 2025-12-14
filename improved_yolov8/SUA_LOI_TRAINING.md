# Sửa lỗi Training Script

## Các thay đổi đã thực hiện

### 1. Tạo Dataset Loader (`models/data/dataset.py`)
- ✅ Class `YOLODataset` để load images và labels từ YOLO format
- ✅ Hỗ trợ resize và padding về kích thước cố định (640x640)
- ✅ Hỗ trợ data augmentation cơ bản
- ✅ Load labels từ file .txt với format: `class_id x_center y_center width height`

### 2. Tạo Detection Loss (`models/utils/detection_loss.py`)
- ✅ Class `DetectionLoss` để tính loss cho object detection
- ✅ Kết hợp Classification loss và Box regression loss
- ✅ Match targets với grid cells (simplified version)
- ✅ Tính loss trên các positive positions

### 3. Cập nhật Training Script (`train.py`)
- ✅ Load dataset từ YAML config file
- ✅ Tạo DataLoader cho train và validation
- ✅ Implement training loop đầy đủ với:
  - Forward pass
  - Loss computation
  - Backward pass
  - Optimizer step
  - Learning rate scheduling
- ✅ Validation loop
- ✅ Save checkpoints (best và last)
- ✅ Progress bar với tqdm
- ✅ Log metrics (loss, cls_loss, box_loss)

## Cách sử dụng

```bash
cd improved_yolov8
python train.py --data data.yaml --epochs 150 --batch 16 --imgsz 640
```

## Các tham số

- `--data`: Đường dẫn đến file YAML config dataset
- `--epochs`: Số epochs để training
- `--batch`: Batch size
- `--imgsz`: Kích thước ảnh input
- `--device`: Device (cuda/cpu), mặc định tự động detect
- `--workers`: Số workers cho DataLoader
- `--lr0`: Initial learning rate

## Cấu trúc output

```
runs/train/
└── weights/
    ├── best.pt    # Best model (lowest validation loss)
    └── last.pt    # Last checkpoint
```

## Lưu ý

1. **Loss Function**: Hiện tại sử dụng simplified version của loss function. Để có kết quả tốt hơn, nên implement:
   - Task Aligned Learning (TAL) cho target assignment
   - Proper anchor matching
   - Distribution Focal Loss (DFL) cho box regression

2. **Data Augmentation**: Hiện tại chỉ có ColorJitter. Có thể thêm:
   - Mosaic augmentation
   - MixUp
   - Random flip, rotate, etc.

3. **Model Output Format**: Model hiện tại output format `(batch, num_classes+1+4, H, W)` với:
   - `num_classes`: Classification logits
   - `1`: Center regression
   - `4`: Box regression (x, y, w, h)

## Các file đã tạo/sửa

1. `models/data/dataset.py` - Dataset loader
2. `models/data/__init__.py` - Package init
3. `models/utils/detection_loss.py` - Loss function
4. `train.py` - Training script (đã cập nhật)

## Kiểm tra

Sau khi chạy, script sẽ:
- Load dataset từ `data.yaml`
- Tạo model với số parameters
- Training với progress bar
- Validate sau mỗi epoch
- Save checkpoints

Nếu có lỗi, kiểm tra:
- Dataset paths trong `data.yaml` có đúng không
- Images và labels có tồn tại không
- CUDA có available không (nếu dùng GPU)

