# Hướng dẫn chạy Improved YOLOv8s

## Bước 1: Cài đặt dependencies

```bash
cd improved_yolov8
pip install -r requirements.txt
```

## Bước 2: Kiểm tra cấu trúc dataset

Đảm bảo dataset có cấu trúc:
```
dataset/
├── train/
│   ├── images/  (3500 ảnh)
│   └── labels/  (3500 file .txt)
├── val/
│   ├── images/  (88 ảnh)
│   └── labels/  (88 file .txt)
└── test/
    ├── images/  (88 ảnh)
    └── labels/  (88 file .txt)
```

## Bước 3: Chạy training

### Cú pháp cơ bản:

```bash
python train.py --data data.yaml --epochs 150 --batch 16 --imgsz 640
```

### Các tham số:

- `--data`: Đường dẫn đến file data.yaml (mặc định: `data.yaml`)
- `--epochs`: Số epoch training (mặc định: 150)
- `--batch`: Batch size (mặc định: 16)
- `--imgsz`: Kích thước ảnh (mặc định: 640)
- `--device`: Device để train (`cuda` hoặc `cpu`, mặc định: tự động phát hiện)
- `--workers`: Số worker cho DataLoader (mặc định: 4)
- `--lr0`: Learning rate ban đầu (mặc định: 0.01)

### Ví dụ các câu lệnh:

#### 1. Training cơ bản:
```bash
python train.py
```

#### 2. Training với tham số tùy chỉnh:
```bash
python train.py --data data.yaml --epochs 200 --batch 8 --imgsz 640 --lr0 0.001
```

#### 3. Training trên CPU:
```bash
python train.py --device cpu --batch 4
```

#### 4. Training với batch size nhỏ (nếu GPU memory hạn chế):
```bash
python train.py --batch 8 --imgsz 416
```

#### 5. Training với nhiều workers:
```bash
python train.py --workers 8 --batch 16
```

## Lưu ý quan trọng

⚠️ **Script hiện tại chỉ tạo cấu trúc model và loss function. Để training đầy đủ, bạn cần:**

1. **Tích hợp dataset loader**: Load dataset từ YOLO format
2. **Implement training loop**: Vòng lặp training với validation
3. **Data augmentation**: Áp dụng augmentation cho training
4. **Save checkpoints**: Lưu model weights sau mỗi epoch
5. **Logging metrics**: Ghi lại metrics (loss, mAP, etc.)

## Kiểm tra model

Để kiểm tra model có thể load được không:

```python
python -c "from train import ImprovedYOLOv8s; import torch; model = ImprovedYOLOv8s(num_classes=6); x = torch.randn(1, 3, 640, 640); out = model(x); print('Model OK!')"
```

## Troubleshooting

### Lỗi: ModuleNotFoundError
**Giải pháp**: Cài đặt lại dependencies
```bash
pip install -r requirements.txt
```

### Lỗi: CUDA out of memory
**Giải pháp**: Giảm batch size hoặc image size
```bash
python train.py --batch 4 --imgsz 416
```

### Lỗi: Cannot find dataset
**Giải pháp**: Kiểm tra đường dẫn trong `data.yaml` và đảm bảo dataset tồn tại

