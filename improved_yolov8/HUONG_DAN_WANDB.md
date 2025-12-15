# Hướng Dẫn Sử Dụng Weights & Biases (WandB)

## 1. Cài Đặt

### **Cài đặt WandB:**

```bash
pip install wandb
```

Hoặc từ `requirements.txt`:
```bash
pip install -r requirements.txt
```

### **Đăng nhập WandB:**

```bash
wandb login
```

Nhập API key từ https://wandb.ai/settings

---

## 2. Các Metrics Đã Được Log

### ✅ **Training Metrics:**

- `train/box_loss`: Box regression loss
- `train/cls_loss`: Classification loss
- `train/dfl_loss`: DFL loss (using inner_iou_loss as approximation)

### ✅ **Validation Metrics:**

- `val/box_loss`: Box regression loss
- `val/cls_loss`: Classification loss
- `val/dfl_loss`: DFL loss

### ✅ **Performance Metrics (mỗi 5 epochs):**

- `metrics/precision(B)`: Precision
- `metrics/recall(B)`: Recall
- `metrics/mAP50(B)`: mAP@0.5
- `metrics/mAP50-95(B)`: mAP@0.5:0.95

### ✅ **Other Metrics:**

- `lr`: Learning rate
- `epoch`: Current epoch

---

## 3. Cách Sử Dụng

### **3.1. Training với WandB (mặc định):**

```bash
python train.py --data data.yaml --epochs 150 --batch 16
```

WandB sẽ tự động:
- Tạo run mới
- Log tất cả metrics
- Hiển thị trên dashboard

### **3.2. Training với tên run tùy chỉnh:**

```bash
python train.py \
    --data data.yaml \
    --epochs 150 \
    --batch 16 \
    --project improved-yolov8s \
    --name experiment-1
```

### **3.3. Resume training từ WandB run:**

```bash
python train.py \
    --data data.yaml \
    --epochs 150 \
    --batch 16 \
    --resume <wandb_run_id>
```

---

## 4. Xem Kết Quả trên WandB

### **4.1. Truy cập Dashboard:**

1. Vào https://wandb.ai
2. Chọn project `improved-yolov8s` (hoặc project bạn đã chỉ định)
3. Xem các metrics trong real-time

### **4.2. Các Biểu Đồ Tự Động:**

WandB sẽ tự động tạo các biểu đồ:

1. **Loss Curves:**
   - `train/box_loss` vs `val/box_loss`
   - `train/cls_loss` vs `val/cls_loss`
   - `train/dfl_loss` vs `val/dfl_loss`

2. **Performance Metrics:**
   - `metrics/precision(B)`
   - `metrics/recall(B)`
   - `metrics/mAP50(B)`
   - `metrics/mAP50-95(B)`

3. **Learning Rate:**
   - `lr` over epochs

---

## 5. Cấu Hình WandB

### **5.1. Disable WandB (nếu không muốn dùng):**

Sửa trong `train.py`:
```python
# Comment out wandb.init() và wandb.log()
# wandb.init(...)
# wandb.log(...)
```

Hoặc set environment variable:
```bash
export WANDB_MODE=disabled
python train.py --data data.yaml --epochs 150
```

### **5.2. Offline Mode:**

```bash
export WANDB_MODE=offline
python train.py --data data.yaml --epochs 150
```

Sau đó sync:
```bash
wandb sync <offline_run_dir>
```

---

## 6. So Sánh với Hình Ảnh

### **Metrics trong hình ảnh:**

1. ✅ `train/box_loss` - Giảm từ ~0.8 → ~0.3
2. ✅ `train/cls_loss` - Giảm từ ~4.0 → ~0.5
3. ✅ `train/dfl_loss` - Giảm từ ~2.0 → ~1.1
4. ✅ `metrics/precision(B)` - Tăng từ ~0.0 → ~0.85
5. ✅ `metrics/recall(B)` - Tăng từ ~0.0 → ~0.85
6. ✅ `val/box_loss` - Giảm từ ~0.7 → ~0.25
7. ✅ `val/cls_loss` - Giảm từ ~5.5 → ~0.5
8. ✅ `val/dfl_loss` - Giảm từ ~2.2 → ~1.1
9. ✅ `metrics/mAP50(B)` - Tăng từ ~0.0 → ~0.9
10. ✅ `metrics/mAP50-95(B)` - Tăng từ ~0.0 → ~0.65

**Tất cả metrics này đã được tích hợp vào code!**

---

## 7. Ví Dụ Output

### **Console Output:**

```
Epoch 1/150
  Train Loss: 2.1234 (cls: 1.2345, box: 0.5678, dfl: 1.2345)
  Val Loss: 2.0123 (cls: 1.1234, box: 0.4567, dfl: 1.1234)
  LR: 0.010000
  - Saved best model (loss: 2.0123)
------------------------------------------------------------

Epoch 5/150
  Train Loss: 1.2345 (cls: 0.5678, box: 0.3456, dfl: 0.7890)
  Val Loss: 1.1234 (cls: 0.4567, box: 0.2345, dfl: 0.6789)
  Val mAP@0.5: 0.4567 (45.67%)
  Val mAP@0.5:0.95: 0.2345 (23.45%)
  Val Precision: 0.5678 (56.78%)
  Val Recall: 0.4321 (43.21%)
  LR: 0.009500
  ✓ Saved best model (mAP@0.5: 0.4567)
------------------------------------------------------------
```

### **WandB Dashboard:**

Tự động hiển thị:
- Real-time metrics
- Loss curves
- Performance metrics
- Learning rate schedule
- System metrics (GPU, CPU, memory)

---

## 8. Best Practices

### **8.1. Đặt tên run có ý nghĩa:**

```bash
python train.py \
    --name "baseline-640-bs16-lr0.01" \
    --project improved-yolov8s
```

### **8.2. So sánh experiments:**

```bash
# Experiment 1
python train.py --name "exp1-bs16" --batch 16

# Experiment 2
python train.py --name "exp2-bs32" --batch 32
```

Sau đó so sánh trên WandB dashboard.

### **8.3. Log thêm thông tin:**

Có thể thêm tags, notes trong WandB dashboard để ghi chú về experiment.

---

## 9. Troubleshooting

### **Vấn đề: WandB không log metrics**

**Giải pháp:**
- Kiểm tra đã login: `wandb login`
- Kiểm tra internet connection
- Kiểm tra API key

### **Vấn đề: Metrics không hiển thị**

**Giải pháp:**
- Metrics chỉ được tính mỗi 5 epochs (để tiết kiệm thời gian)
- Ở epoch cuối cùng sẽ luôn tính metrics
- Có thể sửa trong code để tính mỗi epoch (nhưng sẽ chậm hơn)

### **Vấn đề: Muốn disable WandB**

**Giải pháp:**
```bash
export WANDB_MODE=disabled
python train.py --data data.yaml --epochs 150
```

---

## 10. Tóm Tắt

### **Đã tích hợp:**

- ✅ WandB logging
- ✅ Tất cả metrics từ hình ảnh
- ✅ Training và validation losses
- ✅ Performance metrics (mAP, Precision, Recall)
- ✅ Learning rate tracking
- ✅ Auto-save best model based on mAP

### **Cách sử dụng:**

```bash
# Cài đặt
pip install wandb
wandb login

# Training
python train.py --data data.yaml --epochs 150 --batch 16

# Với tên run
python train.py --data data.yaml --epochs 150 --batch 16 --name my-experiment
```

### **Xem kết quả:**

Truy cập https://wandb.ai và chọn project `improved-yolov8s` để xem dashboard!

