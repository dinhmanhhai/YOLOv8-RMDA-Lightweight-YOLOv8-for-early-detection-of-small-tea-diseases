# Tổng Hợp Metrics Đã Có và Đã Bổ Sung

## So Sánh với Bảng Metrics trong Ảnh

### Bảng Metrics trong Ảnh:

| Metric | Giá trị | Đã có trong code? |
|--------|---------|-------------------|
| **P/% (Precision)** | 71.86% | ✅ **CÓ** - `metrics/precision(B)` |
| **R/% (Recall)** | 72.38% | ✅ **CÓ** - `metrics/recall(B)` |
| **mAP@0.5/%** | 80.34% | ✅ **CÓ** - `metrics/mAP50(B)` |
| **FPS/S** | 177 | ✅ **ĐÃ BỔ SUNG** - `metrics/FPS` |

---

## Chi Tiết Các Metrics

### ✅ **1. Precision (P/%)**

**Trong code:**
- **Tên trong WandB**: `metrics/precision(B)`
- **Tính toán**: Mỗi 5 epochs hoặc epoch cuối cùng
- **Công thức**: `TP / (TP + FP)`
- **Ý nghĩa**: Tỷ lệ predictions đúng trong tất cả predictions

**Vị trí trong code:**
```python
# train.py line 462
'metrics/precision(B)': val_precision
```

---

### ✅ **2. Recall (R/%)**

**Trong code:**
- **Tên trong WandB**: `metrics/recall(B)`
- **Tính toán**: Mỗi 5 epochs hoặc epoch cuối cùng
- **Công thức**: `TP / (TP + FN)`
- **Ý nghĩa**: Tỷ lệ objects thực tế được detect

**Vị trí trong code:**
```python
# train.py line 463
'metrics/recall(B)': val_recall
```

---

### ✅ **3. mAP@0.5 (mAP@0.5/%)**

**Trong code:**
- **Tên trong WandB**: `metrics/mAP50(B)`
- **Tính toán**: Mỗi 5 epochs hoặc epoch cuối cùng
- **Công thức**: Mean Average Precision với IoU threshold = 0.5
- **Ý nghĩa**: Metric tổng hợp cho object detection

**Vị trí trong code:**
```python
# train.py line 464
'metrics/mAP50(B)': val_map_50
```

---

### ✅ **4. FPS (FPS/S)** - ĐÃ BỔ SUNG

**Trong code:**
- **Tên trong WandB**: `metrics/FPS`
- **Tính toán**: 
  - Đo một lần ở đầu training (sau khi tạo model)
  - Log lại mỗi 5 epochs hoặc epoch cuối cùng
- **Công thức**: `num_runs / total_time`
- **Ý nghĩa**: Tốc độ inference (frames per second)

**Vị trí trong code:**
```python
# train.py - Đo FPS sau khi tạo model
# Warmup + measure 100 runs
fps = num_runs / total_time

# Log vào WandB
wandb.log({'metrics/FPS': fps})  # Initial log
'metrics/FPS': fps  # Log mỗi 5 epochs
```

**Chi tiết implementation:**
1. Warmup: 10 runs để model ổn định
2. Measure: 100 runs để tính FPS chính xác
3. Synchronize CUDA nếu dùng GPU
4. Tính FPS = số runs / thời gian

---

## Các Metrics Bổ Sung Khác

Ngoài các metrics trong bảng, code còn có:

### **Training Metrics:**
- `train/box_loss`: Box regression loss
- `train/cls_loss`: Classification loss
- `train/dfl_loss`: DFL loss

### **Validation Metrics:**
- `val/box_loss`: Box regression loss
- `val/cls_loss`: Classification loss
- `val/dfl_loss`: DFL loss

### **Other Metrics:**
- `metrics/mAP50-95(B)`: mAP@0.5:0.95 (average over multiple IoU thresholds)
- `lr`: Learning rate

---

## Cách Xem Metrics trên WandB

### **1. Truy cập Dashboard:**

1. Vào https://wandb.ai
2. Chọn project `improved-yolov8s`
3. Chọn run bạn muốn xem

### **2. Các Metrics sẽ hiển thị:**

#### **Performance Metrics Panel:**
- `metrics/precision(B)` - Precision curve
- `metrics/recall(B)` - Recall curve
- `metrics/mAP50(B)` - mAP@0.5 curve
- `metrics/mAP50-95(B)` - mAP@0.5:0.95 curve
- `metrics/FPS` - FPS value (constant line)

#### **Loss Metrics Panel:**
- `train/box_loss` vs `val/box_loss`
- `train/cls_loss` vs `val/cls_loss`
- `train/dfl_loss` vs `val/dfl_loss`

---

## Output Mẫu

### **Console Output:**

```
Measuring inference speed (FPS)...
Inference speed: 156.23 FPS
============================================================

Epoch 1/150
  Train Loss: 2.1234 (cls: 1.2345, box: 0.5678, dfl: 1.2345)
  Val Loss: 2.0123 (cls: 1.1234, box: 0.4567, dfl: 1.1234)
  LR: 0.010000
------------------------------------------------------------

Epoch 5/150
  Train Loss: 1.2345 (cls: 0.5678, box: 0.3456, dfl: 0.7890)
  Val Loss: 1.1234 (cls: 0.4567, box: 0.2345, dfl: 0.6789)
  Val mAP@0.5: 0.4567 (45.67%)
  Val mAP@0.5:0.95: 0.2345 (23.45%)
  Val Precision: 0.5678 (56.78%)
  Val Recall: 0.4321 (43.21%)
  FPS: 156.23
  LR: 0.009500
------------------------------------------------------------
```

### **WandB Dashboard:**

Tự động hiển thị:
- ✅ Precision curve
- ✅ Recall curve
- ✅ mAP@0.5 curve
- ✅ mAP@0.5:0.95 curve
- ✅ **FPS value** (mới bổ sung)

---

## So Sánh với Baseline

### **YOLOV8 + MobileNetV3 (từ bảng):**

| Metric | Baseline | Improved YOLOv8s (Mục tiêu) |
|--------|----------|------------------------------|
| **Precision** | 71.86% | > 75% |
| **Recall** | 72.38% | > 75% |
| **mAP@0.5** | 80.34% | > 80% |
| **FPS** | 177 | Tối ưu tùy hardware |

### **Cách So Sánh:**

1. Sau khi training xong, xem metrics trên WandB
2. So sánh với baseline:
   - Precision: Có cao hơn 71.86% không?
   - Recall: Có cao hơn 72.38% không?
   - mAP@0.5: Có cao hơn 80.34% không?
   - FPS: So sánh tốc độ (phụ thuộc vào hardware)

---

## Tóm Tắt

### ✅ **Đã có đầy đủ:**

1. ✅ **Precision** - `metrics/precision(B)`
2. ✅ **Recall** - `metrics/recall(B)`
3. ✅ **mAP@0.5** - `metrics/mAP50(B)`
4. ✅ **FPS** - `metrics/FPS` (vừa bổ sung)

### **Cách sử dụng:**

```bash
# Training với tất cả metrics
python train.py --data data.yaml --epochs 150 --batch 16

# Xem metrics trên WandB
# Truy cập https://wandb.ai và chọn project improved-yolov8s
```

### **Kết quả:**

- Tất cả metrics từ bảng đã được log vào WandB
- FPS được đo chính xác với warmup và multiple runs
- Metrics được tính mỗi 5 epochs để tiết kiệm thời gian
- Epoch cuối cùng luôn tính tất cả metrics

