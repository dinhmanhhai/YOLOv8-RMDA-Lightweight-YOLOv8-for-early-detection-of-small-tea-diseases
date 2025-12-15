# Hướng Dẫn Đánh Giá Model Detection

## 1. Các Metrics Đã Được Bổ Sung

### ✅ **mAP Calculation**
- **File**: `models/utils/metrics.py`
- **Function**: `calculate_map()`
- **Tính năng**:
  - mAP@0.5 (IoU threshold = 0.5)
  - mAP@0.5:0.95 (average over multiple IoU thresholds)
  - AP per class
  - Precision-Recall curve

### ✅ **Precision/Recall Calculation**
- **File**: `models/utils/metrics.py`
- **Function**: `calculate_precision_recall()`
- **Tính năng**:
  - Precision (độ chính xác)
  - Recall (độ bao phủ)
  - F1-Score
  - True Positives, False Positives, False Negatives

### ✅ **Evaluation Script**
- **File**: `evaluate.py`
- **Tính năng**:
  - Đánh giá model trên test/validation set
  - Tính tất cả metrics
  - Hiển thị kết quả chi tiết

---

## 2. Cách Sử Dụng

### **2.1. Đánh Giá Model Sau Khi Training**

```bash
# Đánh giá model checkpoint
python evaluate.py --model weights/best.pt --data data.yaml --batch 8

# Với các tham số tùy chỉnh
python evaluate.py \
    --model weights/best.pt \
    --data data.yaml \
    --batch 8 \
    --imgsz 640 \
    --conf 0.25 \
    --iou 0.45 \
    --workers 4
```

**Tham số:**
- `--model`: Đường dẫn đến model checkpoint (bắt buộc)
- `--data`: File YAML dataset (mặc định: `data.yaml`)
- `--batch`: Batch size (mặc định: 8)
- `--imgsz`: Kích thước ảnh (mặc định: 640)
- `--conf`: Confidence threshold cho NMS (mặc định: 0.25)
- `--iou`: IoU threshold cho NMS và mAP (mặc định: 0.45)
- `--workers`: Số workers cho DataLoader (mặc định: 4)
- `--device`: Device (cuda/cpu, mặc định: auto-detect)

---

### **2.2. Metrics Trong Training**

Metrics đã được tích hợp vào `train.py`:
- **Tự động tính mAP, Precision, Recall** mỗi 5 epochs
- **Tự động tính metrics ở epoch cuối cùng**

**Output trong training:**
```
Epoch 5/150
  Train Loss: 1.2345 (cls: 0.1234, box: 0.5678)
  Val Loss: 1.1234
  Val mAP@0.5: 0.4567 (45.67%)
  Val Precision: 0.5678 (56.78%)
  Val Recall: 0.4321 (43.21%)
  LR: 0.001000
```

---

## 3. Output của Evaluation Script

### **3.1. Overall Metrics**

```
EVALUATION RESULTS
============================================================

Overall Metrics:
  mAP@0.5: 0.7234 (72.34%)
  Precision: 0.7891 (78.91%)
  Recall: 0.7123 (71.23%)
  F1-Score: 0.7489 (74.89%)

  True Positives: 1234
  False Positives: 345
  False Negatives: 567
```

### **3.2. Per-Class AP**

```
Per-Class AP@0.5:
  ✅ Tea algae leaf spot: 0.7567 (75.67%)
  ✅ Tea cake: 0.7234 (72.34%)
  ✅ Tea cloud leaf blight: 0.6891 (68.91%)
  ✅ Tea exobasidium blight: 0.7123 (71.23%)
  ✅ Tea red rust: 0.7456 (74.56%)
  ✅ Tea red scab: 0.7012 (70.12%)
```

### **3.3. mAP@0.5:0.95**

```
Calculating mAP@0.5:0.95...
  mAP@0.5:0.95: 0.5234 (52.34%)
```

### **3.4. Evaluation Summary**

```
Evaluation Summary:
  ✅ mAP@0.5 >= 0.7 - Model is EXCELLENT!
  ✅ Precision and Recall >= 0.7 - Model is GOOD!
```

---

## 4. Ý Nghĩa Các Metrics

### **4.1. mAP@0.5**
- **Ý nghĩa**: Mean Average Precision với IoU threshold = 0.5
- **Mục tiêu**: > 0.5 (50%) là tốt, > 0.7 (70%) là rất tốt
- **Công thức**: Trung bình AP của tất cả classes

### **4.2. mAP@0.5:0.95**
- **Ý nghĩa**: Mean Average Precision với IoU từ 0.5 đến 0.95
- **Mục tiêu**: > 0.3 (30%) là tốt, > 0.5 (50%) là rất tốt
- **Công thức**: Trung bình mAP@0.5, mAP@0.55, ..., mAP@0.95

### **4.3. Precision**
- **Ý nghĩa**: Tỷ lệ predictions đúng trong tất cả predictions
- **Mục tiêu**: > 0.7 (70%) là tốt
- **Công thức**: TP / (TP + FP)

### **4.4. Recall**
- **Ý nghĩa**: Tỷ lệ objects thực tế được detect
- **Mục tiêu**: > 0.7 (70%) là tốt
- **Công thức**: TP / (TP + FN)

### **4.5. F1-Score**
- **Ý nghĩa**: Cân bằng giữa Precision và Recall
- **Mục tiêu**: > 0.7 (70%) là tốt
- **Công thức**: 2 * (Precision * Recall) / (Precision + Recall)

---

## 5. Checklist Đánh Giá Model Tốt

### ✅ **Model TỐT khi:**

- [x] **mAP@0.5 > 0.7** (70%)
- [x] **mAP@0.5:0.95 > 0.5** (50%)
- [x] **Precision > 0.75** (75%)
- [x] **Recall > 0.75** (75%)
- [x] **F1-Score > 0.75** (75%)
- [x] **Per-class AP > 0.6** cho mỗi class
- [x] **Không có class bị bỏ quên** (AP = 0)

### ❌ **Model CHƯA TỐT khi:**

- [ ] mAP@0.5 < 0.5
- [ ] Precision < 0.6 hoặc Recall < 0.6
- [ ] Có class bị bỏ quên (AP = 0)
- [ ] Nhiều false positives hoặc false negatives

---

## 6. So Sánh với Baseline

### **YOLOv8s Baseline (ước tính):**
- mAP@0.5: ~0.6-0.7
- mAP@0.5:0.95: ~0.4-0.5
- Precision: ~0.7-0.8
- Recall: ~0.7-0.8

### **Improved YOLOv8s (mục tiêu):**
- mAP@0.5: **> 0.7** (cải thiện so với baseline)
- mAP@0.5:0.95: **> 0.5** (cải thiện so với baseline)
- Precision: **> 0.75**
- Recall: **> 0.75**

---

## 7. Troubleshooting

### **Vấn đề: mAP = 0.0**

**Nguyên nhân có thể:**
- Model chưa được train đủ
- Confidence threshold quá cao
- IoU threshold quá cao
- Decode boxes không đúng

**Giải pháp:**
- Giảm `--conf` xuống 0.1-0.15
- Giảm `--iou` xuống 0.3-0.4
- Kiểm tra model output format

### **Vấn đề: Precision cao nhưng Recall thấp**

**Nguyên nhân:**
- Model quá conservative (chỉ predict khi rất chắc chắn)
- Confidence threshold quá cao

**Giải pháp:**
- Giảm confidence threshold
- Tăng recall bằng cách giảm NMS threshold

### **Vấn đề: Recall cao nhưng Precision thấp**

**Nguyên nhân:**
- Model predict quá nhiều (nhiều false positives)
- Confidence threshold quá thấp

**Giải pháp:**
- Tăng confidence threshold
- Tăng NMS threshold để loại bỏ duplicate detections

---

## 8. Files Đã Tạo

1. **`models/utils/metrics.py`**:
   - `xywh2xyxy()`: Convert boxes format
   - `box_iou()`: Calculate IoU
   - `non_max_suppression()`: NMS for detections
   - `calculate_ap()`: Calculate Average Precision
   - `calculate_map()`: Calculate mAP
   - `calculate_precision_recall()`: Calculate Precision/Recall

2. **`evaluate.py`**:
   - Script đánh giá model
   - Tính tất cả metrics
   - Hiển thị kết quả chi tiết

3. **`train.py`** (đã cập nhật):
   - Tích hợp metrics vào training
   - Tự động tính mAP mỗi 5 epochs

---

## 9. Tóm Tắt

### **Đã bổ sung:**
- ✅ mAP calculation (mAP@0.5 và mAP@0.5:0.95)
- ✅ Precision/Recall calculation
- ✅ F1-Score calculation
- ✅ Per-class AP
- ✅ Evaluation script (`evaluate.py`)
- ✅ Tích hợp metrics vào training

### **Cách sử dụng:**
```bash
# Đánh giá model
python evaluate.py --model weights/best.pt --data data.yaml

# Training sẽ tự động tính metrics mỗi 5 epochs
python train.py --data data.yaml --epochs 150 --batch 16
```

### **Kết quả mong đợi:**
- mAP@0.5 > 0.7
- Precision > 0.75
- Recall > 0.75
- F1-Score > 0.75

