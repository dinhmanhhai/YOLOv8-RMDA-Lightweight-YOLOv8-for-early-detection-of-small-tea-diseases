#!/usr/bin/env bash
set -e

# Runner for TDDet theo README:
# Mặc định dùng config/đường dẫn trong README (Ultralytics):
#   model=ultralytics/cfg/models/v8/TDDet.yaml
#   data=ultralytics/dataset/chayev11/data.yaml
#   epochs=150, batch=16, imgsz=640, device=0, close_mosaic=10, workers=1, optimizer=SGD, patience=30
#
# Bạn có thể override bằng các tham số bổ sung sau dấu --.
#
# Ví dụ:
#   ./run_tdtrain.sh
#   ./run_tdtrain.sh data=../dataset/data.yaml batch=8 device=0
#   ./run_tdtrain.sh model=cfg/models/v8/yolov8-mobilenetv4.yaml data=../dataset/data.yaml
#
# Yêu cầu: đã cài ultralytics (pip install ultralytics) hoặc pip install -e . trong TDDet/codes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sử dụng lệnh yolo (CLI của ultralytics). Nếu muốn chắc chắn, có thể thay bằng: python -m ultralytics.cfg train ...
if ! command -v yolo >/dev/null 2>&1; then
  echo "Lỗi: không tìm thấy lệnh 'yolo'. Hãy cài: pip install ultralytics"
  exit 1
fi

yolo train \
  model=cfg/models/v8/yolov8-mobilenetv4-tea.yaml \
  data=../dataset/data.yaml \
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
  name=exp \
  "$@"

