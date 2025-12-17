#!/usr/bin/env bash
set -e

# Runner dùng script Python nội bộ (tránh lỗi thiếu custom layer khi gọi yolo CLI)
# Mặc định:
#   model: cfg/models/v8/yolov8-mobilenetv4-tea.yaml
#   data : ../dataset/data.yaml
#   epochs: 150, batch: 16, imgsz: 640, device: 0
# Có thể override thêm args sau dấu --.
#
# Ví dụ:
#   ./run_tdtrain.sh
#   ./run_tdtrain.sh --batch 8 --device 0
#   ./run_tdtrain.sh --wandb --wandb-project your_proj --wandb-api-key YOUR_KEY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="654322757bc621b514dc2592badff0c6eeefe6ad"

python run_training.py train \
  --model cfg/models/v8/yolov8-mobilenetv4-tea.yaml \
  --data ../dataset/data.yaml \
  --epochs 150 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --optimizer SGD \
  --patience 30 \
  --project runs/train \
  --name exp \
  --wandb-project tea-disease-detector \
  --wandb-name yolov8-mbv4-tea \
  "$@"

