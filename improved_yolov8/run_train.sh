#!/usr/bin/env bash
set -e

# Usage:
#   ./run_train.sh                 # dùng train_config.yaml mặc định
#   ./run_train.sh my_config.yaml  # truyền file config khác
#   ./run_train.sh my_config.yaml --batch 8 --epochs 50  # override thêm qua CLI

CONFIG_FILE="${1:-train_config.yaml}"
if [ "$1" != "" ]; then
  shift
fi

export WANDB_API_KEY="654322757bc621b514dc2592badff0c6eeefe6ad"

python train.py --config "$CONFIG_FILE" "$@"
