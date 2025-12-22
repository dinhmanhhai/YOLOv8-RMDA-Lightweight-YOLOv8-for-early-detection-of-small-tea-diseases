#!/usr/bin/env bash

# Thiết lập thông tin WandB (sửa lại cho tài khoản của bạn)
export WANDB_ENTITY=works-haidinh-ptit
export WANDB_PROJECT=improved_yolov8s
export WANDB_API_KEY="654322757bc621b514dc2592badff0c6eeefe6ad"

# Chạy train
python train.py


