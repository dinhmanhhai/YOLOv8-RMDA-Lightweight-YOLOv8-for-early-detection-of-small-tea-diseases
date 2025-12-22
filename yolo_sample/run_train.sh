#!/usr/bin/env bash

# Thiết lập thông tin WandB (sửa lại cho tài khoản của bạn)
export WANDB_ENTITY=works-haidinh-ptit
export WANDB_PROJECT=improved_yolov8s

# Chạy train
python train.py


