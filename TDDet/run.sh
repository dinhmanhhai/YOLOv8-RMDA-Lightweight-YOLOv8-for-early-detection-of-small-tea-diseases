#!/bin/bash

# Thiết lập thông tin WandB (Đồng bộ với main.py)
export WANDB_ENTITY="works-haidinh-ptit"
export WANDB_PROJECT="tddet_mqa"
export WANDB_API_KEY="654322757bc621b514dc2592badff0c6eeefe6ad"

# Chạy train
python train.py