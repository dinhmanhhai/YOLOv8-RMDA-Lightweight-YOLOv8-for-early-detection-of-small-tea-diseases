#!/bin/bash

# Thiết lập thông tin WandB (Đồng bộ với main.py)
export WANDB_ENTITY="works-haidinh-ptit"
export WANDB_PROJECT="tddet_mqa"
export WANDB_API_KEY="654322757bc621b514dc2592badff0c6eeefe6ad"

# Run YOLOv8 training using CLI
yolo detect train \
    model="ultralytics/cfg/models/v8/yolov8-mobilenetv4-mqa.yaml" \
    data="data.yaml" \
    epochs=150 \
    patience=50 \
    imgsz=640 \
    device=0 \
    batch=16 \
    project="works-haidinh-ptit" \
    name="tddet_mqa" \
    save=True \
    save_json=True \
    cache=False \
    workers=8 \
    exist_ok=False \
    pretrained=True \
    optimizer="auto" \
    verbose=True \
    seed=0 \
    deterministic=True \
    single_cls=False \
    rect=False \
    cos_lr=False \
    close_mosaic=10 \
    resume=False \
    amp=True \
    fraction=1.0 \
    profile=False \
    freeze=None \
    overlap_mask=True \
    mask_ratio=4 \
    dropout=0.0 \
    val=True \
    split=val \
    save_hybrid=False \
    conf=None \
    iou=0.7 \
    max_det=300 \
    half=False \
    dnn=False \
    plots=True
