#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để chạy training YOLO với TDDet + optional WandB logging
"""

import os
import sys
from pathlib import Path

# Thêm TDDet vào Python path
current_dir = Path(__file__).parent.absolute()
tddet_codes = current_dir / "codes"
sys.path.insert(0, str(tddet_codes))

try:
    from ultralytics import YOLO
except ImportError:
    print("Lỗi: Không tìm thấy module ultralytics")
    print("Vui lòng cài đặt:")
    print("  cd TDDet/codes")
    print("  pip install -e .")
    sys.exit(1)


def train_model(
    model_config="cfg/models/v8/yolov8-mobilenetv4.yaml",
    data_yaml="../../dataset/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    optimizer="SGD",
    patience=30,
    project="runs/train",
    name="exp",
    wandb_enable=False,
    wandb_project=None,
    wandb_name=None,
    wandb_entity=None,
    wandb_api_key=None,
    **kwargs
):
    """
    Huấn luyện model YOLO
    
    Args:
        model_config: Đường dẫn đến file config model (.yaml)
        data_yaml: Đường dẫn đến file data.yaml
        epochs: Số epoch
        imgsz: Kích thước ảnh
        batch: Batch size
        device: GPU device (0, 1, ...) hoặc 'cpu'
        optimizer: Optimizer ('SGD', 'Adam', ...)
        patience: Early stopping patience
        project: Thư mục project
        name: Tên experiment
        wandb_enable: Bật/tắt WandB logging
        wandb_project/name/entity/api_key: Cấu hình WandB (tuỳ chọn)
        **kwargs: Các tham số khác
    """
    print("=" * 60)
    print("BẮT ĐẦU TRAINING TDDet")
    print("=" * 60)
    
    # Đảm bảo đường dẫn tuyệt đối
    model_config = Path(model_config)
    if not model_config.is_absolute():
        model_config = tddet_codes / model_config
    
    data_yaml = Path(data_yaml)
    if not data_yaml.is_absolute():
        data_yaml = current_dir.parent / data_yaml
    
    print(f"Model config: {model_config}")
    print(f"Data YAML: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    if wandb_enable:
        print("WandB: ENABLED")
    else:
        print("WandB: disabled")
    print("=" * 60)
    
    # Kiểm tra file tồn tại
    if not model_config.exists():
        print(f"Lỗi: Không tìm thấy file model config: {model_config}")
        return
    
    if not data_yaml.exists():
        print(f"Lỗi: Không tìm thấy file data.yaml: {data_yaml}")
        return
    
    # Optional WandB setup (Ultralytics auto-logs nếu wandb có mặt)
    if wandb_enable:
        os.environ.pop("WANDB_DISABLED", None)
        if wandb_project:
            os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_name:
            os.environ["WANDB_NAME"] = wandb_name
        if wandb_entity:
            os.environ["WANDB_ENTITY"] = wandb_entity
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ.setdefault("WANDB_MODE", "online")
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Load model
    print("Đang load model...")
    model = YOLO(str(model_config))
    
    # Training
    print("Bắt đầu training...")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        optimizer=optimizer,
        patience=patience,
        project=project,
        name=name,
        **kwargs
    )
    
    print("=" * 60)
    print("TRAINING HOÀN TẤT!")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print("=" * 60)


def validate_model(
    model_path,
    data_yaml="../../dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=16,
    project="runs/val",
    name="exp"
):
    """
    Validate model
    
    Args:
        model_path: Đường dẫn đến file weights (.pt)
        data_yaml: Đường dẫn đến file data.yaml
        split: Split để validate ('test', 'val')
        imgsz: Kích thước ảnh
        batch: Batch size
        project: Thư mục project
        name: Tên experiment
    """
    print("=" * 60)
    print("BẮT ĐẦU VALIDATION")
    print("=" * 60)
    
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = current_dir / model_path
    
    data_yaml = Path(data_yaml)
    if not data_yaml.is_absolute():
        data_yaml = current_dir.parent / data_yaml
    
    print(f"Model: {model_path}")
    print(f"Data YAML: {data_yaml}")
    print(f"Split: {split}")
    print("=" * 60)
    
    if not model_path.exists():
        print(f"Lỗi: Không tìm thấy file model: {model_path}")
        return
    
    if not data_yaml.exists():
        print(f"Lỗi: Không tìm thấy file data.yaml: {data_yaml}")
        return
    
    # Load model
    model = YOLO(str(model_path))
    
    # Validation
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name
    )
    
    print("=" * 60)
    print("VALIDATION HOÀN TẤT!")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("=" * 60)


def predict(
    model_path,
    source,
    imgsz=640,
    conf=0.25,
    save=True,
    project="runs/predict",
    name="exp"
):
    """
    Prediction
    
    Args:
        model_path: Đường dẫn đến file weights (.pt)
        source: Nguồn ảnh (file, folder, URL, ...)
        imgsz: Kích thước ảnh
        conf: Confidence threshold
        save: Có lưu kết quả không
        project: Thư mục project
        name: Tên experiment
    """
    print("=" * 60)
    print("BẮT ĐẦU PREDICTION")
    print("=" * 60)
    
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = current_dir / model_path
    
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence: {conf}")
    print("=" * 60)
    
    if not model_path.exists():
        print(f"Lỗi: Không tìm thấy file model: {model_path}")
        return
    
    # Load model
    model = YOLO(str(model_path))
    
    # Prediction
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        save=save,
        project=project,
        name=name
    )
    
    print("=" * 60)
    print("PREDICTION HOÀN TẤT!")
    print(f"Đã xử lý {len(results)} ảnh")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chạy TDDet YOLO")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train, val, predict")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Training mode")
    train_parser.add_argument("--model", default="cfg/models/v8/yolov8-mobilenetv4.yaml", help="Model config")
    train_parser.add_argument("--data", default="../../dataset/data.yaml", help="Data YAML")
    train_parser.add_argument("--epochs", type=int, default=150, help="Epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    train_parser.add_argument("--device", default=0, help="Device (0, 1, ... or 'cpu')")
    train_parser.add_argument("--optimizer", default="SGD", help="Optimizer")
    train_parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    train_parser.add_argument("--project", default="runs/train", help="Project directory")
    train_parser.add_argument("--name", default="exp", help="Experiment name")
    train_parser.add_argument("--wandb", action="store_true", help="Bật WandB logging")
    train_parser.add_argument("--wandb-project", default=None, help="WandB project")
    train_parser.add_argument("--wandb-name", default=None, help="WandB run name")
    train_parser.add_argument("--wandb-entity", default=None, help="WandB entity")
    train_parser.add_argument("--wandb-api-key", default=None, help="WandB API key")
    
    # Validation parser
    val_parser = subparsers.add_parser("val", help="Validation mode")
    val_parser.add_argument("--model", required=True, help="Model weights (.pt)")
    val_parser.add_argument("--data", default="../../dataset/data.yaml", help="Data YAML")
    val_parser.add_argument("--split", default="test", help="Split (test/val)")
    val_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    val_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    val_parser.add_argument("--project", default="runs/val", help="Project directory")
    val_parser.add_argument("--name", default="exp", help="Experiment name")
    
    # Prediction parser
    pred_parser = subparsers.add_parser("predict", help="Prediction mode")
    pred_parser.add_argument("--model", required=True, help="Model weights (.pt)")
    pred_parser.add_argument("--source", required=True, help="Source (image, folder, URL, ...)")
    pred_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    pred_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    pred_parser.add_argument("--save", action="store_true", default=True, help="Save results")
    pred_parser.add_argument("--project", default="runs/predict", help="Project directory")
    pred_parser.add_argument("--name", default="exp", help="Experiment name")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(
            model_config=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            optimizer=args.optimizer,
            patience=args.patience,
            project=args.project,
            name=args.name,
            wandb_enable=args.wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            wandb_entity=args.wandb_entity,
            wandb_api_key=args.wandb_api_key
        )
    elif args.mode == "val":
        validate_model(
            model_path=args.model,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name
        )
    elif args.mode == "predict":
        predict(
            model_path=args.model,
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            save=args.save,
            project=args.project,
            name=args.name
        )
    else:
        parser.print_help()

