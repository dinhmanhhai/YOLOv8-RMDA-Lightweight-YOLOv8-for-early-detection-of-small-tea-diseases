from ultralytics import YOLO


def main():
    """
    Script train YOLOv8 mẫu.
    Mặc định dùng model COCO pretrain 'yolov8n.pt' và dataset demo 'coco8.yaml'.
    Sau này bạn chỉ cần đổi:
      - model = YOLO("path_to_your_model.yaml" hoặc "path_to_your_weights.pt")
      - data="path/to/your_data.yaml"
    """
    # 1. Load model YOLO với kiến trúc YOLOv8 gốc từ YAML (nc=6)
    model = YOLO("yolov8-rfcbam-backbone.yaml")

    # 2. Train trên dataset của bạn (liệt kê các tham số phổ biến có thể chỉnh)
    results = model.train(
        data="data.yaml",          # path tới data.yaml
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,                  # -1: CPU, 0: GPU0, "0,1" đa GPU
        workers=8,
        project="improved-yolov8s",
        name="exp",
        # Optimizer & LR
        optimizer="SGD",           # auto | SGD | Adam | AdamW | NAdam | RMSProp
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
        auto_augment="randaugment",
        close_mosaic=10,
        # Loss & box
        box=7.5,
        cls=0.5,
        dfl=1.5,
        iou=0.7,
        kobj=1.0,
        # Regularization / dropout
        dropout=0.3,
        # Early stop / patience
        patience=20,              # giảm nếu muốn dừng sớm hơn
        # Saving / checkpoint
        save=True,
        save_period=-1,            # >0 để lưu mỗi N epoch
        exist_ok=False,
        # Validation
        val=True,
        # Misc
        deterministic=True,
        verbose=True,
        plots=True,
        seed=0,
        half=False,
        amp=True,
        resume=False,
    )

    print("Training finished. Results saved to:", results.save_dir)


if __name__ == "__main__":
    main()


