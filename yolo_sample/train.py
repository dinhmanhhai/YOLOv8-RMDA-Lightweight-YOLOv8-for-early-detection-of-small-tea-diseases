from ultralytics import YOLO


def main():
    """
    Script train YOLOv8 mẫu.
    Mặc định dùng model COCO pretrain 'yolov8n.pt' và dataset demo 'coco8.yaml'.
    Sau này bạn chỉ cần đổi:
      - model = YOLO("path_to_your_model.yaml" hoặc "path_to_your_weights.pt")
      - data="path/to/your_data.yaml"
    """
    # 1. Load model YOLOv8 COCO-pretrained
    model = YOLO("yolov8n.pt")

    # 2. Train trên dataset demo coco8 (Ultralytics cung cấp sẵn)
    results = model.train(
        data="../dataset/data.yaml",   # sử dụng dataset của bạn
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,            # -1: CPU, 0: GPU 0
        project="runs/train",
        name="yolo_sample_exp",
    )

    print("Training finished. Results saved to:", results.save_dir)


if __name__ == "__main__":
    main()


