"""
Ví dụ lệnh chạy:

python visualize_predictions.py \\
  --weights improved_yolov8/runs/train/weights/best.pt \\
  --data improved_yolov8/data.yaml \\
  --cfg improved_yolov8/cfg/yolov8-improved.yaml \\
  --split val \\
  --num-images 8 \\
  --conf 0.1 \\
  --iou 0.45 \\
  --output-dir improved_yolov8/runs/vis
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from improved_yolov8.models.data.dataset import YOLODataset, load_data_yaml
from improved_yolov8.train import ImprovedYOLOv8s
from improved_yolov8.models.utils.metrics import non_max_suppression, xywh2xyxy


@torch.no_grad()
def visualize_samples(
    weights_path: str,
    data_yaml: str = "improved_yolov8/data.yaml",
    cfg_path: str = "improved_yolov8/cfg/yolov8-improved.yaml",
    split: str = "val",
    num_images: int = 8,
    conf_threshold: float = 0.1,
    iou_threshold: float = 0.45,
    output_dir: str = "improved_yolov8/runs/vis",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).parent
    data_cfg = load_data_yaml(base_dir / data_yaml)
    img_dir = Path(data_cfg[split])

    dataset = YOLODataset(img_dir=img_dir, imgsz=data_cfg.get("imgsz", 640), augment=False)

    # Tạo model giống train.py
    num_classes = data_cfg["nc"]
    model = ImprovedYOLOv8s(
        cfg_path=str(base_dir / cfg_path),
        num_classes=num_classes,
        in_channels=3,
        img_size=data_cfg.get("imgsz", 640),
    ).to(device)

    # Load trọng số
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(min(num_images, len(dataset))))
    for idx in indices:
        sample = dataset[idx]
        img = sample["img"].unsqueeze(0).to(device)  # [1,3,H,W]
        labels = sample["labels"]  # [N,5] (cls, x,y,w,h) normalized
        img_path = Path(sample["img_path"])

        # Forward
        outputs = model(img)  # list of 3 scales
        preds = non_max_suppression(
            outputs,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )[0]

        # Chuẩn bị ảnh gốc (BGR)
        orig = cv2.imread(str(img_path))
        if orig is None:
            continue
        h, w = orig.shape[:2]

        # Vẽ GT (màu xanh lá)
        if len(labels) > 0:
            gt_boxes_xywh = labels[:, 1:5]
            gt_boxes_xyxy = xywh2xyxy(gt_boxes_xywh)
            for box, cls_id in zip(gt_boxes_xyxy, labels[:, 0].long()):
                x1 = int(box[0].item() * w)
                y1 = int(box[1].item() * h)
                x2 = int(box[2].item() * w)
                y2 = int(box[3].item() * h)
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    orig,
                    f"GT {int(cls_id.item())}",
                    (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Vẽ prediction (màu đỏ)
        if preds["boxes"].shape[0] > 0:
            pred_boxes_xyxy = xywh2xyxy(preds["boxes"])
            for box, cls_id, score in zip(
                pred_boxes_xyxy, preds["classes"].long(), preds["scores"]
            ):
                x1 = int(box[0].item() * w)
                y1 = int(box[1].item() * h)
                x2 = int(box[2].item() * w)
                y2 = int(box[3].item() * h)
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    orig,
                    f"P {int(cls_id.item())}:{score:.2f}",
                    (x1, min(h - 2, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        out_path = output_dir / f"{img_path.stem}_vis.jpg"
        cv2.imwrite(str(out_path), orig)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs prediction for Improved YOLOv8s")
    parser.add_argument(
        "--weights",
        type=str,
        default="improved_yolov8/runs/train/weights/best.pt",
        help="Đường dẫn tới checkpoint (best.pt hoặc last.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="improved_yolov8/data.yaml",
        help="Đường dẫn tới data.yaml",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="improved_yolov8/cfg/yolov8-improved.yaml",
        help="Đường dẫn tới file cấu hình model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Tập dữ liệu để visualize",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Số ảnh muốn vẽ",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold cho NMS",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="improved_yolov8/runs/vis",
        help="Thư mục lưu ảnh kết quả",
    )

    args = parser.parse_args()
    base_dir = Path(__file__).parent
    weights_path = base_dir / args.weights

    visualize_samples(
        weights_path=str(weights_path),
        data_yaml=args.data,
        cfg_path=args.cfg,
        split=args.split,
        num_images=args.num_images,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
