#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để vẽ bounding box lên ảnh từ file label YOLO format
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


# Màu sắc cho các class (BGR format cho OpenCV)
COLORS = [
    (0, 255, 0),      # Xanh lá - Class 0
    (255, 0, 0),      # Xanh dương - Class 1
    (0, 0, 255),      # Đỏ - Class 2
    (255, 255, 0),    # Cyan - Class 3
    (255, 0, 255),    # Magenta - Class 4
    (0, 255, 255),    # Vàng - Class 5
]

# Tên các class (từ data.yaml)
CLASS_NAMES = [
    'Tea algae leaf spot',
    'Tea cake',
    'Tea cloud leaf blight',
    'Tea exobasidium blight',
    'Tea red rust',
    'Tea red scab'
]


def draw_bbox(image_path, label_path, output_path, class_names=None):
    """
    Vẽ bounding box lên ảnh từ file label YOLO format
    
    Args:
        image_path: Đường dẫn đến file ảnh
        label_path: Đường dẫn đến file label (.txt)
        output_path: Đường dẫn để lưu ảnh đã vẽ
        class_names: Danh sách tên các class (optional)
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Đọc file label
    if not os.path.exists(label_path):
        print(f"Không tìm thấy file label: {label_path}")
        return False
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Vẽ từng bounding box
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Chuyển từ normalized coordinates sang pixel coordinates
        x_center_px = x_center * w
        y_center_px = y_center * h
        width_px = width * w
        height_px = height * h
        
        # Tính tọa độ góc trên trái và góc dưới phải
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # Đảm bảo tọa độ nằm trong ảnh
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Chọn màu cho class
        color = COLORS[class_id % len(COLORS)]
        
        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ label text
        if class_names and class_id < len(class_names):
            label_text = f"{class_names[class_id]} ({class_id})"
        else:
            label_text = f"Class {class_id}"
        
        # Tính kích thước text
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Vẽ background cho text
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Vẽ text
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Lưu ảnh
    cv2.imwrite(output_path, img)
    print(f"Đã lưu ảnh: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Vẽ bounding box lên ảnh từ file label YOLO format'
    )
    parser.add_argument(
        'image_name',
        type=str,
        help='Tên file ảnh (ví dụ: 002_7221_JPG_jpg.rf.b54d5b4805056b1d2dd0527c169d7b93.jpg)'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default='dataset/original dataset/images',
        help='Thư mục chứa ảnh (mặc định: dataset/original dataset/images)'
    )
    parser.add_argument(
        '--labels_dir',
        type=str,
        default='dataset/original dataset/labels',
        help='Thư mục chứa labels (mặc định: dataset/original dataset/labels)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset/original dataset/images_with_bbox',
        help='Thư mục lưu ảnh đã vẽ (mặc định: dataset/original dataset/images_with_bbox)'
    )
    
    args = parser.parse_args()
    
    # Lấy đường dẫn đầy đủ
    base_dir = Path(__file__).parent.absolute()
    images_dir = base_dir / args.images_dir
    labels_dir = base_dir / args.labels_dir
    output_dir = base_dir / args.output_dir
    
    # Tìm file ảnh
    image_path = images_dir / args.image_name
    if not image_path.exists():
        print(f"Không tìm thấy file ảnh: {image_path}")
        return
    
    # Tìm file label tương ứng (đổi đuôi .jpg thành .txt)
    label_name = Path(args.image_name).stem + '.txt'
    label_path = labels_dir / label_name
    
    if not label_path.exists():
        print(f"Không tìm thấy file label: {label_path}")
        return
    
    # Tạo đường dẫn output
    output_path = output_dir / args.image_name
    
    # Vẽ bounding box
    draw_bbox(
        str(image_path),
        str(label_path),
        str(output_path),
        class_names=CLASS_NAMES
    )


if __name__ == '__main__':
    main()

