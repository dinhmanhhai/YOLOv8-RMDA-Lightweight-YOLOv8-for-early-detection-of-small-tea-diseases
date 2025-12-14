# Dataset loader for Improved YOLOv8s

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
from PIL import Image
import torchvision.transforms as transforms


class YOLODataset(Dataset):
    """YOLO format dataset loader."""
    
    def __init__(self, img_dir, label_dir=None, imgsz=640, augment=True):
        """
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing labels (if None, tries img_dir parent/labels)
            imgsz: Image size for resizing
            augment: Whether to apply augmentation
        """
        self.img_dir = Path(img_dir)
        self.imgsz = imgsz
        self.augment = augment
        
        if label_dir is None:
            # Try common label directory locations
            possible_label_dirs = [
                self.img_dir.parent / "labels",
                self.img_dir.parent.parent / "labels",
            ]
            self.label_dir = None
            for label_path in possible_label_dirs:
                if label_path.exists():
                    self.label_dir = label_path
                    break
            if self.label_dir is None:
                # Default to same structure as images
                self.label_dir = self.img_dir.parent / "labels"
        else:
            self.label_dir = Path(label_dir)
        
        # Get all image files
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) + 
                               list(self.img_dir.glob("*.png")) +
                               list(self.img_dir.glob("*.jpeg")))
        
        # Basic augmentation
        if augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # Resize
        r = min(self.imgsz / h0, self.imgsz / w0)
        h, w = int(h0 * r), int(w0 * r)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        top = (self.imgsz - h) // 2
        bottom = self.imgsz - h - top
        left = (self.imgsz - w) // 2
        right = self.imgsz - w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert to tensor and normalize
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        
        # Load labels
        label_path = self.label_dir / (img_path.stem + ".txt")
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([cls_id, x_center, y_center, width, height])
        
        # Convert to tensor
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'img': img,
            'labels': labels,
            'img_path': str(img_path),
            'shape': (h0, w0),
            'pad': (top, left)
        }


def load_data_yaml(yaml_path):
    """Load dataset configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

