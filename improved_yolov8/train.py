#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Improved YOLOv8s Architecture
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add models to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import model components
from models.modules.conv import Conv
from models.modules.block import C2f, MixSPPF
from models.extra_modules.RFAConv import RFCBAMConv
from models.extra_modules.block import C2f_RFCBAMConv, AKConv, RepGFPN
from models.extra_modules.head import DynamicHead
from models.utils.loss import InnerIoULoss, CombinedIoULoss


class ImprovedYOLOv8s(nn.Module):
    """
    Improved YOLOv8s Model Architecture
    """
    def __init__(self, num_classes=6, scale='s'):
        super().__init__()
        self.num_classes = num_classes
        
        # Scale parameters
        scales = {
            'n': [0.33, 0.25, 1024],
            's': [0.33, 0.50, 1024],
            'm': [0.67, 0.75, 768],
            'l': [1.00, 1.00, 512],
            'x': [1.00, 1.25, 512]
        }
        depth, width, max_channels = scales[scale]
        
        # Backbone
        self.backbone = nn.Sequential(
            RFCBAMConv(3, int(64 * width), 3, 2),      # P1/2
            RFCBAMConv(int(64 * width), int(128 * width), 3, 2),  # P2/4
            C2f_RFCBAMConv(int(128 * width), int(128 * width), 1),
            RFCBAMConv(int(128 * width), int(256 * width), 3, 2),  # P3/8
            C2f_RFCBAMConv(int(256 * width), int(256 * width), 2),
            RFCBAMConv(int(256 * width), int(512 * width), 3, 2),  # P4/16
            C2f_RFCBAMConv(int(512 * width), int(512 * width), 2),
            MixSPPF(int(512 * width), int(512 * width), 5)  # P5/32
        )
        
        # Neck
        # P3 path
        self.neck_p3 = nn.Sequential(
            C2f(int(256 * width), int(256 * width), 1),
            RepGFPN(int(256 * width), int(256 * width), 1),
            AKConv(int(256 * width), int(256 * width), 5)
        )
        
        # P4 path
        self.neck_p4_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck_p4 = nn.Sequential(
            C2f(int(512 * width) + int(256 * width), int(512 * width), 1),
            RepGFPN(int(512 * width), int(512 * width), 1),
            AKConv(int(512 * width), int(512 * width), 5)
        )
        
        # P5 path
        self.neck_p5_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.neck_p5 = nn.Sequential(
            C2f(int(512 * width) + int(512 * width), int(1024 * width), 1),
            RepGFPN(int(1024 * width), int(1024 * width), 1)
        )
        
        # Bottom-up paths
        self.neck_p4_down = Conv(int(1024 * width), int(512 * width), 3, 2)
        self.neck_p4_bottom = C2f(int(512 * width) * 2, int(512 * width), 1)
        
        self.neck_p3_down = Conv(int(512 * width), int(256 * width), 3, 2)
        self.neck_p3_bottom = C2f(int(256 * width) * 2, int(256 * width), 1)
        
        # Head
        self.head = DynamicHead(int(256 * width), num_classes, num_tasks=3)
        
    def forward(self, x):
        # Backbone
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 7]:  # P3, P4, P5
                features.append(x)
        
        p3, p4, p5 = features
        
        # Neck - Top-down
        p3_out = self.neck_p3(p3)
        
        p4_up = self.neck_p4_up(p3_out)
        p4_cat = torch.cat([p4_up, p4], dim=1)
        p4_out = self.neck_p4(p4_cat)
        
        p5_up = self.neck_p5_up(p4_out)
        p5_cat = torch.cat([p5_up, p5], dim=1)
        p5_out = self.neck_p5(p5_cat)
        
        # Neck - Bottom-up
        p4_down = self.neck_p4_down(p5_out)
        p4_cat_bottom = torch.cat([p4_down, p4_out], dim=1)
        p4_final = self.neck_p4_bottom(p4_cat_bottom)
        
        p3_down = self.neck_p3_down(p4_final)
        p3_cat_bottom = torch.cat([p3_down, p3_out], dim=1)
        p3_final = self.neck_p3_bottom(p3_cat_bottom)
        
        # Head
        outputs = self.head([p3_final, p4_final, p5_out])
        
        return outputs


def train_model(
    data_yaml='data.yaml',
    epochs=150,
    batch_size=16,
    imgsz=640,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    workers=4,
    lr0=0.01,
    weight_decay=0.0005,
    save_dir='runs/train'
):
    """
    Training function for Improved YOLOv8s
    """
    print("=" * 60)
    print("Training Improved YOLOv8s")
    print("=" * 60)
    
    # Create model
    model = ImprovedYOLOv8s(num_classes=6, scale='s')
    model = model.to(device)
    
    # Loss function
    criterion = CombinedIoULoss(scale_factor=0.7, inner_weight=0.5)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr0,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr0 * 0.01
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
    print("=" * 60)
    
    # Note: This is a simplified training script
    # For full training, you would need to:
    # 1. Load dataset using YOLO dataset format
    # 2. Implement data augmentation
    # 3. Implement training loop with validation
    # 4. Save checkpoints
    # 5. Log metrics
    
    print("Training script structure created.")
    print("Please integrate with your dataset loader and training pipeline.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Improved YOLOv8s')
    parser.add_argument('--data', type=str, default='data.yaml', help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        lr0=args.lr0
    )

