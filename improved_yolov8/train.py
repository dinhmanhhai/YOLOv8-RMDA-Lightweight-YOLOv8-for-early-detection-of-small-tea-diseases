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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import time

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
from models.data.dataset import YOLODataset, load_data_yaml
from models.utils.detection_loss import DetectionLoss


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
        # P3 path: process P3 (H/8 → H/8, maintains spatial size)
        p3_out = self.neck_p3(p3)  # Output: H/8
        
        # P4 path: 
        # Upsample p3_out 2x to match p4's spatial size
        # p3_out is H/8, upsample 2x → H/4
        # p4 is H/16, upsample 2x → H/8
        # Actually, we need to match: upsample p4 to H/4 to concat with p3_out upsampled
        p4_up_from_p3 = self.neck_p4_up(p3_out)  # H/8 → H/4
        p4_up_from_backbone = F.interpolate(p4, size=p4_up_from_p3.shape[2:], mode='nearest')  # H/16 → H/4
        p4_cat = torch.cat([p4_up_from_p3, p4_up_from_backbone], dim=1)
        p4_out = self.neck_p4(p4_cat)  # Output: H/4 (C2f maintains spatial size)
        # Downsample p4_out back to H/16 to maintain original P4 scale
        p4_out = F.interpolate(p4_out, size=p4.shape[2:], mode='nearest')  # H/4 → H/16
        
        # P5 path:
        # Upsample p4_out 2x to match p5's spatial size  
        # p4_out is H/16, upsample 2x → H/8
        # p5 is H/32, upsample 4x → H/8
        p5_up_from_p4 = self.neck_p5_up(p4_out)  # H/16 → H/8
        p5_up_from_backbone = F.interpolate(p5, size=p5_up_from_p4.shape[2:], mode='nearest')  # H/32 → H/8
        p5_cat = torch.cat([p5_up_from_p4, p5_up_from_backbone], dim=1)
        p5_out = self.neck_p5(p5_cat)  # Output: H/8 (C2f maintains spatial size)
        # Downsample p5_out back to H/32 to maintain original P5 scale for bottom-up
        # This ensures p5_out matches p5's original size (H/32)
        p5_out = F.interpolate(p5_out, size=p5.shape[2:], mode='nearest')  # H/8 → H/32
        
        # Neck - Bottom-up
        # Ensure p5_out is H/32 before bottom-up
        if p5_out.shape[2:] != p5.shape[2:]:
            p5_out = F.interpolate(p5_out, size=p5.shape[2:], mode='nearest')
        
        # p5_out is H/32, downsample 2x → H/16
        # p4_out is H/16
        p4_down = self.neck_p4_down(p5_out)  # H/32 → H/16 (stride=2 downsamples 2x)
        # Match p4_out to p4_down size (should both be H/16)
        if p4_out.shape[2:] != p4_down.shape[2:]:
            p4_out = F.interpolate(p4_out, size=p4_down.shape[2:], mode='nearest')
        p4_cat_bottom = torch.cat([p4_down, p4_out], dim=1)
        p4_final = self.neck_p4_bottom(p4_cat_bottom)  # Output: H/16
        
        # p4_final is H/16, downsample 2x → H/8
        # p3_out is H/8
        p3_down = self.neck_p3_down(p4_final)  # H/16 → H/8 (stride=2 downsamples 2x)
        # Match p3_out to p3_down size (should both be H/8)
        if p3_out.shape[2:] != p3_down.shape[2:]:
            p3_out = F.interpolate(p3_out, size=p3_down.shape[2:], mode='nearest')
        p3_cat_bottom = torch.cat([p3_down, p3_out], dim=1)
        p3_final = self.neck_p3_bottom(p3_cat_bottom)  # Output: H/8
        
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
    
    # Load dataset config
    data_config = load_data_yaml(data_yaml)
    num_classes = data_config['nc']
    
    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    # Create model
    model = ImprovedYOLOv8s(num_classes=num_classes, scale='s')
    model = model.to(device)
    
    # Loss function
    criterion = DetectionLoss(num_classes=num_classes)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr0,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr0 * 0.01
    )
    
    # Load datasets
    train_dataset = YOLODataset(
        img_dir=data_config['train'],
        imgsz=imgsz,
        augment=True
    )
    
    val_dataset = YOLODataset(
        img_dir=data_config['val'],
        imgsz=imgsz,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")
    print(f"Dataset: {data_yaml}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
    print("=" * 60)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_box_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['img'].to(device)
            labels = batch['labels']  # List of label tensors
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Prepare targets for loss computation
            # Note: This is a simplified version - full implementation would need
            # proper anchor matching and target assignment like YOLO
            batch_size = images.shape[0]
            targets = []
            for i in range(batch_size):
                if len(labels[i]) > 0:
                    targets.append(labels[i].to(device))
                else:
                    # Empty target
                    targets.append(torch.zeros((0, 5), device=device))
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_cls_loss += loss_dict['cls_loss'].item()
            train_box_loss += loss_dict['box_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_dict["cls_loss"].item():.4f}',
                'box': f'{loss_dict["box_loss"].item():.4f}'
            })
        
        # Average losses
        train_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_box_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                images = batch['img'].to(device)
                labels = batch['labels']
                
                outputs = model(images)
                
                # Prepare targets
                batch_size = images.shape[0]
                targets = []
                for i in range(batch_size):
                    if len(labels[i]) > 0:
                        targets.append(labels[i].to(device))
                    else:
                        targets.append(torch.zeros((0, 5), device=device))
                
                loss_dict = criterion(outputs, targets)
                
                val_loss += loss_dict['loss'].item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, box: {train_box_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
        }
        
        # Save last checkpoint
        torch.save(checkpoint, weights_dir / 'last.pt')
        
        # Save best checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, weights_dir / 'best.pt')
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
        
        print("-" * 60)
    
    print(f"\nTraining completed! Best validation loss: {best_loss:.4f}")
    print(f"Model saved to: {weights_dir}")


def collate_fn(batch):
    """Custom collate function for batching."""
    images = torch.stack([item['img'] for item in batch])
    labels = [item['labels'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    shapes = [item['shape'] for item in batch]
    pads = [item['pad'] for item in batch]
    
    return {
        'img': images,
        'labels': labels,
        'img_path': img_paths,
        'shape': shapes,
        'pad': pads
    }


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

