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
import wandb
import numpy as np

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
    save_dir='runs/train',
    project='improved-yolov8s',
    name=None,
    resume=None,
    early_stop_patience=20,
    early_stop_min_delta=1e-4
):
    """
    Training function for Improved YOLOv8s
    
    Args:
        data_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Image size
        device: Device to train on
        workers: Number of data loader workers
        lr0: Initial learning rate
        weight_decay: Weight decay
        save_dir: Directory to save checkpoints
        project: WandB project name
        name: WandB run name (None for auto-generated)
        resume: WandB run ID to resume (None for new run)
        early_stop_patience: Stop if no improvement after this many epochs
        early_stop_min_delta: Minimum improvement to reset patience
    """
    print("=" * 60)
    print("Training Improved YOLOv8s")
    print("=" * 60)
    
    # Load dataset config
    data_config = load_data_yaml(data_yaml)
    num_classes = data_config['nc']
    
    # Initialize WandB
    wandb.init(
        project=project,
        name=name,
        resume=resume,
        config={
            'model': 'Improved YOLOv8s',
            'num_classes': num_classes,
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'lr0': lr0,
            'weight_decay': weight_decay,
            'device': device,
            'data_yaml': data_yaml
        }
    )
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params} parameters ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params} ({trainable_params/1e6:.2f}M)")
    print(f"Training on device: {device}")
    print(f"Dataset: {data_yaml}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
    print("=" * 60)
    
    # Log model info to WandB
    wandb.config.update({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    })
    
    # Calculate FPS (inference speed) - measure once at the beginning
    print("Measuring inference speed (FPS)...")
    model.eval()
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure FPS
    if device == 'cuda':
        torch.cuda.synchronize()
    
    num_runs = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_runs / total_time
    
    print(f"Inference speed: {fps:.2f} FPS")
    print("=" * 60)
    
    # Log initial FPS to WandB
    wandb.log({'metrics/FPS': fps})
    
    model.train()
    
    # Training loop
    best_loss = float('inf')
    best_map = 0.0
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_box_loss = 0.0
        train_dfl_loss = 0.0  # DFL loss (using inner_iou_loss as approximation)
        
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
            # Use inner_iou_loss as DFL loss approximation
            train_dfl_loss += loss_dict.get('inner_iou_loss', loss_dict['box_loss']).item()
            
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
        train_dfl_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_box_loss = 0.0
        val_dfl_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validating')):
                images = batch['img'].to(device)
                labels = batch['labels']
                img_paths = batch.get('img_path', [])
                
                # Extract image names from paths
                img_names = []
                if img_paths:
                    for path in img_paths:
                        if path:
                            img_names.append(Path(path).name)
                        else:
                            img_names.append("unknown")
                else:
                    img_names = [f"img_{i}" for i in range(len(labels))]
                
                gt_counts = [len(l) for l in labels]
                print(f"[Debug] Val batch {batch_idx} GT counts per image: {gt_counts}")
                print(f"[Debug] Val batch {batch_idx} Image names: {img_names}")
                
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
                val_cls_loss += loss_dict['cls_loss'].item()
                val_box_loss += loss_dict['box_loss'].item()
                val_dfl_loss += loss_dict.get('inner_iou_loss', loss_dict['box_loss']).item()
                
                # Collect predictions and targets cho metrics (mỗi epoch)
                from models.utils.metrics import non_max_suppression
                predictions = non_max_suppression(
                    outputs,
                    conf_threshold=0.1,   # align với metrics.py
                    iou_threshold=0.45
                )
                val_predictions.extend(predictions)
                val_targets.extend(targets)
        
        val_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_box_loss /= len(val_loader)
        val_dfl_loss /= len(val_loader)
        
        # Calculate metrics
        val_map_50 = 0.0
        val_map_50_95 = 0.0
        val_precision = 0.0
        val_recall = 0.0
        from models.utils.metrics import calculate_map, calculate_precision_recall
        # Calculate mAP@0.5 (mỗi epoch)
        map_results_50 = calculate_map(
            val_predictions,
            val_targets,
            num_classes=num_classes,
            iou_threshold=0.5
        )
        val_map_50 = map_results_50['map']
        # Precision/Recall tính lại với cùng ngưỡng conf/iou như NMS (mỗi epoch)
        pr_results = calculate_precision_recall(
            val_predictions,
            val_targets,
            num_classes=num_classes,
            iou_threshold=0.5,
            conf_threshold=0.1
        )
        val_precision = pr_results['precision']
        val_recall = pr_results['recall']
        
        # Calculate mAP@0.5:0.95 (average over multiple IoU thresholds) mỗi 5 epoch hoặc epoch cuối
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95
            maps_50_95 = []
            for iou_thresh in iou_thresholds:
                map_result = calculate_map(
                    val_predictions,
                    val_targets,
                    num_classes=num_classes,
                    iou_threshold=iou_thresh
                )
                maps_50_95.append(map_result['map'])
            val_map_50_95 = np.mean(maps_50_95) if len(maps_50_95) > 0 else 0.0
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Early stopping tracking
        prev_best_map = best_map
        prev_best_loss = best_loss
        improved_map = False
        improved_loss = False

        if val_map_50 > best_map + early_stop_min_delta:
            best_map = val_map_50
            improved_map = True

        if val_loss < best_loss - early_stop_min_delta:
            best_loss = val_loss
            improved_loss = True

        if improved_map or improved_loss:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Log metrics to WandB
        log_dict = {
            'epoch': epoch + 1,
            'train/box_loss': train_box_loss,
            'train/cls_loss': train_cls_loss,
            'train/dfl_loss': train_dfl_loss,
            'val/box_loss': val_box_loss,
            'val/cls_loss': val_cls_loss,
            'val/dfl_loss': val_dfl_loss,
            'lr': current_lr,
            'early_stop/epochs_no_improve': epochs_no_improve
        }
        
        # Add metrics (precision/mAP@0.5 mỗi epoch, mAP@0.5:0.95 + FPS mỗi 5 epoch)
        log_dict.update({
            'metrics/precision(B)': val_precision,
            'metrics/recall(B)': val_recall,
            'metrics/mAP50(B)': val_map_50,
        })
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            log_dict.update({
                'metrics/mAP50-95(B)': val_map_50_95,
                'metrics/FPS': fps  # Log FPS every time metrics are calculated
            })
        
        wandb.log(log_dict)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, box: {train_box_loss:.4f}, dfl: {train_dfl_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (cls: {val_cls_loss:.4f}, box: {val_box_loss:.4f}, dfl: {val_dfl_loss:.4f})")
        print(f"  Val mAP@0.5: {val_map_50:.4f} ({val_map_50*100:.2f}%)")
        print(f"  Val Precision: {val_precision:.4f} ({val_precision*100:.2f}%)")
        print(f"  Val Recall: {val_recall:.4f} ({val_recall*100:.2f}%)")
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Val mAP@0.5:0.95: {val_map_50_95:.4f} ({val_map_50_95*100:.2f}%)")
            print(f"  FPS: {fps:.2f}")
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
        
        # Save best checkpoint (based on mAP if available, otherwise loss)
        if improved_map:
            checkpoint['map'] = best_map
            checkpoint['map_50_95'] = val_map_50_95
            torch.save(checkpoint, weights_dir / 'best.pt')
            print(f"  ✓ Saved best model (mAP@0.5: {best_map:.4f})")
        elif improved_loss:
            torch.save(checkpoint, weights_dir / 'best.pt')
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
        
        # Early stopping check
        if epochs_no_improve >= early_stop_patience:
            print(f"  Early stopping triggered at epoch {epoch+1} (no improve for {epochs_no_improve} epochs)")
            wandb.log({'early_stop/trigger_epoch': epoch + 1})
            break
        
        print("-" * 60)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    if best_map > 0:
        print(f"Best mAP@0.5: {best_map:.4f} ({best_map*100:.2f}%)")
    print(f"Model saved to: {weights_dir}")
    
    # Finish WandB run
    wandb.finish()


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
    parser.add_argument('--config', type=str, default=None, help='Path to training config YAML')
    parser.add_argument('--data', type=str, default=None, help='Dataset YAML file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--lr0', type=float, default=None, help='Initial learning rate')
    parser.add_argument('--project', type=str, default=None, help='WandB project name')
    parser.add_argument('--name', type=str, default=None, help='WandB run name')
    parser.add_argument('--resume', type=str, default=None, help='WandB run ID to resume')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--early-stop-patience', type=int, default=None, help='Early stopping patience (epochs)')
    parser.add_argument('--early-stop-min-delta', type=float, default=None, help='Minimum improvement to reset patience')
    
    args = parser.parse_args()
    
    # Base defaults
    config = {
        'data': 'data.yaml',
        'epochs': 150,
        'batch': 16,
        'imgsz': 640,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 4,
        'lr0': 0.01,
        'project': 'improved-yolov8s',
        'name': None,
        'resume': None,
        'save_dir': 'runs/train',
        'early_stop_patience': 20,
        'early_stop_min_delta': 1e-4
    }
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_cfg = yaml.safe_load(f) or {}
        config.update({k: v for k, v in file_cfg.items() if v is not None})
    
    # Override with CLI args if provided
    override_map = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'lr0': args.lr0,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'save_dir': args.save_dir,
        'early_stop_patience': args.early_stop_patience,
        'early_stop_min_delta': args.early_stop_min_delta
    }
    for k, v in override_map.items():
        if v is not None:
            config[k] = v
    
    train_model(
        data_yaml=config['data'],
        epochs=config['epochs'],
        batch_size=config['batch'],
        imgsz=config['imgsz'],
        device=config['device'],
        workers=config['workers'],
        lr0=config['lr0'],
        project=config['project'],
        name=config['name'],
        resume=config['resume'],
        save_dir=config['save_dir'],
        early_stop_patience=config['early_stop_patience'],
        early_stop_min_delta=config['early_stop_min_delta']
    )

