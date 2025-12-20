#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script để kiểm tra loss computation và model outputs
"""

import torch
import sys
from pathlib import Path

# Add models to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from models.utils.detection_loss import DetectionLoss
from models.utils.tal import TaskAlignedAssigner


def debug_loss_computation():
    """Debug loss computation với dummy data"""
    print("=" * 60)
    print("DEBUG: Loss Computation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 6
    batch_size = 2
    
    # Tạo dummy predictions (3 scales: P3, P4, P5)
    # Format: [batch, num_classes + 1 + 4, H, W] = [batch, 11, H, W]
    predictions = [
        torch.randn(batch_size, 11, 80, 80, device=device),  # P3: H/8
        torch.randn(batch_size, 11, 40, 40, device=device),  # P4: H/16
        torch.randn(batch_size, 11, 20, 20, device=device),  # P5: H/32
    ]
    
    # Tạo dummy targets
    # Format: List of [N, 5] tensors where N varies per image
    targets = [
        torch.tensor([
            [0, 0.5, 0.5, 0.2, 0.2],  # class 0, center (0.5, 0.5), size 0.2x0.2
            [1, 0.3, 0.7, 0.15, 0.15], # class 1, center (0.3, 0.7), size 0.15x0.15
        ], device=device),
        torch.tensor([
            [2, 0.6, 0.4, 0.25, 0.25], # class 2, center (0.6, 0.4), size 0.25x0.25
        ], device=device),
    ]
    
    # Initialize loss
    criterion = DetectionLoss(num_classes=num_classes)
    criterion = criterion.to(device)
    
    # Forward pass
    print("\n1. Computing loss...")
    loss_dict = criterion(predictions, targets)
    
    print(f"\n2. Loss Results:")
    print(f"   Total Loss: {loss_dict['loss'].item():.6f}")
    print(f"   Cls Loss: {loss_dict['cls_loss'].item():.6f}")
    print(f"   Box Loss: {loss_dict['box_loss'].item():.6f}")
    print(f"   Requires Grad: {loss_dict['loss'].requires_grad}")
    
    # Check gradients
    print(f"\n3. Backward pass...")
    loss_dict['loss'].backward()
    
    # Check if model has gradients
    has_grad = False
    for name, param in criterion.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                has_grad = True
                print(f"   {name}: grad_norm = {grad_norm:.6f}")
    
    if not has_grad:
        print("   ⚠️  WARNING: No gradients found!")
    
    print("\n" + "=" * 60)
    print("DEBUG: TaskAlignedAssigner")
    print("=" * 60)
    
    # Test assigner directly
    assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)
    
    # Prepare inputs
    B, N, C = batch_size, 80*80 + 40*40 + 20*20, num_classes
    pred_scores = torch.randn(B, N, C, device=device)
    pred_bboxes = torch.rand(B, N, 4, device=device)  # Normalized xywh
    
    # Create anchor points
    anchor_points = torch.rand(N, 2, device=device)  # Normalized xy
    
    # Create GT
    max_gt = 2
    gt_labels = torch.tensor([[0, 1], [2, -1]], device=device, dtype=torch.long)
    gt_bboxes = torch.tensor([
        [[0.5, 0.5, 0.2, 0.2], [0.3, 0.7, 0.15, 0.15]],
        [[0.6, 0.4, 0.25, 0.25], [0.0, 0.0, 0.0, 0.0]],
    ], device=device)
    mask_gt = torch.tensor([[True, True], [True, False]], device=device)
    
    print("\n4. Testing TaskAlignedAssigner...")
    target_labels, target_bboxes, target_scores, fg_mask, max_gt_out = assigner(
        pred_scores.detach(), pred_bboxes.detach(),
        anchor_points, gt_labels, gt_bboxes, mask_gt
    )
    
    print(f"   fg_mask.sum(): {fg_mask.sum().item()}")
    print(f"   target_scores stats:")
    print(f"      min: {target_scores.min().item():.6f}")
    print(f"      max: {target_scores.max().item():.6f}")
    print(f"      mean: {target_scores.mean().item():.6f}")
    print(f"      non-zero count: {(target_scores > 0).sum().item()}")
    
    if fg_mask.sum() == 0:
        print("   ⚠️  WARNING: No foreground anchors found!")
    else:
        print(f"   ✓ Found {fg_mask.sum().item()} foreground anchors")
    
    print("\n" + "=" * 60)
    print("DEBUG Complete!")
    print("=" * 60)


if __name__ == '__main__':
    debug_loss_computation()

