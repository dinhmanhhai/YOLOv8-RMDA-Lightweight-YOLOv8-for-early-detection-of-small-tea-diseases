# Loss functions for Improved YOLOv8s

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    """
    Calculate IoU between two boxes.
    
    Args:
        box1: [N, 4] or [B, N, 4]
        box2: [M, 4] or [B, M, 4]
        xywh: If True, boxes are in (x_center, y_center, width, height) format
              If False, boxes are in (x1, y1, x2, y2) format
        eps: Small value to avoid division by zero
    
    Returns:
        IoU: [N, M] or [B, N, M]
    """
    if xywh:
        # Convert to (x1, y1, x2, y2)
        box1 = torch.cat([box1[..., :2] - box1[..., 2:] / 2,
                         box1[..., :2] + box1[..., 2:] / 2], dim=-1)
        box2 = torch.cat([box2[..., :2] - box2[..., 2:] / 2,
                         box2[..., :2] + box2[..., 2:] / 2], dim=-1)
    
    # Calculate intersection
    inter_x1 = torch.max(box1[..., 0:1], box2[..., 0:1])
    inter_y1 = torch.max(box1[..., 1:2], box2[..., 1:2])
    inter_x2 = torch.min(box1[..., 2:3], box2[..., 2:3])
    inter_y2 = torch.min(box1[..., 3:4], box2[..., 3:4])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    box1_area = (box1[..., 2:3] - box1[..., 0:1]) * (box1[..., 3:4] - box1[..., 1:2])
    box2_area = (box2[..., 2:3] - box2[..., 0:1]) * (box2[..., 3:4] - box2[..., 1:2])
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + eps)
    return iou


def get_inner_box(box, scale_factor=0.7):
    """
    Get inner box from center with smaller size.
    
    Args:
        box: [N, 4] boxes in (x_center, y_center, width, height) format
        scale_factor: Scale factor for inner box (default: 0.7)
    
    Returns:
        inner_box: [N, 4] inner boxes in (x_center, y_center, width, height) format
    """
    x_center, y_center, width, height = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    
    inner_width = width * scale_factor
    inner_height = height * scale_factor
    
    inner_box = torch.stack([
        x_center,
        y_center,
        inner_width,
        inner_height
    ], dim=-1)
    
    return inner_box


class InnerIoULoss(nn.Module):
    """
    Inner-IoU Loss: Focuses on the inner region of bounding boxes.
    Helps the model concentrate on the central area of objects.
    """
    
    def __init__(self, scale_factor=0.7, reduction='mean'):
        """
        Args:
            scale_factor: Scale factor for inner box (default: 0.7)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        """
        Calculate Inner-IoU loss.
        
        Args:
            pred_boxes: [N, 4] or [B, N, 4] predicted boxes in (x_center, y_center, width, height) format
            target_boxes: [M, 4] or [B, M, 4] target boxes in (x_center, y_center, width, height) format
        
        Returns:
            loss: Scalar loss value
        """
        # Get inner boxes
        pred_inner = get_inner_box(pred_boxes, self.scale_factor)
        target_inner = get_inner_box(target_boxes, self.scale_factor)
        
        # Calculate IoU between inner boxes
        inner_iou = bbox_iou(pred_inner, target_inner, xywh=True)
        
        # Calculate IoU between original boxes
        original_iou = bbox_iou(pred_boxes, target_boxes, xywh=True)
        
        # Inner-IoU loss: 1 - inner_iou
        # We can also combine with original IoU for better training
        loss = 1.0 - inner_iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedIoULoss(nn.Module):
    """
    Combined loss using both original IoU and Inner-IoU.
    """
    
    def __init__(self, scale_factor=0.7, inner_weight=0.5, reduction='mean'):
        """
        Args:
            scale_factor: Scale factor for inner box
            inner_weight: Weight for inner IoU loss (0-1)
            reduction: Reduction method
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.inner_weight = inner_weight
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes):
        """
        Calculate combined IoU loss.
        
        Args:
            pred_boxes: [N, 4] or [B, N, 4] predicted boxes
            target_boxes: [M, 4] or [B, M, 4] target boxes
        
        Returns:
            loss: Scalar loss value
        """
        # Original IoU loss
        original_iou = bbox_iou(pred_boxes, target_boxes, xywh=True)
        original_loss = 1.0 - original_iou
        
        # Inner IoU loss
        pred_inner = get_inner_box(pred_boxes, self.scale_factor)
        target_inner = get_inner_box(target_boxes, self.scale_factor)
        inner_iou = bbox_iou(pred_inner, target_inner, xywh=True)
        inner_loss = 1.0 - inner_iou
        
        # Combined loss
        loss = (1.0 - self.inner_weight) * original_loss + self.inner_weight * inner_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

