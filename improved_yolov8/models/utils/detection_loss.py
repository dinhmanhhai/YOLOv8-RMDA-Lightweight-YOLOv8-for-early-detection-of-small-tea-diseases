# Detection loss for Improved YOLOv8s

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import CombinedIoULoss, bbox_iou


class DetectionLoss(nn.Module):
    """Combined loss for object detection: Classification + Box Regression + Inner-IoU"""
    
    def __init__(self, num_classes=6, box_weight=7.5, cls_weight=0.5, inner_iou_weight=0.5):
        """
        Args:
            num_classes: Number of classes
            box_weight: Weight for box regression loss
            cls_weight: Weight for classification loss
            inner_iou_weight: Weight for Inner-IoU loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.inner_iou_loss = CombinedIoULoss(scale_factor=0.7, inner_weight=inner_iou_weight)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of predictions from model [P3, P4, P5]
                Each prediction shape: (batch, num_classes+1+4, H, W)
            targets: List of target tensors, each with shape (N, 5) where N varies
                Format: [class_id, x_center, y_center, width, height] (normalized)
        
        Returns:
            dict with loss components
        """
        total_loss = 0.0
        cls_loss_total = 0.0
        box_loss_total = 0.0
        
        # Process each scale
        for scale_idx, pred in enumerate(predictions):
            batch_size = pred.shape[0]
            H, W = pred.shape[2], pred.shape[3]
            
            # pred shape: (batch, num_classes+1+4, H, W)
            # Split into cls, center, box
            cls_pred = pred[:, :self.num_classes, :, :]  # (batch, num_classes, H, W)
            center_pred = pred[:, self.num_classes:self.num_classes+1, :, :]  # (batch, 1, H, W)
            box_pred = pred[:, self.num_classes+1:, :, :]  # (batch, 4, H, W)
            
            # Initialize target tensors
            cls_target = torch.zeros_like(cls_pred)
            box_target = torch.zeros_like(box_pred)
            pos_mask = torch.zeros((batch_size, H, W), device=pred.device, dtype=torch.bool)
            
            # Match targets to grid cells (simplified version)
            # In full implementation, would use anchor matching or TAL (Task Aligned Learning)
            for b in range(batch_size):
                if len(targets[b]) > 0:
                    # For each target in this batch
                    for target in targets[b]:
                        cls_id = int(target[0])
                        x_center, y_center = target[1].item(), target[2].item()
                        width, height = target[3].item(), target[4].item()
                        
                        # Convert normalized coordinates to grid coordinates
                        grid_x = int(x_center * W)
                        grid_y = int(y_center * H)
                        grid_x = max(0, min(W - 1, grid_x))
                        grid_y = max(0, min(H - 1, grid_y))
                        
                        # Set target
                        cls_target[b, cls_id, grid_y, grid_x] = 1.0
                        box_target[b, 0, grid_y, grid_x] = x_center
                        box_target[b, 1, grid_y, grid_x] = y_center
                        box_target[b, 2, grid_y, grid_x] = width
                        box_target[b, 3, grid_y, grid_x] = height
                        pos_mask[b, grid_y, grid_x] = True
            
            # Classification loss (only on positive positions)
            cls_loss = self.bce_loss(cls_pred, cls_target)
            pos_mask_expanded = pos_mask.unsqueeze(1).expand_as(cls_loss)
            cls_loss = (cls_loss * pos_mask_expanded).sum() / (pos_mask.sum() + 1e-8)
            
            # Box loss (only on positive positions)
            # Convert to xyxy format for IoU calculation
            def xywh2xyxy(box_xywh):
                """Convert (x_center, y_center, width, height) to (x1, y1, x2, y2)"""
                x_center, y_center, width, height = box_xywh[..., 0], box_xywh[..., 1], box_xywh[..., 2], box_xywh[..., 3]
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                return torch.stack([x1, y1, x2, y2], dim=-1)
            
            # Extract positive predictions and targets
            pos_indices = pos_mask.nonzero(as_tuple=False)  # [N, 2] (batch_idx, grid_y, grid_x)
            
            if len(pos_indices) > 0:
                # Get positive predictions
                pos_box_pred = []
                pos_box_target = []
                
                for idx in pos_indices:
                    b, gy, gx = idx[0].item(), idx[1].item(), idx[2].item()
                    pos_box_pred.append(box_pred[b, :, gy, gx])  # [4]
                    pos_box_target.append(box_target[b, :, gy, gx])  # [4]
                
                if len(pos_box_pred) > 0:
                    pos_box_pred = torch.stack(pos_box_pred)  # [N, 4]
                    pos_box_target = torch.stack(pos_box_target)  # [N, 4]
                    
                    # IoU-based loss
                    pred_xyxy = xywh2xyxy(pos_box_pred)
                    target_xyxy = xywh2xyxy(pos_box_target)
                    
                    # Calculate IoU (element-wise)
                    iou_values = []
                    for i in range(len(pos_box_pred)):
                        iou = bbox_iou(pred_xyxy[i:i+1], target_xyxy[i:i+1], xywh=False)
                        iou_values.append(iou.squeeze())
                    iou = torch.stack(iou_values)  # [N]
                    iou_loss = (1.0 - iou).mean()
                    
                    # Inner-IoU loss (theo paper)
                    inner_iou_loss = self.inner_iou_loss(pos_box_pred, pos_box_target)
                    
                    # Combined box loss: 50% IoU + 50% Inner-IoU
                    box_loss = 0.5 * iou_loss + 0.5 * inner_iou_loss
                else:
                    box_loss = torch.tensor(0.0, device=box_pred.device)
            else:
                box_loss = torch.tensor(0.0, device=box_pred.device)
            
            cls_loss_total += cls_loss
            box_loss_total += box_loss
        
        # Average across scales
        num_scales = len(predictions)
        cls_loss_total = cls_loss_total / num_scales
        box_loss_total = box_loss_total / num_scales
        
        total_loss = self.cls_weight * cls_loss_total + self.box_weight * box_loss_total
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss_total,
            'box_loss': box_loss_total,
            'iou_loss': box_loss_total,  # For logging
            'inner_iou_loss': box_loss_total  # For logging (combined in box_loss)
        }

