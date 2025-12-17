# Detection loss for Improved YOLOv8s

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import CombinedIoULoss
from .tal import TaskAlignedAssigner, xywh2xyxy


class DetectionLoss(nn.Module):
    """Combined loss for object detection: Classification + Box Regression + Inner-IoU"""
    
    def __init__(self, num_classes=6, box_weight=7.5, cls_weight=0.5, inner_iou_weight=0.5, topk=10, alpha=0.5, beta=6.0):
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
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta)
    
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
        
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        # Build padded GT tensors
        max_gt = max(len(t) for t in targets) if targets else 0
        if max_gt == 0:
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'cls_loss': torch.tensor(0.0, device=device),
                'box_loss': torch.tensor(0.0, device=device),
                'iou_loss': torch.tensor(0.0, device=device),
                'inner_iou_loss': torch.tensor(0.0, device=device)
            }

        gt_labels = torch.full((batch_size, max_gt), -1, device=device, dtype=torch.long)
        gt_bboxes = torch.zeros((batch_size, max_gt, 4), device=device)
        mask_gt = torch.zeros((batch_size, max_gt), device=device, dtype=torch.bool)
        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue
            n = len(targets[b])
            gt_labels[b, :n] = targets[b][:, 0].long()
            gt_bboxes[b, :n] = targets[b][:, 1:5]
            mask_gt[b, :n] = True

        # Collect predictions across scales
        cls_preds = []
        box_preds = []
        anchor_points = []
        strides = []

        for pred in predictions:
            B, _, H, W = pred.shape
            cls_pred = pred[:, :self.num_classes, :, :]  # logits
            box_pred = pred[:, self.num_classes+1:, :, :]  # 4

            # sigmoid + clamp to align with NMS decoding (normalized 0-1)
            box_pred = torch.sigmoid(box_pred).clamp(1e-4, 1 - 1e-4)

            # flatten
            cls_preds.append(cls_pred.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            box_preds.append(box_pred.permute(0, 2, 3, 1).reshape(B, -1, 4))

            # anchor centers normalized
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            anc_x = (grid_x + 0.5) / W
            anc_y = (grid_y + 0.5) / H
            anchor_points.append(torch.stack([anc_x, anc_y], dim=-1).reshape(-1, 2))
            strides.append(torch.tensor([1.0 / W, 1.0 / H], device=device).unsqueeze(0).repeat(H * W, 1))

        cls_pred_all = torch.cat(cls_preds, dim=1)  # (B, N, C)
        box_pred_all = torch.cat(box_preds, dim=1)  # (B, N, 4) xywh normalized
        anchor_points_all = torch.cat(anchor_points, dim=0)  # (N, 2)
        stride_tensor = torch.cat(strides, dim=0)  # (N, 2) not used heavily but kept for symmetry

        # Assign
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            cls_pred_all.detach(), box_pred_all.detach(),
            anchor_points_all, gt_labels, gt_bboxes, mask_gt
        )

        num_fg = fg_mask.sum().clamp(min=1)

        # Classification loss (BCE with alignment scores)
        cls_loss = self.bce_loss(cls_pred_all, target_scores)
        cls_loss = (cls_loss.sum() / num_fg)

        # Box loss on positives
        if fg_mask.any():
            pred_pos = box_pred_all[fg_mask]
            target_pos = target_bboxes[fg_mask]
            box_loss = self.inner_iou_loss(pred_pos, target_pos)
        else:
            box_loss = torch.tensor(0.0, device=device)

        total_loss = self.cls_weight * cls_loss + self.box_weight * box_loss

        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'iou_loss': box_loss,
            'inner_iou_loss': box_loss
        }

