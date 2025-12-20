"""
Inner-IoU loss implementation for improved YOLOv8.
Based on the Inner-IoU paper for better bounding box regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    """
    Calculate Inner-IoU between two boxes.
    
    Args:
        box1: Predicted boxes [N, 4] or [B, N, 4]
        box2: Target boxes [N, 4] or [B, N, 4]
        xywh: If True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)
        eps: Small value to avoid division by zero
        ratio: Scale factor for inner boxes (0.7 is default)
        
    Returns:
        Inner-IoU values
    """
    if xywh:
        # Extract center coordinates and dimensions from xywh format
        # box format: [x_center, y_center, width, height]
        x_c, y_c, w, h = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        x_c_gt, y_c_gt, w_gt, h_gt = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    else:
        # Convert from xyxy format to center coordinates
        # box format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        x_c = (x1 + x2) / 2  # center x
        y_c = (y1 + y2) / 2  # center y
        w = x2 - x1
        h = y2 - y1
        
        x1_gt, y1_gt, x2_gt, y2_gt = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
        x_c_gt = (x1_gt + x2_gt) / 2  # center x
        y_c_gt = (y1_gt + y2_gt) / 2  # center y
        w_gt = x2_gt - x1_gt
        h_gt = y2_gt - y1_gt
    
    # Calculate inner box coordinates according to Formula (19-22)
    # Inner target box: b_l^gt = x_c^gt - (w^gt * ratio) / 2
    b_l_gt = x_c_gt - (w_gt * ratio) / 2
    b_r_gt = x_c_gt + (w_gt * ratio) / 2
    b_t_gt = y_c_gt - (h_gt * ratio) / 2
    b_b_gt = y_c_gt + (h_gt * ratio) / 2
    
    # Inner prediction box: b_l = x_c - (w * ratio) / 2
    b_l = x_c - (w * ratio) / 2
    b_r = x_c + (w * ratio) / 2
    b_t = y_c - (h * ratio) / 2
    b_b = y_c + (h * ratio) / 2
    
    # Calculate intersection
    inter_left = torch.max(b_l_gt, b_l)
    inter_right = torch.min(b_r_gt, b_r)
    inter_top = torch.max(b_t_gt, b_t)
    inter_bottom = torch.min(b_b_gt, b_b)
    
    inter = torch.clamp(inter_right - inter_left, min=0) * torch.clamp(inter_bottom - inter_top, min=0)
    
    # Calculate union according to Formula (24)
    # union = (w^gt * h^gt) * (ratio)² + (w * h) * (ratio)² - inter
    area_gt = (w_gt * ratio) * (h_gt * ratio)  # (w^gt * ratio) * (h^gt * ratio)
    area_pred = (w * ratio) * (h * ratio)  # (w * ratio) * (h * ratio)
    union = area_gt + area_pred - inter + eps
    
    # Inner-IoU
    iou_inner = inter / union
    
    return iou_inner


def bbox_inner_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, 
                   eps=1e-7, ratio=0.7):
    """
    Calculate Inner-IoU with optional enhancements (GIoU, DIoU, CIoU, EIoU).
    
    Args:
        box1: Predicted boxes
        box2: Target boxes
        xywh: If True, boxes are in (x, y, w, h) format
        GIoU: Use Generalized IoU
        DIoU: Use Distance IoU
        CIoU: Use Complete IoU
        EIoU: Use Efficient IoU
        eps: Small value
        ratio: Inner box scale factor
        
    Returns:
        IoU values with enhancements
    """
    # Get basic Inner-IoU
    iou_inner = get_inner_iou(box1, box2, xywh=xywh, eps=eps, ratio=ratio)
    
    if not (GIoU or DIoU or CIoU or EIoU):
        return iou_inner
    
    # Convert to xyxy for additional calculations
    if xywh:
        from ultralytics.utils.ops import xywh2xyxy
        box1_xyxy = xywh2xyxy(box1)
        box2_xyxy = xywh2xyxy(box2)
    else:
        box1_xyxy = box1
        box2_xyxy = box2
    
    # Calculate areas
    area1 = (box1_xyxy[..., 2] - box1_xyxy[..., 0]) * (box1_xyxy[..., 3] - box1_xyxy[..., 1])
    area2 = (box2_xyxy[..., 2] - box2_xyxy[..., 0]) * (box2_xyxy[..., 3] - box2_xyxy[..., 1])
    
    # Intersection
    inter_left = torch.max(box1_xyxy[..., 0], box2_xyxy[..., 0])
    inter_right = torch.min(box1_xyxy[..., 2], box2_xyxy[..., 2])
    inter_top = torch.max(box1_xyxy[..., 1], box2_xyxy[..., 1])
    inter_bottom = torch.min(box1_xyxy[..., 3], box2_xyxy[..., 3])
    inter = torch.clamp(inter_right - inter_left, min=0) * torch.clamp(inter_bottom - inter_top, min=0)
    
    # Union
    union = area1 + area2 - inter + eps
    
    # IoU
    iou = inter / union
    
    if GIoU or DIoU or CIoU or EIoU:
        # Enclosing box
        enclose_left = torch.min(box1_xyxy[..., 0], box2_xyxy[..., 0])
        enclose_right = torch.max(box1_xyxy[..., 2], box2_xyxy[..., 2])
        enclose_top = torch.min(box1_xyxy[..., 1], box2_xyxy[..., 1])
        enclose_bottom = torch.max(box1_xyxy[..., 3], box2_xyxy[..., 3])
        enclose_area = (enclose_right - enclose_left) * (enclose_bottom - enclose_top) + eps
        
        if GIoU:
            return iou_inner - (enclose_area - union) / enclose_area
        
        # Center points
        c1_x = (box1_xyxy[..., 0] + box1_xyxy[..., 2]) / 2
        c1_y = (box1_xyxy[..., 1] + box1_xyxy[..., 3]) / 2
        c2_x = (box2_xyxy[..., 0] + box2_xyxy[..., 2]) / 2
        c2_y = (box2_xyxy[..., 1] + box2_xyxy[..., 3]) / 2
        
        # Distance between centers
        c_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
        enclose_diag_sq = (enclose_right - enclose_left) ** 2 + (enclose_bottom - enclose_top) ** 2 + eps
        
        if DIoU:
            return iou_inner - c_dist_sq / enclose_diag_sq
        
        if CIoU:
            # Aspect ratio consistency
            w1 = box1_xyxy[..., 2] - box1_xyxy[..., 0]
            h1 = box1_xyxy[..., 3] - box1_xyxy[..., 1]
            w2 = box2_xyxy[..., 2] - box2_xyxy[..., 0]
            h2 = box2_xyxy[..., 3] - box2_xyxy[..., 1]
            
            v = (4 / (torch.pi ** 2)) * torch.pow(
                torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
            )
            alpha = v / (1 - iou + v + eps)
            return iou_inner - c_dist_sq / enclose_diag_sq - alpha * v
        
        if EIoU:
            # Width and height differences according to Formula (18)
            w1 = box1_xyxy[..., 2] - box1_xyxy[..., 0]
            h1 = box1_xyxy[..., 3] - box1_xyxy[..., 1]
            w2 = box2_xyxy[..., 2] - box2_xyxy[..., 0]
            h2 = box2_xyxy[..., 3] - box2_xyxy[..., 1]
            
            w_c = enclose_right - enclose_left
            h_c = enclose_bottom - enclose_top
            
            cw_sq = (w1 - w2) ** 2 / (w_c ** 2 + eps)
            ch_sq = (h1 - h2) ** 2 / (h_c ** 2 + eps)
            
            # Calculate EIoU loss components: L_EIou = 1 - IoU + (ρ²(b, b^gt)) / c² + (ρ²(w, w^gt)) / (w^c)² + (ρ²(h, h^gt)) / (h^c)²
            l_eiou_components = c_dist_sq / enclose_diag_sq + cw_sq + ch_sq
            
            # Apply Inner-EIoU formula: L_Inner-EIou = L_EIou + IoU - IoU_inner (Formula 26)
            # L_Inner-EIou = (1 - IoU + l_eiou_components) + IoU - IoU_inner
            #              = 1 - IoU_inner + l_eiou_components
            # Since this function returns IoU value (not loss), we return:
            # iou_inner - l_eiou_components
            # When converted to loss: loss = 1 - (iou_inner - l_eiou_components) = 1 - iou_inner + l_eiou_components = L_Inner-EIou
            return iou_inner - l_eiou_components
    
    return iou_inner


class InnerIoULoss(nn.Module):
    """
    Inner-IoU Loss module for bounding box regression.
    Can be integrated into ultralytics loss calculation.
    """
    
    def __init__(self, ratio=0.7, EIoU=True):
        """
        Initialize Inner-IoU Loss.
        
        Args:
            ratio: Scale factor for inner boxes (default: 0.7)
            EIoU: Use Efficient IoU enhancement
        """
        super().__init__()
        self.ratio = ratio
        self.EIoU = EIoU
    
    def forward(self, pred_boxes, target_boxes, xywh=False, CIoU=False):
        """
        Calculate Inner-IoU loss according to Formula (26): L_Inner-EIou = L_EIou + IoU - IoU_inner
        
        Args:
            pred_boxes: Predicted boxes [N, 4]
            target_boxes: Target boxes [N, 4]
            xywh: If True, boxes are in (x, y, w, h) format
            CIoU: If True, use CIoU instead of EIoU
            
        Returns:
            Loss value according to Inner-EIoU formula
        """
        # Get Inner-IoU value (with EIoU enhancement if enabled)
        iou_value = bbox_inner_iou(
            pred_boxes, target_boxes, 
            xywh=xywh, 
            EIoU=self.EIoU,
            CIoU=CIoU,
            ratio=self.ratio
        )
        # Convert IoU value to loss: loss = 1 - IoU
        # This implements L_Inner-EIou = L_EIou + IoU - IoU_inner
        return 1.0 - iou_value

