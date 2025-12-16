# Evaluation metrics for Object Detection

import torch
import numpy as np
from collections import defaultdict


def xywh2xyxy(boxes):
    """
    Convert boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2).
    
    Args:
        boxes: [N, 4] tensor in xywh format
    
    Returns:
        [N, 4] tensor in xyxy format
    """
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: [N, 4] in xyxy format
        box2: [M, 4] in xyxy format
    
    Returns:
        [N, M] IoU matrix
    """
    # Calculate intersection
    inter_x1 = torch.max(box1[:, 0:1], box2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(box1[:, 1:2], box2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(box1[:, 2:3], box2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(box1[:, 3:4], box2[:, 3].unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-7)
    return iou


def non_max_suppression(predictions, conf_threshold=0.1, iou_threshold=0.45, max_det=300):
    """
    Non-Maximum Suppression (NMS) for object detection.
    
    Args:
        predictions: List of predictions from model [P3, P4, P5]
                    Each prediction: [batch, num_classes+1+4, H, W]
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_det: Maximum number of detections
    
    Returns:
        List of detections for each image: [[boxes, scores, classes], ...]
    """
    batch_size = predictions[0].shape[0]
    num_classes = predictions[0].shape[1] - 5  # -1 (center) -4 (box)
    
    all_detections = []
    
    for b in range(batch_size):
        boxes_list = []
        scores_list = []
        classes_list = []
        
        # Process each scale
        for pred in predictions:
            pred_b = pred[b]  # [num_classes+1+4, H, W]
            H, W = pred_b.shape[1], pred_b.shape[2]
            
            # Split predictions
            cls_pred = pred_b[:num_classes, :, :]  # [num_classes, H, W]
            center_pred = pred_b[num_classes:num_classes+1, :, :]  # [1, H, W]
            box_pred = pred_b[num_classes+1:, :, :]  # [4, H, W]

            # Activation: clamp về [1e-4, 1-1e-4] để tránh 0/1 tuyệt đối
            cls_prob = torch.sigmoid(cls_pred).clamp(1e-4, 1 - 1e-4)     # [num_classes, H, W]
            center_prob = torch.sigmoid(center_pred).clamp(1e-4, 1 - 1e-4)  # [1, H, W]
            box_act = torch.sigmoid(box_pred).clamp(1e-4, 1 - 1e-4)      # xywh đều normalized 0-1
            
            # Get grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=pred_b.device, dtype=torch.float32),
                torch.arange(W, device=pred_b.device, dtype=torch.float32),
                indexing='ij'
            )
            
            # Decode boxes
            x_center = (grid_x + box_act[0, :, :]) / W  # Normalized 0-1
            y_center = (grid_y + box_act[1, :, :]) / H  # Normalized 0-1
            width = box_act[2, :, :]                    # Normalized 0-1
            height = box_act[3, :, :]                   # Normalized 0-1
            
            # Filter by confidence
            conf_mask = (center_prob.squeeze(0) > conf_threshold)  # [H, W]
            
            if conf_mask.sum() == 0:
                continue
            
            # Get valid positions
            valid_y, valid_x = torch.where(conf_mask)
            
            for y, x in zip(valid_y, valid_x):
                # Get box
                box = torch.tensor([
                    x_center[y, x].item(),
                    y_center[y, x].item(),
                    width[y, x].item(),
                    height[y, x].item()
                ], device=pred_b.device)
                
                # Get class scores
                class_scores = cls_prob[:, y, x]  # [num_classes]
                center_score = center_prob[0, y, x].item()
                
                # Get best class
                class_score, class_id = torch.max(class_scores, dim=0)
                final_score = class_score.item() * center_score
                
                if final_score > conf_threshold:
                    boxes_list.append(box)
                    scores_list.append(final_score)
                    classes_list.append(class_id.item())
        
        if len(boxes_list) == 0:
            all_detections.append({
                'boxes': torch.zeros((0, 4), device=predictions[0].device),
                'scores': torch.zeros((0,), device=predictions[0].device),
                'classes': torch.zeros((0,), dtype=torch.long, device=predictions[0].device)
            })
            continue
        
        # Convert to tensors
        boxes = torch.stack(boxes_list)  # [N, 4] in xywh format
        scores = torch.tensor(scores_list, device=boxes.device)
        classes = torch.tensor(classes_list, dtype=torch.long, device=boxes.device)
        
        # Convert to xyxy for NMS
        boxes_xyxy = xywh2xyxy(boxes)
        
        # Apply NMS per class
        keep_indices = []
        for c in range(num_classes):
            class_mask = (classes == c)
            if class_mask.sum() == 0:
                continue
            
            class_boxes = boxes_xyxy[class_mask]
            class_scores = scores[class_mask]
            
            # Sort by score
            sorted_indices = torch.argsort(class_scores, descending=True)
            class_boxes = class_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]
            
            # NMS
            keep = []
            while len(sorted_indices) > 0:
                keep.append(sorted_indices[0].item())
                if len(sorted_indices) == 1:
                    break
                
                ious = box_iou(class_boxes[0:1], class_boxes[1:])
                mask = ious.squeeze(0) < iou_threshold
                sorted_indices = sorted_indices[1:][mask]
                class_boxes = class_boxes[1:][mask]
            
            # Map back to original indices
            class_original_indices = torch.where(class_mask)[0]
            keep_indices.extend([class_original_indices[i] for i in keep])
        
        if len(keep_indices) == 0:
            all_detections.append({
                'boxes': torch.zeros((0, 4), device=predictions[0].device),
                'scores': torch.zeros((0,), device=predictions[0].device),
                'classes': torch.zeros((0,), dtype=torch.long, device=predictions[0].device)
            })
        else:
            keep_indices = sorted(keep_indices)[:max_det]
            keep_indices = torch.tensor(keep_indices, device=boxes.device, dtype=torch.long).view(-1)
            all_detections.append({
                'boxes': boxes.index_select(0, keep_indices),
                'scores': scores.index_select(0, keep_indices),
                'classes': classes.index_select(0, keep_indices)
            })
    
    return all_detections


def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision (AP) from precision-recall curve.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
    
    Returns:
        AP value
    """
    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def calculate_map(predictions, targets, num_classes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP).
    
    Args:
        predictions: List of detections for each image
                    Each detection: {'boxes': [N, 4], 'scores': [N], 'classes': [N]}
        targets: List of targets for each image
                Each target: [M, 5] in format [class_id, x_center, y_center, width, height]
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
    
    Returns:
        dict with mAP, AP per class, precision, recall
    """
    # Convert targets to xyxy format and store class info
    targets_processed = []
    for target in targets:
        if len(target) == 0:
            targets_processed.append({
                'boxes': torch.zeros((0, 4), device='cpu'),
                'classes': torch.zeros((0,), dtype=torch.long, device='cpu')
            })
        else:
            boxes_xywh = target[:, 1:5]  # [M, 4]
            boxes_xyxy = xywh2xyxy(boxes_xywh)
            classes = target[:, 0].long()
            targets_processed.append({
                'boxes': boxes_xyxy,
                'classes': classes
            })
    
    # Calculate AP for each class
    aps = []
    precisions_per_class = []
    recalls_per_class = []
    
    for c in range(num_classes):
        # Collect predictions and targets for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_target_boxes = []
        image_indices = []  # Track which image each prediction belongs to
        
        for img_idx, (pred, target) in enumerate(zip(predictions, targets_processed)):
            # Get predictions for this class
            class_mask = (pred['classes'] == c)
            if class_mask.sum() > 0:
                pred_boxes = pred['boxes'][class_mask]
                pred_scores = pred['scores'][class_mask]
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                image_indices.extend([img_idx] * len(pred_boxes))
            
            # Get targets for this class
            target_class_mask = (target['classes'] == c)
            if target_class_mask.sum() > 0:
                all_target_boxes.append({
                    'boxes': target['boxes'][target_class_mask],
                    'image_idx': img_idx
                })
        
        if len(all_pred_boxes) == 0:
            aps.append(0.0)
            precisions_per_class.append(np.array([]))
            recalls_per_class.append(np.array([]))
            continue
        
        # Concatenate all predictions
        all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
        all_pred_scores = torch.cat(all_pred_scores, dim=0)
        
        # Count total targets for this class
        total_targets = sum(len(t['boxes']) for t in all_target_boxes)
        
        if total_targets == 0:
            # All predictions are false positives
            aps.append(0.0)
            precisions_per_class.append(np.array([0.0] * len(all_pred_boxes)))
            recalls_per_class.append(np.array([0.0] * len(all_pred_boxes)))
            continue
        
        # Sort predictions by score
        sorted_indices = torch.argsort(all_pred_scores, descending=True)
        all_pred_boxes = all_pred_boxes[sorted_indices]
        all_pred_scores = all_pred_scores[sorted_indices]
        image_indices = [image_indices[i] for i in sorted_indices]
        
        # Match predictions to targets
        tp = torch.zeros(len(all_pred_boxes), dtype=torch.bool)
        fp = torch.zeros(len(all_pred_boxes), dtype=torch.bool)
        
        # Track which targets have been matched (per image)
        matched_targets = {img_idx: torch.zeros(len(t['boxes']), dtype=torch.bool, device=t['boxes'].device)
                           for img_idx, t in enumerate(targets_processed) if (t['classes'] == c).sum() > 0}
        
        for i, pred_box in enumerate(all_pred_boxes):
            img_idx = image_indices[i]
            
            # Find targets for this image
            target_info = None
            for t in all_target_boxes:
                if t['image_idx'] == img_idx:
                    target_info = t
                    break
            
            if target_info is None or len(target_info['boxes']) == 0:
                fp[i] = True
                continue
            
            # Calculate IoU with targets in this image
            ious = box_iou(pred_box.unsqueeze(0), target_info['boxes']).squeeze(0)
            
            # Find best match
            max_iou, best_target_idx = torch.max(ious, dim=0)
            
            if max_iou >= iou_threshold:
                # Check if target already matched
                if not matched_targets[img_idx][best_target_idx]:
                    tp[i] = True
                    matched_targets[img_idx][best_target_idx] = True
                else:
                    fp[i] = True
            else:
                fp[i] = True
        
        # Calculate precision and recall
        tp_cumsum = torch.cumsum(tp.float(), dim=0)
        fp_cumsum = torch.cumsum(fp.float(), dim=0)
        
        recalls = (tp_cumsum / (total_targets + 1e-7)).cpu().numpy()
        precisions = (tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)).cpu().numpy()
        
        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
        precisions_per_class.append(precisions)
        recalls_per_class.append(recalls)
    
    # Calculate mAP
    map_value = np.mean(aps) if len(aps) > 0 else 0.0
    
    # Calculate overall precision and recall (average of last precision/recall per class)
    overall_precision = np.mean([p[-1] if len(p) > 0 else 0.0 for p in precisions_per_class])
    overall_recall = np.mean([r[-1] if len(r) > 0 else 0.0 for r in recalls_per_class])
    
    return {
        'map': map_value,
        'ap_per_class': aps,
        'precision': overall_precision,
        'recall': overall_recall,
        'precisions_per_class': precisions_per_class,
        'recalls_per_class': recalls_per_class
    }


def calculate_precision_recall(predictions, targets, num_classes, iou_threshold=0.5, conf_threshold=0.1):
    """
    Calculate Precision and Recall.
    
    Args:
        predictions: List of detections
        targets: List of targets
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold
    
    Returns:
        dict with precision, recall, f1_score
    """
    # Filter predictions by confidence
    filtered_predictions = []
    for pred in predictions:
        conf_mask = pred['scores'] > conf_threshold
        filtered_predictions.append({
            'boxes': pred['boxes'][conf_mask],
            'scores': pred['scores'][conf_mask],
            'classes': pred['classes'][conf_mask]
        })
    
    # Calculate metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, target in zip(filtered_predictions, targets):
        if len(target) == 0:
            # All predictions are false positives
            total_fp += len(pred['boxes'])
            continue
        
        # Convert target to xyxy
        target_xywh = target[:, 1:5]
        target_xyxy = xywh2xyxy(target_xywh)
        target_classes = target[:, 0].long()
        
        if len(pred['boxes']) == 0:
            # All targets are false negatives
            total_fn += len(target)
            continue
        
        # Convert pred boxes to xyxy
        pred_xyxy = xywh2xyxy(pred['boxes'])
        
        # Match predictions to targets
        matched_targets = torch.zeros(len(target), dtype=torch.bool, device=target.device)
        
        for i, pred_box in enumerate(pred_xyxy):
            # Calculate IoU with all targets
            ious = box_iou(pred_box.unsqueeze(0), target_xyxy).squeeze(0)
            
            # Check class match
            class_match = (pred['classes'][i] == target_classes)
            
            # Find best match
            valid_mask = class_match & (ious >= iou_threshold)
            if valid_mask.any():
                best_idx = torch.argmax(ious * valid_mask.float())
                if not matched_targets[best_idx]:
                    total_tp += 1
                    matched_targets[best_idx] = True
                    continue
            
            total_fp += 1
        
        # Count unmatched targets as false negatives
        total_fn += (~matched_targets).sum().item()
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

