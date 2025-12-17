import torch


def xywh2xyxy(boxes):
    """Convert (x, y, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def box_iou(box1, box2, eps=1e-7):
    """IoU between box1 [M,4] and box2 [N,4] (xyxy)."""
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2 - inter
    return inter / (union + eps)


class TaskAlignedAssigner:
    """
    Simplified Task-Aligned Assigner (TAL) adapted from TDDet/YOLOv8.
    Works on normalized coords (0-1) and anchor centers on feature grids.
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pred_scores: (B, N, C) logits
            pred_bboxes: (B, N, 4) in xywh (normalized 0-1)
            anchor_points: (N, 2) center points normalized 0-1
            gt_labels: (B, max_gt)
            gt_bboxes: (B, max_gt, 4) xywh normalized
            mask_gt: (B, max_gt) bool
        Returns:
            target_labels (B, N), target_bboxes (B, N, 4),
            target_scores (B, N, C), fg_mask (B, N)
        """
        B, N, C = pred_scores.shape
        max_gt = gt_labels.shape[1]

        device = pred_scores.device
        target_labels = torch.zeros((B, N), device=device, dtype=torch.long)
        target_bboxes = torch.zeros((B, N, 4), device=device)
        target_scores = torch.zeros((B, N, C), device=device)
        fg_mask = torch.zeros((B, N), device=device, dtype=torch.bool)

        if max_gt == 0:
            return target_labels, target_bboxes, target_scores, fg_mask, 0

        pred_scores_sigmoid = pred_scores.sigmoid()
        pred_bboxes_xyxy = xywh2xyxy(pred_bboxes)
        anchor_points = anchor_points.to(device)

        for b in range(B):
            num_gt = mask_gt[b].sum()
            if num_gt == 0:
                continue

            gt_xywh = gt_bboxes[b, :num_gt]  # (n,4)
            gt_xyxy = xywh2xyxy(gt_xywh)
            gt_cls = gt_labels[b, :num_gt]  # (n,)

            # anchor inside gt box
            anc_x, anc_y = anchor_points[:, 0], anchor_points[:, 1]
            x1, y1, x2, y2 = gt_xyxy.unbind(-1)
            mask_in_gts = (
                (anc_x[:, None] >= x1[None]) &
                (anc_x[:, None] <= x2[None]) &
                (anc_y[:, None] >= y1[None]) &
                (anc_y[:, None] <= y2[None])
            )  # (N, n)

            # IoU between pred bboxes and gt
            overlaps = box_iou(pred_bboxes_xyxy[b], gt_xyxy)  # (N, n)

            # classification scores per gt class
            cls_scores = pred_scores_sigmoid[b]  # (N, C)
            bbox_scores = cls_scores[:, gt_cls]  # (N, n)

            # alignment metric
            align_metric = (bbox_scores.pow(self.alpha) * overlaps.pow(self.beta))

            # top-k per gt
            topk = min(self.topk, N)
            _, topk_idx = align_metric.topk(topk, dim=0, largest=True, sorted=False)
            mask_topk = torch.zeros_like(align_metric, dtype=torch.bool)
            mask_topk.scatter_(0, topk_idx, True)

            mask_pos = mask_topk & mask_in_gts

            # resolve multiple GTs per anchor: keep best metric
            align_metric *= mask_pos
            max_over_gt, gt_idx = align_metric.max(dim=1)  # (N,)
            pos_mask = max_over_gt > 0

            fg_mask[b] = pos_mask
            if pos_mask.sum() == 0:
                continue

            matched_gt_idx = gt_idx[pos_mask]
            matched_gt_boxes = gt_xywh[matched_gt_idx]
            matched_gt_cls = gt_cls[matched_gt_idx]
            target_bboxes[b, pos_mask] = matched_gt_boxes
            target_labels[b, pos_mask] = matched_gt_cls

            # normalized score per-gt
            norm_align = max_over_gt[pos_mask]
            # avoid divide by zero
            target_scores[b, pos_mask, matched_gt_cls] = norm_align / (norm_align.max() + self.eps)

        return target_labels, target_bboxes, target_scores, fg_mask, max_gt

