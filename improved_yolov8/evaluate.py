#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for Improved YOLOv8s
Calculates mAP, Precision, Recall, and other metrics
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import argparse

# Add models to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import model and utilities
from train import ImprovedYOLOv8s, collate_fn
from models.data.dataset import YOLODataset, load_data_yaml
from models.utils.metrics import non_max_suppression, calculate_map, calculate_precision_recall


def evaluate_model(
    model_path,
    data_yaml='data.yaml',
    batch_size=8,
    imgsz=640,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    conf_threshold=0.25,
    iou_threshold=0.45,
    workers=4
):
    """
    Evaluate model on validation/test set.
    
    Args:
        model_path: Path to model checkpoint
        data_yaml: Dataset YAML file
        batch_size: Batch size for evaluation
        imgsz: Image size
        device: Device to run on
        conf_threshold: Confidence threshold for NMS
        iou_threshold: IoU threshold for NMS and mAP
        workers: Number of workers for DataLoader
    """
    print("=" * 60)
    print("Evaluating Improved YOLOv8s")
    print("=" * 60)
    
    # Load dataset config
    data_config = load_data_yaml(data_yaml)
    num_classes = data_config['nc']
    class_names = data_config['names']
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ImprovedYOLOv8s(num_classes=num_classes, scale='s')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print("=" * 60)
    
    # Load test/validation dataset
    test_dataset = YOLODataset(
        img_dir=data_config.get('test', data_config['val']),
        imgsz=imgsz,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['img'].to(device)
            labels = batch['labels']
            
            # Forward pass
            outputs = model(images)
            
            # Apply NMS
            predictions = non_max_suppression(
                outputs,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            # Prepare targets
            batch_size = images.shape[0]
            targets = []
            for i in range(batch_size):
                if len(labels[i]) > 0:
                    targets.append(labels[i].to(device))
                else:
                    targets.append(torch.zeros((0, 5), device=device))
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    print("\n" + "=" * 60)
    print("Calculating metrics...")
    print("=" * 60)
    
    # Calculate mAP
    map_results = calculate_map(
        all_predictions,
        all_targets,
        num_classes=num_classes,
        iou_threshold=iou_threshold
    )
    
    # Calculate Precision/Recall
    pr_results = calculate_precision_recall(
        all_predictions,
        all_targets,
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  mAP@0.5: {map_results['map']:.4f} ({map_results['map']*100:.2f}%)")
    print(f"  Precision: {pr_results['precision']:.4f} ({pr_results['precision']*100:.2f}%)")
    print(f"  Recall: {pr_results['recall']:.4f} ({pr_results['recall']*100:.2f}%)")
    print(f"  F1-Score: {pr_results['f1_score']:.4f} ({pr_results['f1_score']*100:.2f}%)")
    print(f"\n  True Positives: {pr_results['tp']}")
    print(f"  False Positives: {pr_results['fp']}")
    print(f"  False Negatives: {pr_results['fn']}")
    
    print(f"\nPer-Class AP@0.5:")
    for i, (class_name, ap) in enumerate(zip(class_names, map_results['ap_per_class'])):
        status = "✅" if ap >= 0.5 else "⚠️" if ap >= 0.3 else "❌"
        print(f"  {status} {class_name}: {ap:.4f} ({ap*100:.2f}%)")
    
    # Calculate mAP@0.5:0.95 (average over multiple IoU thresholds)
    print(f"\nCalculating mAP@0.5:0.95...")
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95
    maps_50_95 = []
    for iou_thresh in iou_thresholds:
        map_result = calculate_map(
            all_predictions,
            all_targets,
            num_classes=num_classes,
            iou_threshold=iou_thresh
        )
        maps_50_95.append(map_result['map'])
    
    map_50_95 = np.mean(maps_50_95) if len(maps_50_95) > 0 else 0.0
    print(f"  mAP@0.5:0.95: {map_50_95:.4f} ({map_50_95*100:.2f}%)")
    
    print("\n" + "=" * 60)
    
    # Evaluation summary
    print("\nEvaluation Summary:")
    if map_results['map'] >= 0.7:
        print("  ✅ mAP@0.5 >= 0.7 - Model is EXCELLENT!")
    elif map_results['map'] >= 0.5:
        print("  ✅ mAP@0.5 >= 0.5 - Model is GOOD!")
    else:
        print("  ⚠️ mAP@0.5 < 0.5 - Model needs improvement")
    
    if pr_results['precision'] >= 0.75 and pr_results['recall'] >= 0.75:
        print("  ✅ Precision and Recall >= 0.75 - Model is EXCELLENT!")
    elif pr_results['precision'] >= 0.7 and pr_results['recall'] >= 0.7:
        print("  ✅ Precision and Recall >= 0.7 - Model is GOOD!")
    else:
        print("  ⚠️ Precision or Recall < 0.7 - Model needs improvement")
    
    print("=" * 60)
    
    return {
        'map_50': map_results['map'],
        'map_50_95': map_50_95,
        'precision': pr_results['precision'],
        'recall': pr_results['recall'],
        'f1_score': pr_results['f1_score'],
        'ap_per_class': map_results['ap_per_class'],
        'class_names': class_names
    }


if __name__ == '__main__':
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Evaluate Improved YOLOv8s')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data.yaml', help='Dataset YAML file')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        workers=args.workers
    )

