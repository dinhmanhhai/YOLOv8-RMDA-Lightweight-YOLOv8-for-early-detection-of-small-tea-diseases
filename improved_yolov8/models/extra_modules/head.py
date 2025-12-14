# Dynamic Head for Improved YOLOv8s

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv


class ScaleAwareAttention(nn.Module):
    """Scale-aware Attention (π_L) - Dynamic weight for different scales."""
    
    def __init__(self, channels):
        super().__init__()
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Apply scale-aware attention."""
        scale_weight = self.scale_attention(x)
        return x * scale_weight


class SpatialAwareAttention(nn.Module):
    """Spatial-aware Attention (π_S) - 2D attention map on H×W."""
    
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Apply spatial-aware attention."""
        spatial_weight = self.spatial_attention(x)
        return x * spatial_weight


class TaskAwareAttention(nn.Module):
    """Task-aware Attention (π_C) - Channel-wise attention for each task."""
    
    def __init__(self, channels, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
    
    def forward(self, x):
        """Apply task-aware attention."""
        task_weights = [att(x) for att in self.task_attention]
        # Combine task weights (can be modified for different fusion strategies)
        combined_weight = torch.stack(task_weights, dim=0).mean(dim=0)
        return x * combined_weight


class DynamicHead(nn.Module):
    """
    Dynamic Head with three attention mechanisms:
    - Scale-aware Attention (π_L)
    - Spatial-aware Attention (π_S)
    - Task-aware Attention (π_C)
    """
    
    def __init__(self, in_channels, num_classes=80, num_tasks=3):
        """
        Args:
            in_channels: Input channels (should match feature pyramid channels)
            num_classes: Number of object classes
            num_tasks: Number of tasks (classification, center regression, box regression)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        
        # General View: Unified representation
        self.general_conv = Conv(in_channels, in_channels, k=1, act=True)
        
        # Three attention mechanisms
        self.scale_attention = ScaleAwareAttention(in_channels)
        self.spatial_attention = SpatialAwareAttention(in_channels)
        self.task_attention = TaskAwareAttention(in_channels, num_tasks)
        
        # Task-specific heads
        self.cls_head = nn.Sequential(
            Conv(in_channels, in_channels, k=3, act=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        
        self.center_head = nn.Sequential(
            Conv(in_channels, in_channels, k=3, act=True),
            nn.Conv2d(in_channels, 1, 1)  # Center regression
        )
        
        self.box_head = nn.Sequential(
            Conv(in_channels, in_channels, k=3, act=True),
            nn.Conv2d(in_channels, 4, 1)  # Box regression (x, y, w, h)
        )
    
    def forward(self, features):
        """
        Forward pass through Dynamic Head.
        
        Args:
            features: List of feature maps from feature pyramid [P3, P4, P5]
        
        Returns:
            List of outputs for each scale: [cls, center, box]
        """
        outputs = []
        
        for feat in features:
            # General View
            x = self.general_conv(feat)
            
            # Apply three attention mechanisms sequentially
            x = self.scale_attention(x)      # π_L
            x = self.spatial_attention(x)     # π_S
            x = self.task_attention(x)        # π_C
            
            # Task-specific outputs
            cls_out = self.cls_head(x)
            center_out = self.center_head(x)
            box_out = self.box_head(x)
            
            # Concatenate outputs: [batch, num_classes + 1 + 4, H, W]
            output = torch.cat([cls_out, center_out, box_out], dim=1)
            outputs.append(output)
        
        return outputs

