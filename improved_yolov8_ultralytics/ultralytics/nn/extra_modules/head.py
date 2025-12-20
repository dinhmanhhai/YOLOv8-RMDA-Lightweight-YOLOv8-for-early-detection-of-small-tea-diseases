# Dynamic Head for Improved YOLOv8s - Ultralytics Integration

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ['DynamicHead', 'Detect_DynamicHead', 'ScaleAwareAttention', 'SpatialAwareAttention', 'TaskAwareAttention', 'DynamicReLU']


def hard_sigmoid(x):
    """
    Hard sigmoid function: σ(x) = max(0, min(1, (x+1)/2))
    As defined in paper Formula 8
    """
    return torch.clamp((x + 1) / 2, 0, 1)


class ScaleAwareAttention(nn.Module):
    """
    Scale-aware Attention (π_L) - Dynamic weight for different scales.
    
    Paper Formula 8: π_L(F) · F = σ(f(1/(S·C) ΣS,C F)) · F
    where σ(x) = max(0, min(1, (x+1)/2)) is hard sigmoid
    """
    
    def __init__(self, channels):
        super().__init__()
        # f(·) is linear function approximated using 1x1 convolution
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 1/(S·C) ΣS,C F
            nn.Conv2d(channels, channels // 4, 1),  # f(·) part 1
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),  # f(·) part 2
            # Note: Hard sigmoid applied in forward, not here
        )

    def forward(self, x):
        """
        Apply scale-aware attention with hard sigmoid.
        """
        scale_weight = self.scale_attention(x)
        # Apply hard sigmoid: σ(x) = max(0, min(1, (x+1)/2))
        scale_weight = hard_sigmoid(scale_weight)
        return x * scale_weight


class SpatialAwareAttention(nn.Module):
    """
    Spatial-aware Attention (π_S) - Sparse sampling with deformable convolution.
    
    Paper Formula 9: π_S(F) · F = (1/L) Σ(l=1 to L) Σ(k=1 to K) w_l,k F(l; p_k + Δp_k; c) · Δm_k
    
    Simplified implementation:
    - Uses offset learning for adaptive sampling
    - Applies spatial attention weights
    """
    
    def __init__(self, channels, K=9):
        """
        Args:
            channels: Input channels
            K: Number of sparse sampling positions (default: 9 for 3x3 grid)
        """
        super().__init__()
        self.K = K
        
        # Learn offsets for deformable sampling: Δp_k
        # Output: [B, 2*K, H, W] for (x, y) offsets
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2 * K, 3, padding=1)
        )
        
        # Learn weight factors: Δm_k
        # Output: [B, K, H, W]
        self.weight_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, K, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Final spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply spatial-aware attention with sparse sampling.
        
        Simplified version: Uses offset learning and weight factors.
        """
        B, C, H, W = x.shape
        
        # Learn offsets and weights
        offsets = self.offset_conv(x)  # [B, 2*K, H, W]
        weights = self.weight_conv(x)   # [B, K, H, W]
        
        # Simplified: Apply weighted spatial attention
        # In full implementation, would use bilinear sampling with offsets
        spatial_attention = self.spatial_conv(x)  # [B, 1, H, W]
        
        # Combine with learned weights (simplified)
        # Average weights across K positions
        weight_avg = weights.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_attention = spatial_attention * weight_avg
        
        return x * spatial_attention


class DynamicReLU(nn.Module):
    """
    Dynamic ReLU activation function.
    
    Paper Formula 10: π_C(F) · F = max(α¹(F) · F_C + β¹(F), α²(F) · F_C + β²(F))
    where [α¹, α², β¹, β²]^T = θ(·) is the learning control activation threshold super function
    """
    
    def __init__(self, channels):
        super().__init__()
        # θ(·): Learn α¹, α², β¹, β² parameters
        self.theta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Conv2d(channels, channels * 4, 1)  # Output: [α¹, α², β¹, β²] for each channel
        )

    def forward(self, x):
        """
        Apply Dynamic ReLU: max(α¹·F_C + β¹, α²·F_C + β²)
        """
        # Get parameters: [B, 4*C, 1, 1]
        params = self.theta(x)
        B, C4, H, W = params.shape
        C = C4 // 4
        
        # Reshape to [B, C, 4, 1, 1] and split
        params = params.view(B, C, 4, 1, 1)
        alpha1 = params[:, :, 0, :, :]  # [B, C, 1, 1]
        alpha2 = params[:, :, 1, :, :]  # [B, C, 1, 1]
        beta1 = params[:, :, 2, :, :]   # [B, C, 1, 1]
        beta2 = params[:, :, 3, :, :]   # [B, C, 1, 1]
        
        # Apply Dynamic ReLU: max(α¹·F_C + β¹, α²·F_C + β²)
        out1 = alpha1 * x + beta1
        out2 = alpha2 * x + beta2
        return torch.max(out1, out2)


class TaskAwareAttention(nn.Module):
    """
    Task-aware Attention (π_C) - Channel-wise attention using Dynamic ReLU.
    
    Paper Formula 10: π_C(F) · F = max(α¹(F) · F_C + β¹(F), α²(F) · F_C + β²(F))
    """
    
    def __init__(self, channels, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        # Each task has its own Dynamic ReLU
        self.task_attention = nn.ModuleList([
            DynamicReLU(channels) for _ in range(num_tasks)
        ])

    def forward(self, x):
        """
        Apply task-aware attention using Dynamic ReLU for each task.
        """
        # Apply Dynamic ReLU for each task
        task_outputs = [att(x) for att in self.task_attention]
        # Combine task outputs (average fusion)
        combined_output = torch.stack(task_outputs, dim=0).mean(dim=0)
        return combined_output


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
            in_channels: Base input channels (target channels after projection)
            num_classes: Number of object classes
            num_tasks: Number of tasks (classification, center regression, box regression)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.base_channels = in_channels
        
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
        
        # Projection layers for different input channels (created dynamically)
        self.projections = nn.ModuleDict()
    
    def forward(self, features):
        """
        Forward pass through Dynamic Head.
        
        Args:
            features: List of feature maps from feature pyramid [P3, P4, P5]
                     Each may have different channels
        
        Returns:
            List of outputs for each scale: [cls, center, box]
        """
        outputs = []
        
        for feat in features:
            feat_channels = feat.shape[1]
            
            # Project to base_channels if needed
            if feat_channels != self.base_channels:
                # Create projection layer if not exists
                proj_key = f'proj_{feat_channels}'
                if proj_key not in self.projections:
                    proj = Conv(feat_channels, self.base_channels, k=1, act=True)
                    # Move to same device as input
                    proj = proj.to(feat.device)
                    self.projections[proj_key] = proj
                x = self.projections[proj_key](feat)
            else:
                x = feat
            
            # General View
            x = self.general_conv(x)
            
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


class Detect_DynamicHead(Detect):
    """
    Detect head using DynamicHead for Improved YOLOv8s.
    Wraps DynamicHead to be compatible with Ultralytics Detect interface.
    """
    
    def __init__(self, nc=80, ch=(), reg_max=16):
        """
        Args:
            nc: Number of classes
            ch: List of input channels for each scale
            reg_max: Maximum regression value (for DFL) - not used in DynamicHead
        """
        # Initialize base Detect class
        super().__init__(nc, ch)
        self.reg_max = reg_max
        
        # Create DynamicHead for each scale
        # Use the first channel as base_channels
        base_channels = ch[0] if ch else 256
        self.dynamic_head = DynamicHead(
            in_channels=base_channels,
            num_classes=nc,
            num_tasks=3
        )
        
        # Override no to match DynamicHead output format: [cls (nc), center (1), box (4)]
        self.no = nc + 1 + 4  # num_classes + center + box

    def forward(self, x):
        """
        Forward pass through DynamicHead.
        
        Args:
            x: List of feature maps [P3, P4, P5]
        
        Returns:
            Output in Detect format
        """
        # Get outputs from DynamicHead
        # DynamicHead returns list of [batch, nc+1+4, H, W] for each scale
        outputs = self.dynamic_head(x)
        
        if self.training:
            return outputs
        
        # Inference path - convert to Detect format
        shape = outputs[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], 2)
        
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(outputs, self.stride, 0.5))
            self.shape = shape

        # Split into box and cls
        # DynamicHead outputs: [cls (nc), center (1), box (4)]
        # We need to convert to: [box (4), cls (nc)] for compatibility
        # Note: center is not used in standard Detect, we'll use box directly
        cls = x_cat[:, :self.nc]  # First nc channels are cls
        center = x_cat[:, self.nc:self.nc+1]  # Next 1 channel is center (not used)
        box = x_cat[:, self.nc+1:self.nc+5]  # Next 4 channels are box
        
        # Decode boxes (simplified - just use box directly)
        # In standard Detect, boxes are decoded with DFL, but DynamicHead outputs direct box coords
        dbox = box  # Use box directly for now

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, outputs)
