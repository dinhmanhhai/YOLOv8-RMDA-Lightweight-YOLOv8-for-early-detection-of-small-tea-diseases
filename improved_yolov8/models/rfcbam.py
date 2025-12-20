"""
RFCBAM (Receptive Field Concentration-Based Attention Module) implementation.
"""

import torch
import torch.nn as nn
from einops import rearrange
from ultralytics.nn.modules import Conv


class SE(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""
    
    def __init__(self, in_channel, ratio=16):
        super(SE, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),  # from c -> c/r
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),  # from c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class RFCBAMConv(nn.Module):
    """
    RFCBAMConv: Receptive Field Concentration-Based Attention Module Convolution.
    Combines channel attention (SE) and spatial attention with receptive field features.
    """
    
    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize RFCBAMConv module.
        
        Args:
            c1: Input channels (YOLO format)
            c2: Output channels (YOLO format)
            k: Kernel size (must be odd, default: 3)
            s: Stride (default: 1)
        """
        super().__init__()
        # Convert YOLO format to internal format
        in_channel = c1
        out_channel = c2
        kernel_size = k
        stride = s
        
        if kernel_size % 2 == 0:
            raise ValueError(f"the kernel_size must be odd, got {kernel_size}.")
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size**2), kernel_size, 
                     padding=kernel_size//2, stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size**2)),
            nn.ReLU()
        )
        self.get_weight = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.se = SE(in_channel)
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)
        
    def forward(self, x):
        """
        Forward pass through RFCBAMConv.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C_out, H', W']
        """
        b, c = x.shape[0:2]
        
        # Channel attention
        channel_attention = self.se(x)
        
        # Generate receptive field features
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size**2, h, w)
        
        # Reshape to expanded spatial dimensions
        generate_feature = rearrange(
            generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', 
            n1=self.kernel_size, n2=self.kernel_size
        )
        
        # Apply channel attention
        unfold_feature = generate_feature * channel_attention
        
        # Spatial attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(
            torch.cat((max_feature, mean_feature), dim=1)
        )
        
        # Apply spatial attention and final convolution
        conv_data = unfold_feature * receptive_field_attention
        return self.conv(conv_data)

