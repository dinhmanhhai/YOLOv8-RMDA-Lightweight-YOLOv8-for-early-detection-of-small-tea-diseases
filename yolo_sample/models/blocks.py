"""
Block modules for improved YOLOv8 architecture.
Includes MixSPPF, RFCBAM_Neck, and C2f_RFCBAM.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f


class MixSPPF(nn.Module):
    """
    Mix Spatial Pyramid Pooling - Fast (MixSPPF) layer.
    Combines MaxPool and AvgPool to improve global information extraction.
    
    Formula:
    - y_max = Cat(Max(Max(Max(x))), Max(Max(x)), Max(x))
    - y_avg = Cat(Avg(Avg(Avg(x))), Avg(Avg(x)), Avg(x))
    - y_out = Conv(Cat(y_max, y_avg))
    """

    def __init__(self, c1, c2, k=5):
        """
        Initializes the MixSPPF layer with given input/output channels and kernel size.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size for pooling operations
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 8, c2, 1, 1)  # 8 = 4 (max) + 4 (avg)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avgpool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        Forward pass through MixSPPF layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        x = self.cv1(x)
        
        # Max pooling branch
        y1_max = self.maxpool(x)
        y2_max = self.maxpool(y1_max)
        y3_max = self.maxpool(y2_max)
        y_max = torch.cat((x, y1_max, y2_max, y3_max), 1)
        
        # Average pooling branch
        y1_avg = self.avgpool(x)
        y2_avg = self.avgpool(y1_avg)
        y3_avg = self.avgpool(y2_avg)
        y_avg = torch.cat((x, y1_avg, y2_avg, y3_avg), 1)
        
        # Concatenate and output
        return self.cv2(torch.cat((y_max, y_avg), 1))


class RFCBAM_Neck(nn.Module):
    """
    RFCBAM_Neck module used in C2f_RFCBAM.
    Structure: Conv -> RFCBAMConv (no residual connection)
    """
    def __init__(self, c1, c2, k=(3, 3)):
        """
        Initialize RFCBAM_Neck module.
        
        Args:
            c1: Input channels
            c2: Output channels  
            k: Kernel sizes for Conv and RFCBAMConv
        """
        super().__init__()
        from .rfcbam import RFCBAMConv
        self.cv1 = Conv(c1, c2, k[0], 1)
        self.cv2 = RFCBAMConv(c2, c2, k[1])
    
    def forward(self, x):
        """Forward pass through RFCBAM_Neck."""
        return self.cv2(self.cv1(x))


class C2f_RFCBAM(C2f):
    """
    C2f_RFCBAM module using RFCBAM_Neck instead of standard Bottleneck.
    Structure: Split -> RFCBAM_Neck blocks -> Concat -> Conv
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f_RFCBAM module.
        
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of RFCBAM_Neck blocks
            shortcut: Whether to use shortcut (not used in RFCBAM_Neck)
            g: Groups for convolution
            e: Expansion ratio
        """
        # Initialize parent but override the bottleneck modules
        super(C2f, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(RFCBAM_Neck(self.c, self.c, k=(3, 3)) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f_RFCBAM layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

