"""
RepGFPN (Reparameterized Generalized Feature Pyramid Network) module.
Feature Pyramid Network using RepConv for efficient feature fusion.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, RepConv
from ultralytics.nn.modules.block import C2f

__all__ = ['RepGFPN']


class RepGFPN(nn.Module):
    """
    RepGFPN module for multi-scale feature fusion.
    Uses RepConv and C2f for efficient feature processing.
    Processes features at different scales with upsampling and concatenation.
    """
    
    def __init__(self, c1, c2, n=1):
        """
        Initialize RepGFPN module.
        
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of C2f blocks
        """
        super().__init__()
        # C2f for feature processing
        self.c2f = C2f(c1, c2, n=n, shortcut=False)
        # RepConv for efficient feature fusion
        self.rep_conv = RepConv(c2, c2, k=3, s=1, p=1)
    
    def forward(self, x):
        """
        Forward pass through RepGFPN.
        
        Args:
            x: Input feature map (can be single tensor or list of tensors for multi-scale)
            
        Returns:
            Processed feature map
        """
        # If input is a list (multi-scale), process each scale
        if isinstance(x, (list, tuple)):
            outputs = []
            for feat in x:
                feat = self.c2f(feat)
                feat = self.rep_conv(feat)
                outputs.append(feat)
            return outputs if len(outputs) > 1 else outputs[0]
        else:
            # Single scale input
            x = self.c2f(x)
            x = self.rep_conv(x)
            return x

