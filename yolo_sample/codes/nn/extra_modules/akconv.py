"""
AKConv (Adaptive Kernel Convolution) implementation.
Adaptive kernel convolution with learnable sampling positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AKConv(nn.Module):
    """
    AKConv: Adaptive Kernel Convolution.
    Dynamically adjusts convolution kernel size and shape based on input features.
    """
    
    def __init__(self, inc, outc, num_param=5, stride=1, bias=None):
        """
        Initialize AKConv module.
        
X        Structure matches Figure 9:
        Input -> Conv2d (offset) -> Resample -> Resample -> Conv -> Norm -> SiLU -> Output
        
        Args:
            inc: Input channels
            outc: Output channels
            num_param: Number of sampling parameters (default: 5 for 5x5 kernel)
            stride: Stride
            bias: Bias flag
        """
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        
        # Conv2d for generating offset (2N channels: x and y offsets for N sampling points)
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        
        # Final processing: Resample -> Conv -> Norm -> SiLU
        # After resampling, we have C*N channels, then apply conv to get output channels
        self.conv = nn.Sequential(
            nn.Conv2d(inc * num_param, outc, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        """Set learning rate for offset learning."""
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def _get_p(self, offset, dtype):
        """Get sampling positions from offset."""
        N = offset.size(1) // 2
        # Generate initial sampling positions
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(offset, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_p_n(self, N, dtype):
        """Get initial sampling positions."""
        # Generate grid of sampling positions
        # Create a simple grid pattern (e.g., 5x5 for num_param=5)
        p_n_list = []
        for i in range(-(self.num_param - 1) // 2, (self.num_param - 1) // 2 + 1):
            for j in range(-(self.num_param - 1) // 2, (self.num_param - 1) // 2 + 1):
                p_n_list.append([i, j])
        p_n = torch.tensor(p_n_list, dtype=dtype).view(1, 2 * N, 1, 1)
        return p_n

    def _get_p_0(self, offset, dtype):
        """Get center positions."""
        N = offset.size(1) // 2
        h, w = offset.size(2), offset.size(3)
        p_0_x = torch.arange(0, h, dtype=dtype).view(1, 1, h, 1).repeat(1, N, 1, w)
        p_0_y = torch.arange(0, w, dtype=dtype).view(1, 1, 1, w).repeat(1, N, h, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1)
        return p_0

    def _get_x_q(self, x, q, N):
        """Get features at sampling positions using bilinear interpolation."""
        b, h, w, _ = q.size()
        padded_x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        q = q.contiguous()
        
        # Get integer coordinates
        q_lt = q.detach().floor().long()
        q_rb = q_lt + 1
        
        # Clamp coordinates
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, padded_x.size(2) - 1),
            torch.clamp(q_lt[..., N:], 0, padded_x.size(3) - 1)
        ], dim=-1)
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, padded_x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, padded_x.size(3) - 1)
        ], dim=-1)
        
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        
        # Bilinear interpolation weights
        p = q - q_lt.type_as(q)
        g_lt = (1 - p[..., :N]) * (1 - p[..., N:])
        g_rb = p[..., :N] * p[..., N:]
        g_lb = (1 - p[..., :N]) * p[..., N:]
        g_rt = p[..., :N] * (1 - p[..., N:])
        
        # Sample features at four corners
        # Simplified version using grid_sample for better performance
        q_norm = q / torch.tensor([h - 1, w - 1], dtype=q.dtype, device=q.device) * 2.0 - 1.0
        q_norm = q_norm.view(b, h, w, N, 2)
        
        x_q_list = []
        for i in range(N):
            q_i = q_norm[..., i, :].unsqueeze(1)  # [B, 1, H, W, 2]
            x_i = F.grid_sample(padded_x, q_i, mode='bilinear', padding_mode='zeros', align_corners=False)
            x_q_list.append(x_i.squeeze(2))
        
        x_q = torch.cat(x_q_list, dim=1)  # [B, C*N, H, W]
        return x_q

    def forward(self, x):
        """
        Forward pass through AKConv.
        
        Flow matches Figure 9:
        1. Input [B, C, H, W]
        2. Conv2d -> Offset [B, 2N, H, W]
        3. Adjust initial sampled shapes by offsets
        4. Resample feature map -> [B, C, N, H, W] -> [B, C*N, H, W]
        5. Resample (reshape) -> [B, C*N, H, W]
        6. Conv -> Norm -> SiLU -> Output [B, C_out, H', W']
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C_out, H', W']
        """
        # Step 1: Generate offset from input (Conv2d)
        offset = self.p_conv(x)  # [B, 2N, H, W]
        dtype = offset.data.type()
        N = offset.size(1) // 2
        
        # Step 2: Get sampling positions (initial shapes adjusted by offsets)
        p = self._get_p(offset, dtype)  # [B, 2N, H, W]
        p = p.contiguous().permute(0, 2, 3, 1)  # [B, H, W, 2N]
        
        # Step 3: Resample feature map based on adjusted sampling shapes
        x_q = self._get_x_q(x, p, N)  # [B, C*N, H, W]
        
        # Step 4: Resample (reshape) - prepare for final conv
        # Already in shape [B, C*N, H, W], ready for conv
        
        # Step 5: Conv -> Norm -> SiLU -> Output
        return self.conv(x_q)  # [B, C_out, H', W']

