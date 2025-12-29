"""
Multi-Query Attention (MQA) module for TDDet.

This module implements Multi-Query Attention mechanism which reduces computational complexity
by sharing key and value projections across all attention heads while maintaining independent
query projections.

Reference: "TDDet: A novel lightweight and efficient tea disease detector"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) module.
    
    MQA reduces memory bandwidth requirements and computational complexity by sharing
    keys and values across all attention heads while keeping independent queries.
    
    Complexity:
        - Traditional Multi-Head Attention: O(B * H * (N * M + N^2 + M^2))
        - Multi-Query Attention: O(B * (N * M + N^2))
    
    where B=batch size, H=number of heads, N=input sequence length, M=context sequence length.
    
    Args:
        dim (int): Input channel dimension
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool): If True, add bias to qkv projections. Default: True
        attn_drop (float): Attention dropout rate. Default: 0.0
        proj_drop (float): Output projection dropout rate. Default: 0.0
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Independent query projections for each head
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Shared key and value projections (single head)
        self.k = nn.Linear(dim, self.head_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.head_dim, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        Forward pass of Multi-Query Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W  # sequence length
        
        # Reshape: (B, C, H, W) -> (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Query projection: (B, N, C) -> (B, num_heads, N, head_dim)
        q = self.q(x_flat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Shared key and value projections: (B, N, C) -> (B, N, head_dim)
        k = self.k(x_flat)  # (B, N, head_dim)
        v = self.v(x_flat)  # (B, N, head_dim)
        
        # Expand k and v to match number of heads: (B, N, head_dim) -> (B, num_heads, N, head_dim)
        # Broadcasting will handle this during attention computation
        k = k.unsqueeze(1).expand(B, self.num_heads, N, self.head_dim)
        v = v.unsqueeze(1).expand(B, self.num_heads, N, self.head_dim)
        
        # Scaled dot-product attention
        # q: (B, num_heads, N, head_dim)
        # k: (B, num_heads, N, head_dim)
        # attn: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        # attn: (B, num_heads, N, N)
        # v: (B, num_heads, N, head_dim)
        # out: (B, num_heads, N, head_dim)
        out = attn @ v
        
        # Reshape: (B, num_heads, N, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Reshape back: (B, N, C) -> (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class MQABlock(nn.Module):
    """
    MQA Block with residual connection and layer normalization.
    
    This is a wrapper around MultiQueryAttention that adds normalization
    and residual connection for better training stability.
    
    Args:
        dim (int): Input channel dimension
        num_heads (int): Number of attention heads. Default: 8
    """
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.mqa = MultiQueryAttention(dim, num_heads)
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
        """
        return x + self.mqa(self.norm(x))


if __name__ == "__main__":
    # Test MQA module
    print("Testing Multi-Query Attention module...")
    
    # Test parameters
    batch_size = 2
    channels = 512
    height = 20
    width = 20
    num_heads = 8
    
    # Create module
    mqa = MultiQueryAttention(dim=channels, num_heads=num_heads)
    
    # Create random input
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    out = mqa(x)
    
    # Check output shape
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"✓ MQA output shape correct: {out.shape}")
    
    # Test MQA Block
    mqa_block = MQABlock(dim=channels, num_heads=num_heads)
    out_block = mqa_block(x)
    assert out_block.shape == x.shape, f"Shape mismatch: {out_block.shape} != {x.shape}"
    print(f"✓ MQA Block output shape correct: {out_block.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in mqa.parameters())
    print(f"✓ MQA module parameters: {num_params:,}")
    
    print("\nAll tests passed!")
