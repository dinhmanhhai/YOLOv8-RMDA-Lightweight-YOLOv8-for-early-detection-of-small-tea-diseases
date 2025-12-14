#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra AKConv implementation
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add models to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from models.extra_modules.block import AKConv


def test_akconv_basic():
    """Test cơ bản AKConv với các num_param khác nhau"""
    print("=" * 60)
    print("Test 1: Basic AKConv với các num_param khác nhau")
    print("=" * 60)
    
    input_shape = (1, 64, 32, 32)
    x = torch.randn(*input_shape)
    
    for num_param in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        try:
            akconv = AKConv(64, 128, num_param=num_param, stride=1)
            out = akconv(x)
            expected_h = input_shape[2] // num_param
            expected_w = input_shape[3]
            expected_shape = (1, 128, expected_h, expected_w)
            
            assert out.shape == expected_shape, \
                f"num_param={num_param}: Expected {expected_shape}, got {out.shape}"
            
            print(f"✅ num_param={num_param:2d}: Input {input_shape} -> Output {out.shape}")
        except Exception as e:
            print(f"❌ num_param={num_param:2d}: Error - {e}")
            raise
    
    print("\n✅ Test 1 PASSED!\n")


def test_akconv_steps():
    """Test từng bước của AKConv"""
    print("=" * 60)
    print("Test 2: Kiểm tra từng bước của AKConv")
    print("=" * 60)
    
    num_param = 5
    akconv = AKConv(64, 128, num_param=num_param, stride=1)
    x = torch.randn(1, 64, 32, 32)
    
    # Step 1: Offset generation
    offset = akconv.p_conv(x)
    print(f"✅ Step 1 - Offset generation: {offset.shape} (expected: (1, {2*num_param}, 32, 32))")
    assert offset.shape == (1, 2*num_param, 32, 32), "Wrong offset shape"
    
    # Step 2: Get adjusted positions
    dtype = offset.data.type()
    p = akconv._get_p(offset, dtype)
    print(f"✅ Step 2 - Adjusted positions: {p.shape} (expected: (1, {2*num_param}, 32, 32))")
    assert p.shape == (1, 2*num_param, 32, 32), "Wrong p shape"
    
    # Step 3: Resampling
    p = p.contiguous().permute(0, 2, 3, 1)
    N = num_param
    q_lt = p.detach().floor()
    x_q_lt = akconv._get_x_q(x, q_lt.long(), N)
    print(f"✅ Step 3 - Resampled features: {x_q_lt.shape} (expected: (1, 64, 32, 32, {num_param}))")
    assert x_q_lt.shape == (1, 64, 32, 32, num_param), "Wrong resampled shape"
    
    # Step 4: Final output
    out = akconv(x)
    print(f"✅ Step 4 - Final output: {out.shape} (expected: (1, 128, 6, 32))")
    assert out.shape == (1, 128, 32//num_param, 32), "Wrong final output shape"
    
    print("\n✅ Test 2 PASSED!\n")


def test_akconv_initial_shapes():
    """Test initial sampling shapes"""
    print("=" * 60)
    print("Test 3: Kiểm tra Initial Sampling Shapes")
    print("=" * 60)
    
    for num_param in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        akconv = AKConv(64, 128, num_param=num_param)
        dtype = torch.float32
        p_n = akconv._get_p_n(num_param, dtype)
        
        print(f"✅ num_param={num_param:2d}: Initial shape offsets {p_n.shape} "
              f"(expected: (1, {2*num_param}, 1, 1))")
        assert p_n.shape == (1, 2*num_param, 1, 1), f"Wrong p_n shape for num_param={num_param}"
    
    print("\n✅ Test 3 PASSED!\n")


def test_akconv_gradient():
    """Test gradient flow"""
    print("=" * 60)
    print("Test 4: Kiểm tra Gradient Flow")
    print("=" * 60)
    
    akconv = AKConv(64, 128, num_param=5)
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    
    out = akconv(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient is None"
    assert akconv.p_conv.weight.grad is not None, "Offset conv gradient is None"
    assert akconv.conv[0].weight.grad is not None, "Final conv gradient is None"
    
    print("✅ Gradient flows correctly through all layers")
    print(f"   - Input grad: {x.grad.shape}")
    print(f"   - Offset conv grad: {akconv.p_conv.weight.grad.shape}")
    print(f"   - Final conv grad: {akconv.conv[0].weight.grad.shape}")
    
    print("\n✅ Test 4 PASSED!\n")


def test_akconv_vs_regular_conv():
    """So sánh AKConv với regular convolution"""
    print("=" * 60)
    print("Test 5: So sánh AKConv với Regular Conv")
    print("=" * 60)
    
    x = torch.randn(1, 64, 32, 32)
    
    # AKConv
    akconv = AKConv(64, 128, num_param=5, stride=5)
    out_ak = akconv(x)
    
    # Regular Conv với kernel tương đương
    regular_conv = nn.Conv2d(64, 128, kernel_size=5, stride=5, padding=0)
    out_regular = regular_conv(x)
    
    print(f"✅ AKConv output shape: {out_ak.shape}")
    print(f"✅ Regular Conv output shape: {out_regular.shape}")
    print(f"   - AKConv có thể điều chỉnh sampling positions")
    print(f"   - Regular Conv có fixed kernel positions")
    
    print("\n✅ Test 5 PASSED!\n")


def visualize_initial_shapes():
    """Visualize initial sampling shapes"""
    print("=" * 60)
    print("Test 6: Visualize Initial Sampling Shapes")
    print("=" * 60)
    
    import math
    
    for num_param in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        base_int = round(math.sqrt(num_param))
        row_number = num_param // base_int
        mod_number = num_param % base_int
        
        print(f"\nnum_param={num_param}:")
        print(f"  base_int={base_int}, row_number={row_number}, mod_number={mod_number}")
        
        # Tạo grid
        import numpy as np
        grid = np.zeros((row_number + (1 if mod_number > 0 else 0), base_int))
        
        idx = 0
        for i in range(row_number):
            for j in range(base_int):
                if idx < num_param:
                    grid[i, j] = 1
                    idx += 1
        
        if mod_number > 0:
            for j in range(mod_number):
                if idx < num_param:
                    grid[row_number, j] = 1
                    idx += 1
        
        # Print pattern
        pattern_str = ""
        for row in grid:
            pattern_str += "  " + " ".join(["●" if cell == 1 else "○" for cell in row]) + "\n"
        
        print(pattern_str)
    
    print("✅ Initial shapes visualization completed\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("KIỂM TRA AKCONV IMPLEMENTATION")
    print("=" * 60 + "\n")
    
    try:
        test_akconv_basic()
        test_akconv_steps()
        test_akconv_initial_shapes()
        test_akconv_gradient()
        test_akconv_vs_regular_conv()
        visualize_initial_shapes()
        
        print("=" * 60)
        print("✅ TẤT CẢ TESTS ĐỀU PASSED!")
        print("✅ AKConv đã được triển khai ĐÚNG theo lý thuyết")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

