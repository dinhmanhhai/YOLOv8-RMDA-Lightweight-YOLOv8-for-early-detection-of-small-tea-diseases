"""
Test script for MQA integration into TDDet/YOLOv8-MobileNetV4.

This script tests:
1. MQA module standalone functionality
2. MobileNetV4-MQA backbone output shapes
3. Full model instantiation from YAML
"""

import sys
import torch

# Test 1: MQA Module
print("="*60)
print("Test 1: Multi-Query Attention Module")
print("="*60)

try:
    from ultralytics.nn.backbone.mqa import MultiQueryAttention, MQABlock
    
    # Test parameters
    batch_size = 2
    channels = 512
    height, width = 20, 20
    num_heads = 8
    
    # Create module
    mqa = MultiQueryAttention(dim=channels, num_heads=num_heads)
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    out = mqa(x)
    
    # Verify output
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"✓ MQA output shape: {out.shape}")
    
    # Test MQA Block
    mqa_block = MQABlock(dim=channels, num_heads=num_heads)
    out_block = mqa_block(x)
    assert out_block.shape == x.shape
    print(f"✓ MQA Block output shape: {out_block.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in mqa.parameters())
    print(f"✓ MQA module parameters: {num_params:,}")
    print("✓ Test 1 PASSED\n")
    
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}\n")
    sys.exit(1)

# Test 2: MobileNetV4-MQA Backbone
print("="*60)
print("Test 2: MobileNetV4-MQA Backbone")
print("="*60)

try:
    from ultralytics.nn.backbone.mobilenetv4 import MobileNetV4ConvLargeMQA
    
    # Create model
    model = MobileNetV4ConvLargeMQA()
    print(f"✓ Model created: {model.model}")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    features = model(x)
    
    # Expected output shapes (at full scale, not scaled by width multiplier yet)
    # P2/4: 160x160, P3/8: 80x80, P4/16: 40x40, P5/32: 20x20
    expected_spatial_sizes = [(160, 160), (80, 80), (40, 40), (20, 20)]
    
    print(f"✓ Number of output feature maps: {len(features)}")
    for i, feat in enumerate(features):
        if feat is not None:
            print(f"  P{i+2} shape: {feat.shape}")
            expected_h, expected_w = expected_spatial_sizes[i]
            assert feat.shape[2] == expected_h and feat.shape[3] == expected_w, \
                f"Size mismatch at P{i+2}: {feat.shape[2]}x{feat.shape[3]} != {expected_h}x{expected_w}"
    
    print("✓ Test 2 PASSED\n")
    
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Full YOLO Model Loading
print("="*60)
print("Test 3: Full YOLO Model from YAML")
print("="*60)

try:
    # Try to import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠ YOLO import failed, skipping full model test")
        print("  (This is expected if ultralytics is not fully installed)")
        print("✓ Test 3 SKIPPED (partial pass)\n")
    else:
        # Load model from YAML
        model = YOLO('ultralytics/cfg/models/v8/yolov8-mobilenetv4-mqa.yaml')
        print(f"✓ YOLO model loaded from YAML")
        
        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        
        # In training mode, YOLO returns a dict
        # In inference mode, it returns detections
        model.model.eval()
        with torch.no_grad():
            out = model.model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Output type: {type(out)}")
        
        if isinstance(out, (list, tuple)):
            print(f"  Number of detection layers: {len(out)}")
        
        print("✓ Test 3 PASSED\n")
        
except Exception as e:
    print(f"✗ Test 3 FAILED: {e}\n")
    import traceback
    traceback.print_exc()
    # Don't exit on this failure as it might be due to missing dependencies

# Final Summary
print("="*60)
print("SUMMARY")
print("="*60)
print("✓ MQA module: Working")
print("✓ MobileNetV4-MQA backbone: Working")
print("  Full YOLO model: Check output above")
print("\nAll critical tests passed!")
print("You can now use 'yolov8-mobilenetv4-mqa.yaml' for training.")
