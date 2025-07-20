#!/usr/bin/env python3
"""
Comprehensive Testing Script for Fire Prediction Model Fixes
============================================================
Tests all the implemented fixes before running full training:
1. LULC one-hot encoding
2. Stratified sampling
3. Enhanced augmentation
4. Consistent thresholding
5. Model architecture updates
6. Fire-focused coordinate generation
"""

import os
import sys
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dataset.loader import FireDatasetGenerator
from utils.preprocess import normalize_patch, encode_lulc_onehot, get_fire_focused_coordinates
from utils.metrics import iou_score, dice_coef, focal_loss, fire_recall, fire_precision
from model.resunet_a import build_resunet_a

def test_lulc_encoding():
    """Test LULC one-hot encoding functionality"""
    print("🧪 Testing LULC One-Hot Encoding...")
    
    # Create test data
    test_patch = np.random.rand(256, 256, 9).astype(np.float32)
    test_patch[:, :, 8] = np.random.randint(0, 4, (256, 256))  # LULC band with values 0-3
    
    # Test encoding
    try:
        encoded_patch = normalize_patch(test_patch)
        
        # Check output shape
        if encoded_patch.shape == (256, 256, 12):
            print("   ✅ Output shape correct: (256, 256, 12)")
        else:
            print(f"   ❌ Wrong output shape: {encoded_patch.shape}, expected (256, 256, 12)")
            return False
        
        # Check LULC channels (last 4 channels should be one-hot)
        lulc_channels = encoded_patch[:, :, 8:12]
        channel_sums = np.sum(lulc_channels, axis=2)
        
        # Each pixel should sum to 1 across LULC channels
        if np.allclose(channel_sums, 1.0, atol=1e-6):
            print("   ✅ LULC one-hot encoding correct")
        else:
            print(f"   ❌ LULC encoding error - channel sums range: {channel_sums.min():.6f} to {channel_sums.max():.6f}")
            return False
        
        # Test individual encoding function
        lulc_test = np.array([[0, 1], [2, 3]], dtype=np.int32)
        onehot_test = encode_lulc_onehot(lulc_test, 4)
        
        if onehot_test.shape == (2, 2, 4) and np.allclose(np.sum(onehot_test, axis=2), 1.0):
            print("   ✅ Individual LULC encoding function works")
        else:
            print("   ❌ Individual LULC encoding function failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ LULC encoding test failed: {e}")
        return False

def test_fire_focused_coordinates():
    """Test enhanced fire-focused coordinate generation"""
    print("🧪 Testing Fire-Focused Coordinate Generation...")
    
    try:
        # Create test fire mask with some fire pixels
        fire_mask = np.zeros((1000, 1000), dtype=np.float32)
        fire_mask[100:120, 100:120] = 1.0  # High fire region
        fire_mask[500:510, 500:510] = 0.5  # Medium fire region
        fire_mask[800:805, 800:805] = 0.1  # Low fire region
        
        # Test coordinate generation
        coords = get_fire_focused_coordinates(fire_mask, patch_size=256, n_patches=50, fire_ratio=0.9)
        
        if len(coords) == 50:
            print("   ✅ Correct number of coordinates generated")
        else:
            print(f"   ❌ Wrong number of coordinates: {len(coords)}, expected 50")
            return False
        
        # Check that coordinates are valid
        valid_coords = all(0 <= x <= 1000-256 and 0 <= y <= 1000-256 for x, y in coords)
        if valid_coords:
            print("   ✅ All coordinates are within valid bounds")
        else:
            print("   ❌ Some coordinates are out of bounds")
            return False
        
        # Check that we get fire-focused coordinates
        fire_focused_count = 0
        for x, y in coords:
            patch_fire = fire_mask[y:y+256, x:x+256]
            if np.sum(patch_fire) > 0:
                fire_focused_count += 1
        
        fire_focused_ratio = fire_focused_count / len(coords)
        if fire_focused_ratio > 0.5:  # Should be biased toward fire areas
            print(f"   ✅ Good fire focus: {fire_focused_ratio:.2f} of patches contain fire")
        else:
            print(f"   ⚠️ Low fire focus: {fire_focused_ratio:.2f} of patches contain fire")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fire-focused coordinates test failed: {e}")
        return False

def test_model_architecture():
    """Test updated model architecture with 12-channel input"""
    print("🧪 Testing Model Architecture...")
    
    try:
        # Build model with 12-channel input
        model = build_resunet_a(input_shape=(256, 256, 12), dropout_rate=0.2)
        
        # Check input shape
        expected_shape = (None, 256, 256, 12)
        if model.input_shape == expected_shape:
            print(f"   ✅ Correct input shape: {model.input_shape}")
        else:
            print(f"   ❌ Wrong input shape: {model.input_shape}, expected {expected_shape}")
            return False
        
        # Check output shape
        expected_output = (None, 256, 256, 1)
        if model.output_shape == expected_output:
            print(f"   ✅ Correct output shape: {model.output_shape}")
        else:
            print(f"   ❌ Wrong output shape: {model.output_shape}, expected {expected_output}")
            return False
        
        # Test forward pass
        test_input = np.random.rand(1, 256, 256, 12).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        
        if test_output.shape == (1, 256, 256, 1):
            print("   ✅ Forward pass successful")
        else:
            print(f"   ❌ Forward pass failed: output shape {test_output.shape}")
            return False
        
        # Check output range (should be 0-1 due to sigmoid)
        if 0 <= test_output.min() and test_output.max() <= 1:
            print("   ✅ Output values in valid range [0,1]")
        else:
            print(f"   ❌ Output values out of range: {test_output.min():.4f} to {test_output.max():.4f}")
            return False
        
        print(f"   ℹ️ Model has {model.count_params():,} parameters")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model architecture test failed: {e}")
        return False

def test_metrics_consistency():
    """Test metrics with consistent thresholding"""
    print("🧪 Testing Metrics Consistency...")
    
    try:
        # Create test data
        y_true = tf.constant([[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]], dtype=tf.float32)
        y_pred = tf.constant([[[[0.08]], [[0.02]], [[0.12]], [[0.04]]]], dtype=tf.float32)
        
        # Test metrics with threshold=0.05
        iou_result = iou_score(y_true, y_pred, threshold=0.05)
        dice_result = dice_coef(y_true, y_pred, threshold=0.05)
        recall_result = fire_recall(y_true, y_pred, threshold=0.05)
        precision_result = fire_precision(y_true, y_pred, threshold=0.05)
        
        print(f"   ℹ️ IoU: {float(iou_result):.4f}")
        print(f"   ℹ️ Dice: {float(dice_result):.4f}")
        print(f"   ℹ️ Recall: {float(recall_result):.4f}")
        print(f"   ℹ️ Precision: {float(precision_result):.4f}")
        
        # All metrics should be reasonable values
        if 0 <= float(iou_result) <= 1 and 0 <= float(dice_result) <= 1:
            print("   ✅ Metrics return valid ranges")
        else:
            print("   ❌ Metrics return invalid ranges")
            return False
        
        # Test focal loss
        focal = focal_loss(gamma=2.0, alpha=0.75)
        loss_result = focal(y_true, y_pred)
        
        if 0 <= float(loss_result) <= 10:  # Reasonable loss range
            print(f"   ✅ Focal loss works: {float(loss_result):.4f}")
        else:
            print(f"   ❌ Focal loss unreasonable: {float(loss_result):.4f}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Metrics test failed: {e}")
        return False

def test_data_loader():
    """Test data loader with stratified sampling"""
    print("🧪 Testing Data Loader with Stratified Sampling...")
    
    # This test requires actual data files, so we'll do a mock test
    try:
        # Create a dummy data generator to test the interface
        dummy_files = ["dummy_file_1.tif", "dummy_file_2.tif"]
        
        # Test that we can instantiate the generator with new parameters
        gen = FireDatasetGenerator(
            dummy_files,
            patch_size=256,
            batch_size=4,
            n_patches_per_img=20,
            fire_focus_ratio=0.9,
            fire_patch_ratio=0.2,  # New parameter
            augment=False
        )
        
        # Check new attributes
        if hasattr(gen, 'fire_patch_ratio') and gen.fire_patch_ratio == 0.2:
            print("   ✅ New fire_patch_ratio parameter added")
        else:
            print("   ❌ fire_patch_ratio parameter missing")
            return False
        
        if hasattr(gen, 'fire_focus_ratio') and gen.fire_focus_ratio == 0.9:
            print("   ✅ Enhanced fire_focus_ratio parameter works")
        else:
            print("   ❌ fire_focus_ratio parameter issue")
            return False
        
        # Check new augmentation
        if gen.augment_fn is not None:  # Should be None because augment=False
            print("   ❌ Augmentation should be disabled")
            return False
        else:
            print("   ✅ Augmentation control works")
        
        print("   ✅ Data loader interface updated successfully")
        print("   ⚠️ Full data loader test requires actual data files")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data loader test failed: {e}")
        return False

def test_training_configuration():
    """Test training configuration updates"""
    print("🧪 Testing Training Configuration...")
    
    try:
        # Import training configuration
        import train
        
        # Test that we can access the new configuration
        # This is a basic test since we can't run full training
        print("   ✅ Training module imports successfully")
        
        # Check if focal loss can be created with new parameters
        focal = focal_loss(gamma=2.0, alpha=0.75)
        print("   ✅ Updated focal loss parameters work")
        
        # Check if new metrics can be created
        metrics = [iou_score, dice_coef, fire_recall, fire_precision]
        print(f"   ✅ All {len(metrics)} metrics available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 STARTING COMPREHENSIVE TESTING...")
    print("=" * 60)
    
    tests = [
        ("LULC Encoding", test_lulc_encoding),
        ("Fire-Focused Coordinates", test_fire_focused_coordinates),
        ("Model Architecture", test_model_architecture),
        ("Metrics Consistency", test_metrics_consistency),
        ("Data Loader", test_data_loader),
        ("Training Configuration", test_training_configuration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TESTING SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for training.")
    else:
        print("⚠️ Some tests failed. Please fix issues before training.")
        return False
    
    return True

def main():
    """Main testing function"""
    print("Fire Prediction Model - Comprehensive Testing")
    print("=" * 60)
    
    # Set TensorFlow to not allocate all GPU memory
    if tf.config.experimental.list_physical_devices('GPU'):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration warning: {e}")
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n💡 Next Steps:")
        print("   1. Validate dataset with: python validate_dataset.py")
        print("   2. Run training with: python train.py")
        print("   3. Evaluate results with: python evaluate.py")
    else:
        print("\n🔧 Fix the failing tests before proceeding with training.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
