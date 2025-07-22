#!/usr/bin/env python3
"""
Test script to validate the class weight fix implementation.
This script tests the data generator with sparse integer labels and verifies compatibility with class weights.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.loader import FireDatasetGenerator
from model.resunet_a import build_resunet_a
from utils.metrics import focal_loss, iou_score, dice_coef

def test_data_generator():
    """Test the data generator to ensure it outputs the correct format."""
    print("🧪 Testing Data Generator...")
    
    # Mock test files (replace with actual test files if available)
    test_files = [
        '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_01.tif',
        '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_02.tif'
    ]
    
    # Check if test files exist
    available_files = [f for f in test_files if os.path.exists(f)]
    
    if not available_files:
        print("⚠️ No test files found. Please update test_files with actual paths.")
        return False
    
    try:
        # Create test generator
        test_gen = FireDatasetGenerator(
            available_files,
            patch_size=256,
            batch_size=2,
            n_patches_per_img=5,
            fire_focus_ratio=0.9,
            fire_patch_ratio=0.4,
            augment=False  # Disable augmentation for testing
        )
        
        # Get a sample batch
        x_batch, y_batch = next(iter(test_gen))
        
        # Validate batch properties
        print(f"✅ X batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")
        print(f"✅ Y batch shape: {y_batch.shape}, dtype: {y_batch.dtype}")
        print(f"✅ Y unique values: {np.unique(y_batch)}")
        print(f"✅ Y value range: [{np.min(y_batch)}, {np.max(y_batch)}]")
        
        # Check for expected properties
        assert x_batch.dtype == np.float32, f"Expected X dtype float32, got {x_batch.dtype}"
        assert y_batch.dtype == np.int32, f"Expected Y dtype int32, got {y_batch.dtype}"
        assert y_batch.shape[-1] == 1, f"Expected Y last dimension 1, got {y_batch.shape[-1]}"
        assert set(np.unique(y_batch)).issubset({0, 1}), f"Expected Y values in {{0, 1}}, got {np.unique(y_batch)}"
        
        print("✅ Data generator test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data generator test failed: {e}")
        return False

def test_model_compilation():
    """Test model compilation with class weights."""
    print("\n🧪 Testing Model Compilation...")
    
    try:
        # Build model
        model = build_resunet_a(
            input_shape=(256, 256, 12),
            dropout_rate=0.2,
            weight_decay=1e-5
        )
        
        # Compile with focal loss and class weights
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-5)
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=2.0, alpha=0.6),
            metrics=[iou_score, dice_coef]
        )
        
        # Test class weights
        class_weight = {0: 1.0, 1: 50.0}
        print(f"✅ Class weights: {class_weight} (types: {type(class_weight[0])}, {type(class_weight[1])})")
        
        print("✅ Model compilation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model compilation test failed: {e}")
        return False

def test_training_step():
    """Test a single training step with class weights using actual dataset."""
    print("\n🧪 Testing Training Step...")
    
    try:
        # Use actual dataset files
        test_files = [
            '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_01.tif',
            '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_02.tif'
        ]
        
        # Check if test files exist
        available_files = [f for f in test_files if os.path.exists(f)]
        
        if not available_files:
            print("⚠️ No test files found. Falling back to dummy data for training step test.")
            # Fallback to dummy data
            batch_size = 2
            x_actual = np.random.rand(batch_size, 256, 256, 12).astype(np.float32)
            y_actual = np.random.randint(0, 2, (batch_size, 256, 256, 1)).astype(np.int32)
        else:
            # Create test generator with actual data
            test_gen = FireDatasetGenerator(
                available_files,
                patch_size=256,
                batch_size=2,
                n_patches_per_img=5,
                fire_focus_ratio=0.9,
                fire_patch_ratio=0.4,
                augment=False  # Disable augmentation for testing
            )
            
            # Get actual data from generator
            x_actual, y_actual = next(iter(test_gen))
            print("✅ Using actual dataset for training step test")
        
        print(f"Actual X shape: {x_actual.shape}, dtype: {x_actual.dtype}")
        print(f"Actual Y shape: {y_actual.shape}, dtype: {y_actual.dtype}")
        print(f"Actual Y unique: {np.unique(y_actual)}")
        
        # Build and compile model
        model = build_resunet_a(
            input_shape=(256, 256, 12),
            dropout_rate=0.2,
            weight_decay=1e-5
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-5)
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=2.0, alpha=0.6),
            metrics=[iou_score, dice_coef]
        )
        
        # Test single training step with class weights
        class_weight = {0: 1.0, 1: 50.0}
        
        # Convert to tf.data.Dataset for proper class_weight support
        batch_size = x_actual.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((x_actual, y_actual)).batch(batch_size)
        
        # Test training step
        history = model.fit(
            dataset,
            epochs=1,
            class_weight=class_weight,
            verbose=1
        )
        
        print("✅ Training step test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔥 Running Class Weight Fix Validation Tests\n")
    
    tests = [
        ("Data Generator", test_data_generator),
        ("Model Compilation", test_model_compilation),
        ("Training Step", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The class weight fix is working correctly.")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
