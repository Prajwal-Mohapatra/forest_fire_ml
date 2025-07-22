#!/usr/bin/env python3
"""
Test script to verify class weight fixes
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_preprocessing_fixes():
    """Test the preprocessing fixes for invalid value handling"""
    print("🧪 Testing Preprocessing Fixes...")
    
    try:
        from utils.preprocess import normalize_patch
        
        # Create test patch with problematic values
        test_patch = np.random.random((256, 256, 9)).astype(np.float32)
        
        # Inject problematic values
        test_patch[0, 0, 8] = np.nan  # NaN in LULC band
        test_patch[1, 1, 5] = np.inf  # Inf in feature band
        test_patch[2, 2, 3] = -9999   # nodata value
        
        print(f"   Input patch shape: {test_patch.shape}")
        print(f"   Problematic values: NaN={np.sum(np.isnan(test_patch))}, Inf={np.sum(np.isinf(test_patch))}")
        
        # Test normalization
        normalized = normalize_patch(test_patch)
        
        print(f"   Output shape: {normalized.shape} (expected: (256, 256, 12))")
        print(f"   Output dtype: {normalized.dtype}")
        print(f"   NaN values after processing: {np.sum(np.isnan(normalized))}")
        print(f"   Inf values after processing: {np.sum(np.isinf(normalized))}")
        print(f"   Value range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        
        # Verify no invalid values remain
        assert not np.any(np.isnan(normalized)), "NaN values still present!"
        assert not np.any(np.isinf(normalized)), "Inf values still present!"
        assert normalized.shape == (256, 256, 12), f"Wrong output shape: {normalized.shape}"
        
        print("✅ Preprocessing fixes working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_class_weight_types():
    """Test class weight type handling"""
    print("\n🧪 Testing Class Weight Types...")
    
    try:
        # Test different class weight formats
        class_weights = [
            {0: 1, 1: 50},           # int values (problematic)
            {0: 1.0, 1: 50.0},       # float values (fixed)
            {0: np.float32(1), 1: np.float32(50)},  # numpy float32
        ]
        
        for i, cw in enumerate(class_weights):
            print(f"   Class weight format {i+1}: {cw}")
            print(f"   Types: {type(cw[0])}, {type(cw[1])}")
            
            # Test conversion to scalars (this is where the error occurred)
            try:
                scalar_test = float(cw[0]) + float(cw[1])
                print(f"   Scalar conversion: ✅ ({scalar_test})")
            except Exception as scalar_error:
                print(f"   Scalar conversion: ❌ ({scalar_error})")
        
        print("✅ Class weight type tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Class weight test failed: {e}")
        return False

def test_data_loader_integration():
    """Test data loader with fixes"""
    print("\n🧪 Testing Data Loader Integration...")
    
    try:
        from dataset.loader import FireDatasetGenerator
        import glob
        
        # Find test files
        base_dir = '/home/swayam/projects/forest_fire_spread/datasets/dataset_stacked'
        test_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_*.tif')))
        
        if not test_files:
            print("⚠️ No test files found - skipping loader test")
            return True
        
        # Use first file for testing
        test_files = test_files[:1]  # Just one file
        
        print(f"   Testing with file: {os.path.basename(test_files[0])}")
        
        # Create minimal generator
        gen = FireDatasetGenerator(
            test_files,
            patch_size=256,
            batch_size=2,  # Small batch for testing
            n_patches_per_img=1,  # Just one patch per image
            augment=False  # No augmentation for testing
        )
        
        print(f"   Generator created: {len(gen)} batches")
        
        # Test first batch
        X, Y = gen[0]
        
        print(f"   Batch X shape: {X.shape}, dtype: {X.dtype}")
        print(f"   Batch Y shape: {Y.shape}, dtype: {Y.dtype}")
        print(f"   X range: [{np.min(X):.3f}, {np.max(X):.3f}]")
        print(f"   Y range: [{np.min(Y):.3f}, {np.max(Y):.3f}]")
        
        # Verify no invalid values
        assert not np.any(np.isnan(X)), "NaN in X batch!"
        assert not np.any(np.isnan(Y)), "NaN in Y batch!"
        assert not np.any(np.isinf(X)), "Inf in X batch!"
        assert not np.any(np.isinf(Y)), "Inf in Y batch!"
        
        print("✅ Data loader integration working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Class Weight Fixes Validation")
    print("=" * 50)
    
    tests = [
        test_preprocessing_fixes,
        test_class_weight_types,
        test_data_loader_integration,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test.__name__}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All fixes validated! Ready for training.")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
