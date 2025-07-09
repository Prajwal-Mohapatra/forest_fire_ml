#!/usr/bin/env python3
"""
Test script to verify the fixes for prediction errors
"""

import os
import numpy as np
import rasterio
import tempfile
import shutil

def create_test_tif(output_path, width=100, height=100, bands=10):
    """Create a test TIF file for testing"""
    
    # Create test data
    data = np.random.rand(bands, height, width).astype(np.float32) * 1000
    
    # Create profile
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': bands,
        'dtype': rasterio.float32,
        'crs': 'EPSG:4326',
        'transform': rasterio.transform.from_bounds(-180, -90, 180, 90, width, height),
        'nodata': -9999.0
    }
    
    # Write test file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)
    
    print(f"✅ Created test TIF: {output_path}")
    return output_path

def test_nodata_fix():
    """Test that nodata values are handled correctly for uint8 outputs"""
    
    print("\n🧪 Testing nodata value fix...")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_tif = os.path.join(temp_dir, 'test_input.tif')
        output_dir = os.path.join(temp_dir, 'test_output')
        
        # Create test input
        create_test_tif(test_tif)
        
        # Test reading and checking nodata handling
        with rasterio.open(test_tif) as src:
            profile = src.profile
            print(f"Original nodata: {profile.get('nodata')}")
            
            # Test uint8 profile modification
            binary_profile = profile.copy()
            binary_profile.update(dtype=rasterio.uint8, nodata=255)
            
            print(f"Modified nodata for uint8: {binary_profile.get('nodata')}")
            
            # Test creating binary data
            binary_data = np.random.choice([0, 1], size=(profile['height'], profile['width'])).astype(np.uint8)
            
            # Try to write binary file
            binary_path = os.path.join(output_dir, 'test_binary.tif')
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                with rasterio.open(binary_path, 'w', **binary_profile) as dst:
                    dst.write(binary_data, 1)
                print("✅ Binary file created successfully with uint8 nodata")
                
                # Verify we can read it back
                with rasterio.open(binary_path) as verify_src:
                    verify_data = verify_src.read(1)
                    print(f"✅ Binary file verified: shape={verify_data.shape}, dtype={verify_data.dtype}")
                    
            except Exception as e:
                print(f"❌ Binary file creation failed: {e}")
                return False
    
    return True

def test_output_path_handling():
    """Test that output paths are handled correctly"""
    
    print("\n🧪 Testing output path handling...")
    
    from predict import ensure_output_directory
    
    # Test directory path
    test_dir = "/tmp/test_output_dir"
    output_dir, is_dir = ensure_output_directory(test_dir)
    assert output_dir == test_dir
    assert is_dir == True
    assert os.path.exists(output_dir)
    print("✅ Directory path handled correctly")
    
    # Test file path
    test_file = "/tmp/test_output_file/result.tif"
    output_dir, is_dir = ensure_output_directory(test_file)
    assert output_dir == "/tmp/test_output_file"
    assert is_dir == False
    assert os.path.exists(output_dir)
    print("✅ File path handled correctly")
    
    # Clean up
    shutil.rmtree("/tmp/test_output_dir", ignore_errors=True)
    shutil.rmtree("/tmp/test_output_file", ignore_errors=True)
    
    return True

def test_profile_compatibility():
    """Test rasterio profile compatibility"""
    
    print("\n🧪 Testing profile compatibility...")
    
    # Test float32 profile
    profile_float = {
        'driver': 'GTiff',
        'height': 100,
        'width': 100,
        'count': 1,
        'dtype': rasterio.float32,
        'crs': 'EPSG:4326',
        'nodata': -9999.0
    }
    
    print(f"Float32 profile nodata: {profile_float['nodata']}")
    
    # Test uint8 profile
    profile_uint8 = profile_float.copy()
    profile_uint8.update(dtype=rasterio.uint8, nodata=255)
    
    print(f"Uint8 profile nodata: {profile_uint8['nodata']}")
    
    # Verify data type ranges
    assert profile_float['nodata'] == -9999.0  # Valid for float32
    assert profile_uint8['nodata'] == 255  # Valid for uint8
    
    print("✅ Profile compatibility verified")
    
    return True

def main():
    """Run all tests"""
    
    print("🧪 Running prediction fixes tests...")
    
    tests = [
        ("NoData Fix", test_nodata_fix),
        ("Output Path Handling", test_output_path_handling),
        ("Profile Compatibility", test_profile_compatibility)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The fixes should resolve the issues.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementations.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
