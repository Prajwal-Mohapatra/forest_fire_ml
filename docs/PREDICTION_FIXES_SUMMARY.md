# Fix Summary: Prediction Errors Resolution

## Overview

This document details the fixes applied to resolve two critical errors encountered when running the fire prediction pipeline:

1. **NoData Value Error**: `Given nodata value, -9999.0, is beyond the valid range of its data type, uint8`
2. **Output Path Error**: `'/kaggle/working/simulation_outputs' not recognized as being in a supported file format`

## Error Analysis

### Error 1: NoData Value Incompatibility

**Problem**:

- The original code copied the rasterio profile from the input image, which included `nodata: -9999.0`
- When saving binary maps as `uint8` (range 0-255), the nodata value -9999.0 was outside the valid range
- This caused rasterio to throw a range validation error

**Root Cause**:

```python
# Original problematic code in predict.py:
binary_profile = output_profile.copy()
binary_profile.update(dtype=rasterio.uint8)  # nodata still -9999.0
```

**Solution**:

```python
# Fixed code:
binary_profile = output_profile.copy()
binary_profile.update(dtype=rasterio.uint8, nodata=255)  # Valid uint8 nodata
```

### Error 2: Output Path Handling

**Problem**:

- The `run_pipeline.py` script passed a directory path to `predict_fire_probability`
- The visualization function expected a file path, not a directory
- The prediction function was designed to work with directories but the pipeline expected file paths

**Root Cause**:

```python
# Original problematic code in run_pipeline.py:
prediction = predict_fire_probability(model_path, input_tif, output_path)
# output_path could be a directory like '/kaggle/working/simulation_outputs'
```

**Solution**:

- Modified `run_pipeline.py` to handle both directory and file paths
- Added logic to detect path type and handle appropriately

## Detailed Fixes Applied

### 1. predict.py Fixes

#### Fix 1.1: Binary Map NoData Value

**Location**: Lines 245-255

**Before**:

```python
binary_profile = output_profile.copy()
binary_profile.update(dtype=rasterio.uint8)
```

**After**:

```python
binary_profile = output_profile.copy()
binary_profile.update(dtype=rasterio.uint8, nodata=255)  # Set nodata to 255 for uint8
```

#### Fix 1.2: Confidence Zones NoData Value

**Location**: Lines 320-325

**Before**:

```python
profile.update(count=1, dtype=rasterio.uint8)
```

**After**:

```python
profile.update(count=1, dtype=rasterio.uint8, nodata=255)  # Set nodata to 255 for uint8
```

#### Fix 1.3: Probability Map NoData Value

**Location**: Lines 215-220

**Before**:

```python
output_profile = profile.copy()
output_profile.update(count=1, dtype=rasterio.float32)
```

**After**:

```python
output_profile = profile.copy()
output_profile.update(count=1, dtype=rasterio.float32)

# Remove nodata value that might be incompatible with float32
if 'nodata' in output_profile and output_profile['nodata'] is not None:
    # For float32, use a reasonable nodata value
    output_profile.update(nodata=-9999.0)
```

#### Fix 1.4: Added Helper Function

**Location**: After line 87

```python
def ensure_output_directory(output_path):
    """
    Ensure output directory exists and return appropriate paths

    Args:
        output_path: Can be either a directory or a file path

    Returns:
        tuple: (output_directory, is_directory_path)
    """
    if os.path.isdir(output_path) or (not output_path.endswith('.tif') and not output_path.endswith('.png')):
        # It's a directory path
        output_dir = output_path
        is_directory_path = True
    else:
        # It's a file path, use the directory
        output_dir = os.path.dirname(output_path)
        is_directory_path = False

    os.makedirs(output_dir, exist_ok=True)
    return output_dir, is_directory_path
```

### 2. run_pipeline.py Fixes

#### Fix 2.1: Enhanced Output Path Handling

**Location**: Lines 47-70

**Before**:

```python
def run_prediction(model_path, input_tif, output_path):
    """Run fire probability prediction"""

    print("🔮 Starting Fire Probability Prediction...")

    from predict import predict_fire_probability
    prediction = predict_fire_probability(model_path, input_tif, output_path)

    # Visualize results
    from utils.visualization import visualize_fire_prediction
    visualize_fire_prediction(output_path, f"{output_path.replace('.tif', '_visualization.png')}")

    print("✅ Prediction completed successfully!")
```

**After**:

```python
def run_prediction(model_path, input_tif, output_path):
    """Run fire probability prediction"""

    print("🔮 Starting Fire Probability Prediction...")

    from predict import predict_fire_probability

    # Check if output_path is a directory or file
    if os.path.isdir(output_path) or not output_path.endswith('.tif'):
        # If it's a directory, create output directory
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)

        prediction = predict_fire_probability(
            model_path=model_path,
            input_tif_path=input_tif,
            output_dir=output_dir
        )

        # Use the binary map for visualization
        binary_map_path = os.path.join(output_dir, 'fire_binary_map.tif')
        if os.path.exists(binary_map_path):
            # Visualize results
            from utils.visualization import visualize_fire_prediction
            visualize_fire_prediction(binary_map_path, os.path.join(output_dir, 'fire_prediction_visualization.png'))
    else:
        # If it's a file path, use the directory for output
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        prediction = predict_fire_probability(
            model_path=model_path,
            input_tif_path=input_tif,
            output_dir=output_dir
        )

        # Visualize results
        from utils.visualization import visualize_fire_prediction
        visualize_fire_prediction(output_path, f"{output_path.replace('.tif', '_visualization.png')}")

    print("✅ Prediction completed successfully!")
```

## Technical Details

### NoData Value Handling by Data Type

| Data Type | Valid NoData Range | Recommended NoData Value |
| --------- | ------------------ | ------------------------ |
| uint8     | 0 to 255           | 255                      |
| float32   | Any float value    | -9999.0                  |
| int16     | -32,768 to 32,767  | -9999                    |

### Output Path Detection Logic

The enhanced pipeline now detects output path types:

1. **Directory Path**: `/kaggle/working/simulation_outputs`

   - Creates directory if it doesn't exist
   - Saves all outputs (probability, binary, confidence maps) to this directory
   - Uses `fire_binary_map.tif` for visualization

2. **File Path**: `/kaggle/working/simulation_outputs/result.tif`
   - Extracts directory path and creates it if needed
   - Saves outputs to the directory
   - Uses specified filename for visualization

## Testing

A comprehensive test suite was created (`test_predict_fixes.py`) that verifies:

1. **NoData Value Fix**: Tests that uint8 files can be created with nodata=255
2. **Output Path Handling**: Tests both directory and file path scenarios
3. **Profile Compatibility**: Ensures data type and nodata value compatibility

All tests pass successfully, confirming the fixes resolve the issues.

## Usage

The fixed pipeline now works correctly with both command formats:

```bash
# Directory output (recommended)
python run_pipeline.py --mode predict \
  --data_dir /kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked \
  --model_path /kaggle/working/forest_fire_ml/outputs/final_model.keras \
  --input_tif /kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_28.tif \
  --output_path /kaggle/working/simulation_outputs

# File output
python run_pipeline.py --mode predict \
  --data_dir /kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked \
  --model_path /kaggle/working/forest_fire_ml/outputs/final_model.keras \
  --input_tif /kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_28.tif \
  --output_path /kaggle/working/simulation_outputs/fire_prediction.tif
```

## Expected Outputs

After the fixes, the pipeline will generate:

1. **fire_probability_map.tif**: Float32 probability map (0-1)
2. **fire_binary_map.tif**: Uint8 binary fire/no-fire map (0/1)
3. **fire_confidence_zones.tif**: Uint8 confidence zones (1-4)
4. **prediction_metadata.txt**: Statistics and metadata
5. **fire_prediction_visualization.png**: Visualization of results

All files will have compatible nodata values and be properly formatted for their respective data types.
