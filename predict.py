# ====================
# 8. predict.py (ENHANCED) - Binary Fire/No-Fire Output
# ====================
import os
import numpy as np
import rasterio
import tensorflow as tf
import keras
import os

from utils.metrics import focal_loss, iou_score, dice_coef
from utils.preprocess import normalize_patch

def load_model_safe(model_path):
    """Safely load model with proper custom objects handling"""
    
    print(f"🔄 Attempting to load model from: {model_path}")
    
    # Try different loading strategies
    strategies = [
        # Strategy 1: Load with focal_loss_fixed only (we know this works)
        {
            'focal_loss_fixed': focal_loss(),
        },
        
        # Strategy 2: Load without compilation (fallback)
        None,
        
        # Strategy 3: Load with all possible custom objects
        {
            'focal_loss_fixed': focal_loss(),
            'iou_score': iou_score,
            'dice_coef': dice_coef,
            'focal_loss': focal_loss,
        },
    ]
    
    model = None
    for i, strategy in enumerate(strategies, 1):
        try:
            if strategy is None:
                # Load without compilation
                model = keras.models.load_model(model_path, compile=False)
                print(f"✅ Strategy {i}: Loaded without compilation")
                break
            else:
                # Load with custom objects
                model = keras.models.load_model(model_path, custom_objects=strategy)
                print(f"✅ Strategy {i}: Loaded with custom objects")
                break
                
        except Exception as e:
            print(f"❌ Strategy {i} failed: {str(e)}")
            continue
    
    if model is None:
        raise Exception("Failed to load model with any strategy")
    
    return model

def predict_fire_probability(model_path, input_tif_path, output_dir, 
                           patch_size=256, overlap=64, threshold=0.5,
                           save_probability=True, save_binary=True):
    """
    Predict fire probability for entire region using sliding window approach
    
    Args:
        model_path: Path to trained model
        input_tif_path: Path to input satellite image
        output_dir: Directory to save outputs
        patch_size: Size of patches for prediction
        overlap: Overlap between patches
        threshold: Threshold for binary classification (0-1)
        save_probability: Whether to save probability map
        save_binary: Whether to save binary fire/no-fire map
    
    Returns:
        dict: Contains probability map, binary map, and metadata
    """
    
    # Load model safely
    model = load_model_safe(model_path)
    
    # Read input image
    try:
        with rasterio.open(input_tif_path) as src:
            profile = src.profile
            img_data = src.read().astype(np.float32)
            crs = src.crs
            transform = src.transform
        print(f"✅ Input image loaded: {img_data.shape}")
        print(f"   CRS: {crs}")
        print(f"   Transform: {transform}")
    except Exception as e:
        print(f"❌ Failed to read input image: {str(e)}")
        raise e
    
    # Prepare data
    img_data = np.moveaxis(img_data, 0, -1)  # (H, W, C)
    features = normalize_patch(img_data[:, :, :9])  # First 9 bands
    
    h, w = features.shape[:2]
    print(f"📊 Processing image of size: {h}x{w}")
    
    # Initialize prediction array
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # Sliding window prediction
    stride = patch_size - overlap
    
    # Calculate number of patches
    n_patches_y = (h - patch_size) // stride + 1
    n_patches_x = (w - patch_size) // stride + 1
    total_patches = n_patches_y * n_patches_x
    
    print(f"🔍 Processing {total_patches} patches ({n_patches_y}x{n_patches_x})")
    
    patch_count = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = features[y:y+patch_size, x:x+patch_size, :]
            patch = np.expand_dims(patch, axis=0)  # Add batch dimension
            
            # Predict
            try:
                pred_patch = model.predict(patch, verbose=0)[0, :, :, 0]
            except Exception as e:
                print(f"❌ Prediction failed for patch at ({y}, {x}): {str(e)}")
                continue
            
            # Add to prediction map
            prediction[y:y+patch_size, x:x+patch_size] += pred_patch
            count_map[y:y+patch_size, x:x+patch_size] += 1
            
            patch_count += 1
            if patch_count % 100 == 0:
                print(f"📈 Processed {patch_count}/{total_patches} patches")
    
    # Average overlapping predictions
    prediction = np.divide(prediction, count_map, out=np.zeros_like(prediction), where=count_map!=0)
    
    print(f"📊 Prediction statistics:")
    print(f"  - Min: {prediction.min():.4f}")
    print(f"  - Max: {prediction.max():.4f}")
    print(f"  - Mean: {prediction.mean():.4f}")
    print(f"  - Std: {prediction.std():.4f}")
    
    # Create binary fire/no-fire map
    binary_prediction = (prediction > threshold).astype(np.uint8)
    
    # Calculate fire statistics
    total_pixels = h * w
    fire_pixels = np.sum(binary_prediction)
    fire_percentage = (fire_pixels / total_pixels) * 100
    
    print(f"🔥 Fire Detection Results:")
    print(f"  - Threshold used: {threshold}")
    print(f"  - Fire pixels: {fire_pixels:,}")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Fire coverage: {fire_percentage:.2f}%")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output profile
    output_profile = profile.copy()
    output_profile.update(count=1, dtype=rasterio.float32)
    
    # Save probability map
    if save_probability:
        prob_path = os.path.join(output_dir, 'fire_probability_map.tif')
        try:
            with rasterio.open(prob_path, 'w', **output_profile) as dst:
                dst.write(prediction, 1)
            print(f"✅ Fire probability map saved to {prob_path}")
        except Exception as e:
            print(f"❌ Failed to save probability map: {str(e)}")
    
    # Save binary fire/no-fire map
    if save_binary:
        binary_path = os.path.join(output_dir, 'fire_binary_map.tif')
        binary_profile = output_profile.copy()
        binary_profile.update(dtype=rasterio.uint8)
        
        try:
            with rasterio.open(binary_path, 'w', **binary_profile) as dst:
                dst.write(binary_prediction, 1)
            print(f"✅ Binary fire/no-fire map saved to {binary_path}")
        except Exception as e:
            print(f"❌ Failed to save binary map: {str(e)}")
    
    # Save metadata
    metadata = {
        'threshold': threshold,
        'fire_pixels': int(fire_pixels),
        'total_pixels': int(total_pixels),
        'fire_percentage': fire_percentage,
        'prediction_stats': {
            'min': float(prediction.min()),
            'max': float(prediction.max()),
            'mean': float(prediction.mean()),
            'std': float(prediction.std())
        }
    }
    
    metadata_path = os.path.join(output_dir, 'prediction_metadata.txt')
    try:
        with open(metadata_path, 'w') as f:
            f.write("Fire Prediction Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Fire pixels: {fire_pixels:,}\n")
            f.write(f"Total pixels: {total_pixels:,}\n")
            f.write(f"Fire coverage: {fire_percentage:.2f}%\n")
            f.write(f"Probability min: {prediction.min():.4f}\n")
            f.write(f"Probability max: {prediction.max():.4f}\n")
            f.write(f"Probability mean: {prediction.mean():.4f}\n")
            f.write(f"Probability std: {prediction.std():.4f}\n")
        print(f"✅ Metadata saved to {metadata_path}")
    except Exception as e:
        print(f"❌ Failed to save metadata: {str(e)}")
    
    return {
        'probability_map': prediction,
        'binary_map': binary_prediction,
        'metadata': metadata
    }

def predict_with_confidence_zones(model_path, input_tif_path, output_dir, 
                                patch_size=256, overlap=64):
    """
    Predict fire probability with confidence zones
    
    Creates multiple threshold maps:
    - High confidence fire (>0.8)
    - Medium confidence fire (0.5-0.8)
    - Low confidence fire (0.3-0.5)
    - No fire (<0.3)
    """
    
    # Get probability map
    results = predict_fire_probability(
        model_path, input_tif_path, output_dir,
        patch_size=patch_size, overlap=overlap,
        save_probability=True, save_binary=False
    )
    
    prediction = results['probability_map']
    
    # Create confidence zones
    confidence_map = np.zeros_like(prediction, dtype=np.uint8)
    
    # Assign confidence levels
    confidence_map[prediction >= 0.8] = 4  # High confidence fire
    confidence_map[(prediction >= 0.5) & (prediction < 0.8)] = 3  # Medium confidence fire
    confidence_map[(prediction >= 0.3) & (prediction < 0.5)] = 2  # Low confidence fire
    confidence_map[prediction < 0.3] = 1  # No fire
    
    # Save confidence map
    confidence_path = os.path.join(output_dir, 'fire_confidence_zones.tif')
    
    # Read profile from original
    with rasterio.open(input_tif_path) as src:
        profile = src.profile
    
    profile.update(count=1, dtype=rasterio.uint8)
    
    try:
        with rasterio.open(confidence_path, 'w', **profile) as dst:
            dst.write(confidence_map, 1)
        print(f"✅ Confidence zones map saved to {confidence_path}")
    except Exception as e:
        print(f"❌ Failed to save confidence map: {str(e)}")
    
    # Calculate statistics for each zone
    zones = {
        1: "No Fire",
        2: "Low Confidence Fire", 
        3: "Medium Confidence Fire",
        4: "High Confidence Fire"
    }
    
    print(f"\n🎯 Confidence Zone Statistics:")
    for zone_id, zone_name in zones.items():
        zone_pixels = np.sum(confidence_map == zone_id)
        zone_percentage = (zone_pixels / confidence_map.size) * 100
        print(f"  {zone_name}: {zone_pixels:,} pixels ({zone_percentage:.2f}%)")
    
    return confidence_map

def predict_fire_nextday(model_path, input_tif_path, output_dir, 
                        threshold=0.5, patch_size=256, overlap=64):
    """
    Main function to predict fire/no-fire for next day
    Returns binary raster map at 30m resolution
    """
    
    print("🔥 FIRE PREDICTION FOR NEXT DAY")
    print("=" * 50)
    
    # Run prediction
    results = predict_fire_probability(
        model_path=model_path,
        input_tif_path=input_tif_path,
        output_dir=output_dir,
        patch_size=patch_size,
        overlap=overlap,
        threshold=threshold,
        save_probability=True,
        save_binary=True
    )
    
    # Also create confidence zones
    confidence_map = predict_with_confidence_zones(
        model_path, input_tif_path, output_dir, patch_size, overlap
    )
    
    print(f"\n🎉 PREDICTION COMPLETE!")
    print(f"📁 Output files saved to: {output_dir}")
    print(f"   - fire_binary_map.tif (Binary fire/no-fire)")
    print(f"   - fire_probability_map.tif (Probability 0-1)")
    print(f"   - fire_confidence_zones.tif (Confidence zones)")
    print(f"   - prediction_metadata.txt (Statistics)")
    
    return results

def predict_fire_map(input_path, model_path=None, output_dir="outputs", **kwargs):
    """
    Convenience function that matches expected interface for notebook integration
    
    Args:
        input_path: Path to input geospatial data
        model_path: Path to trained model (optional)
        output_dir: Output directory
        **kwargs: Additional parameters
        
    Returns:
        dict: Prediction results including probability and binary maps
    """
    
    # Use default model path if not provided
    if model_path is None:
        # Try multiple possible model locations
        possible_paths = [
            "/kaggle/working/forest_fire_spread/forest_fire_ml/outputs/final_model.keras",
            "/home/swayam/projects/forest_fire_spread/forest_fire_ml/outputs/final_model.keras",
            "outputs/final_model.keras",
            "forest_fire_ml/outputs/final_model.keras",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("No trained model found. Please provide model_path.")
    
    # Call the main prediction function
    return predict_fire_nextday(
        model_path=model_path,
        input_tif_path=input_path,
        output_dir=output_dir,
        threshold=kwargs.get('threshold', 0.5),
        patch_size=kwargs.get('patch_size', 256),
        overlap=kwargs.get('overlap', 64)
    )

# Test and example usage
if __name__ == "__main__":
    # Configuration
    model_path = "/kaggle/working/forest_fire_ml/outputs/final_model.keras"
    input_tif_path = "/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_26.tif"  # Replace with actual path
    output_dir = "outputs/fire_predictions"
    
    # Test model loading
    try:
        model = load_model_safe(model_path)
        print("🎉 Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        exit(1)
    
    # Example: Predict fire for next day
    if os.path.exists(input_tif_path):
        results = predict_fire_nextday(
            model_path=model_path,
            input_tif_path=input_tif_path,
            output_dir=output_dir,
            threshold=0.5  # Adjust threshold as needed
        )
        
        print("\n📊 FINAL RESULTS:")
        print(f"Fire coverage: {results['metadata']['fire_percentage']:.2f}%")
        
    else:
        print("⚠️ Input file not found. Please update the input_tif_path.")
        print("💡 Example usage:")
        print("python predict.py")
        print("   - Ensure input_tif_path points to your satellite image")
        print("   - Model will generate binary fire/no-fire map")
        print("   - Output will be saved as GeoTIFF with same CRS and resolution")
