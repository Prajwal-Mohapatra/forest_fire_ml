import os
import numpy as np
import rasterio
import tensorflow as tf
import keras
import os

from utils.metrics import focal_loss, iou_score, dice_coef
from utils.preprocess import normalize_patch, create_uttarakhand_mask_from_shapefile

import tensorflow as tf
import keras
import keras.backend as K

# Enable TensorFlow GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

@keras.saving.register_keras_serializable()
def iou_score(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Intersection over Union metric for binary segmentation with configurable threshold"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Dice coefficient for binary segmentation with configurable threshold"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@keras.saving.register_keras_serializable()
def fire_recall(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Fire-specific recall metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

@keras.saving.register_keras_serializable()
def fire_precision(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Fire-specific precision metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return (true_positives + smooth) / (predicted_positives + smooth)

@keras.saving.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.6):
    """Focal loss for handling class imbalance - updated parameters"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed

custom_objects = {
        "focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.6),  # Updated parameters      
        'iou_score': iou_score,
        'dice_coef': dice_coef,
        'fire_recall': fire_recall,
        'fire_precision': fire_precision,
    }

def predict_fire_probability(model_path, input_tif_path, output_dir, 
                           patch_size=256, overlap=32, threshold=0.4,
                           save_probability=True, save_binary=True):
    """
    Predict fire probability for entire region using sliding window approach with memory optimization
    
    Args:
        model_path: Path to trained model
        input_tif_path: Path to input satellite image
        output_dir: Directory to save outputs
        patch_size: Size of patches for prediction (reduced to 256 for memory efficiency)
        overlap: Overlap between patches (reduced to 32 for memory efficiency)
        threshold: Threshold for binary classification (0-1) - Updated default to 0.3
        save_probability: Whether to save probability map
        save_binary: Whether to save binary fire/no-fire map
    
    Returns:
        dict: Contains probability map, binary map, and metadata
    """
    
    # Load model
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    from rasterio.windows import Window
    # Get image metadata without loading data
    with rasterio.open(input_tif_path) as src:
        profile = src.profile
        crs = src.crs
        transform = src.transform
        h, w = src.height, src.width
        print(f"✅ Input image metadata loaded: {h}x{w}")
        print(f"   CRS: {crs}")
        print(f"   Transform: {transform}")
        
        # Create Uttarakhand boundary mask for geographic constraints
        print("🗺️ Creating Uttarakhand boundary mask...")
        uttarakhand_mask = create_uttarakhand_mask_from_shapefile(src)
        print(f"✅ Uttarakhand mask created: {np.sum(uttarakhand_mask):,} valid pixels out of {uttarakhand_mask.size:,}")

    # Initialize full prediction arrays (use float16 to halve memory)
    prediction = np.zeros((h, w), dtype=np.float16)
    count_map = np.zeros((h, w), dtype=np.uint8)  # Smaller dtype

    # Process in windows (e.g., 4096x4096 chunks, adjust based on RAM)
    window_size = 4096
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            win_h = min(window_size, h - y)
            win_w = min(window_size, w - x)
            window = Window(x, y, win_w, win_h)

            # Read only the window
            with rasterio.open(input_tif_path) as src:
                img_data = src.read(window=window).astype(np.float32)
                img_data = np.moveaxis(img_data, 0, -1)  # (win_h, win_w, 10)

            # Normalize only this window (from preprocess.py)
            features = normalize_patch(img_data[:, :, :9])  # Output: (win_h, win_w, 12)

            # Predict on patches within this window
            stride = patch_size - overlap
            for py in range(0, win_h - patch_size + 1, stride):
                for px in range(0, win_w - patch_size + 1, stride):
                    # Get global coordinates for this patch
                    global_y = y + py
                    global_x = x + px
                    
                    # Extract Uttarakhand mask for this patch area
                    mask_patch = uttarakhand_mask[global_y:global_y+patch_size, global_x:global_x+patch_size]
                    
                    # Skip patches that are entirely outside Uttarakhand (less than 10% valid pixels)
                    valid_pixel_ratio = np.mean(mask_patch) if mask_patch.size > 0 else 0
                    if valid_pixel_ratio < 0.1:
                        continue
                    
                    patch = features[py:py+patch_size, px:px+patch_size, :]
                    patch = np.expand_dims(patch, axis=0)
                    
                    try:
                        pred_patch = model.predict(patch, verbose=0)[0, :, :, 0]  # Silent predict
                        
                        # Apply Uttarakhand mask to prediction results
                        # Set predictions outside Uttarakhand boundaries to 0
                        pred_patch_masked = pred_patch * mask_patch
                        
                        prediction[global_y:global_y+patch_size, global_x:global_x+patch_size] += pred_patch_masked
                        # Only count pixels within Uttarakhand for averaging
                        count_map[global_y:global_y+patch_size, global_x:global_x+patch_size] += mask_patch.astype(np.uint8)
                    except Exception as e:
                        print(f"❌ Prediction failed for patch at ({global_y}, {global_x}): {str(e)}")
                        continue

    # Average predictions (handle divisions carefully)
    prediction = np.divide(prediction, count_map, where=count_map != 0).astype(np.float32)
    
    # Apply final Uttarakhand mask to ensure no predictions outside boundaries
    prediction = prediction * uttarakhand_mask
    
    print(f"📊 Prediction statistics (Uttarakhand region only):")
    print(f"  - Min: {prediction.min():.4f}")
    print(f"  - Max: {prediction.max():.4f}")
    print(f"  - Mean: {prediction.mean():.4f}")
    print(f"  - Std: {prediction.std():.4f}")
    
    # Create binary fire/no-fire map
    binary_prediction = (prediction > threshold).astype(np.uint8)
    
    # Calculate fire statistics for Uttarakhand region only
    uttarakhand_pixels = np.sum(uttarakhand_mask)
    total_pixels = h * w
    fire_pixels = np.sum(binary_prediction)
    fire_percentage_uttarakhand = (fire_pixels / uttarakhand_pixels) * 100 if uttarakhand_pixels > 0 else 0
    fire_percentage_total = (fire_pixels / total_pixels) * 100
    
    print(f"🔥 Fire Detection Results (Uttarakhand Region):")
    print(f"  - Threshold used: {threshold}")
    print(f"  - Fire pixels: {fire_pixels:,}")
    print(f"  - Uttarakhand pixels: {uttarakhand_pixels:,}")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Fire coverage (Uttarakhand): {fire_percentage_uttarakhand:.2f}%")
    print(f"  - Fire coverage (Total image): {fire_percentage_total:.2f}%")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output profile
    output_profile = profile.copy()
    output_profile.update(count=1, dtype=rasterio.float32)
    
    # Remove nodata value that might be incompatible with float32
    if 'nodata' in output_profile and output_profile['nodata'] is not None:
        # For float32, use a reasonable nodata value
        output_profile.update(nodata=-9999.0)
    
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
        binary_profile.update(dtype=rasterio.uint8, nodata=255)  # Set nodata to 255 for uint8
        
        try:
            with rasterio.open(binary_path, 'w', **binary_profile) as dst:
                dst.write(binary_prediction, 1)
            print(f"✅ Binary fire/no-fire map saved to {binary_path}")
        except Exception as e:
            print(f"❌ Failed to save binary map: {str(e)}")
        
        # Also save binary maps at multiple thresholds for comparison
        thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]
        for test_threshold in thresholds_to_test:
            test_binary = (prediction > test_threshold).astype(np.uint8)
            test_binary_path = os.path.join(output_dir, f'fire_binary_map_{test_threshold:.1f}.tif')
            fire_pixels_test = np.sum(test_binary)
            fire_percentage_test = (fire_pixels_test / total_pixels) * 100
            
            try:
                with rasterio.open(test_binary_path, 'w', **binary_profile) as dst:
                    dst.write(test_binary, 1)
                print(f"✅ Binary map (threshold {test_threshold}) saved: {fire_pixels_test:,} pixels ({fire_percentage_test:.2f}%)")
            except Exception as e:
                print(f"❌ Failed to save binary map at threshold {test_threshold}: {str(e)}")
    
    # Save metadata
    metadata = {
        'threshold': threshold,
        'fire_pixels': int(fire_pixels),
        'uttarakhand_pixels': int(uttarakhand_pixels),
        'total_pixels': int(total_pixels),
        'fire_percentage_uttarakhand': fire_percentage_uttarakhand,
        'fire_percentage_total': fire_percentage_total,
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
            f.write("Fire Prediction Results (Uttarakhand Region)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Fire pixels: {fire_pixels:,}\n")
            f.write(f"Uttarakhand pixels: {uttarakhand_pixels:,}\n")
            f.write(f"Total pixels: {total_pixels:,}\n")
            f.write(f"Fire coverage (Uttarakhand): {fire_percentage_uttarakhand:.2f}%\n")
            f.write(f"Fire coverage (Total): {fire_percentage_total:.2f}%\n")
            f.write(f"Probability min: {prediction.min():.4f}\n")
            f.write(f"Probability max: {prediction.max():.4f}\n")
            f.write(f"Probability mean: {prediction.mean():.4f}\n")
            f.write(f"Probability std: {prediction.std():.4f}\n")
        print(f"✅ Metadata saved to {metadata_path}")
    except Exception as e:
        print(f"❌ Failed to save metadata: {str(e)}")
    
    # Clear TensorFlow session to release resources
    tf.keras.backend.clear_session()
    
    return {
        'probability_map': prediction,
        'binary_map': binary_prediction,
        'metadata': {
            'input_file': input_tif_path,
            'fire_percentage_uttarakhand': fire_percentage_uttarakhand,
            'fire_percentage_total': fire_percentage_total,
            'output_dir': output_dir,
            'model_file': model_path,
            'timestamp': metadata['timestamp']
        }
    }

def predict_with_confidence_zones(input_tif_path, output_dir, results=None):
    """
    Predict fire probability with confidence zones optimized for improved thresholds
    
    Creates multiple threshold maps based on updated threshold strategy:
    - High confidence fire (>0.3)       # High threshold for confident predictions
    - Medium confidence fire (0.1-0.3)  # Medium range
    - Low confidence fire (0.05-0.1)    # Low but detectable
    - No fire (<0.05)                   # Background/no-fire
    """
    
    # Get probability map    
    prediction = results['probability_map']
    
    # Debug: Print prediction statistics for threshold tuning
    print(f"\n📊 Probability Distribution for Confidence Zones:")
    print(f"  - Min: {prediction.min():.4f}")
    print(f"  - Max: {prediction.max():.4f}")
    print(f"  - Mean: {prediction.mean():.4f}")
    print(f"  - 95th percentile: {np.percentile(prediction, 95):.4f}")
    print(f"  - 90th percentile: {np.percentile(prediction, 90):.4f}")
    print(f"  - 75th percentile: {np.percentile(prediction, 75):.4f}")
    
    # Create confidence zones with updated thresholds for better balance
    confidence_map = np.zeros_like(prediction, dtype=np.uint8)
    
    # Assign confidence levels based on updated threshold strategy
    confidence_map[prediction >= 0.40] = 4   # High confidence fire (>=0.40) - Raised from 0.30
    confidence_map[(prediction >= 0.20) & (prediction < 0.40)] = 3  # Medium confidence fire (0.20-0.40) - Raised from 0.10-0.30
    confidence_map[(prediction >= 0.10) & (prediction < 0.20)] = 2   # Low confidence fire (0.10-0.20) - Raised from 0.05-0.10  
    confidence_map[prediction < 0.10] = 1    # No fire (<0.10) - Raised from <0.05
    
    # Save confidence map
    confidence_path = os.path.join(output_dir, 'fire_confidence_zones.tif')
    
    # Read profile from original
    with rasterio.open(input_tif_path) as src:
        profile = src.profile
    
    profile.update(count=1, dtype=rasterio.uint8, nodata=255)  # Set nodata to 255 for uint8
    
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
    
    print(f"\n🎯 Confidence Zone Statistics (Updated Thresholds):")
    for zone_id, zone_name in zones.items():
        zone_pixels = np.sum(confidence_map == zone_id)
        zone_percentage = (zone_pixels / confidence_map.size) * 100
        print(f"  {zone_name}: {zone_pixels:,} pixels ({zone_percentage:.2f}%)")
    
    # Additional detailed breakdown
    print(f"\n📈 Threshold Breakdown:")
    print(f"  - Pixels >= 0.40 (High): {np.sum(prediction >= 0.40):,}")
    print(f"  - Pixels 0.20-0.40 (Medium): {np.sum((prediction >= 0.20) & (prediction < 0.40)):,}")
    print(f"  - Pixels 0.10-0.20 (Low): {np.sum((prediction >= 0.10) & (prediction < 0.20)):,}")
    print(f"  - Pixels < 0.10 (None): {np.sum(prediction < 0.10):,}")
    
    return confidence_map

def predict_fire_nextday(model_path, input_tif_path, output_dir, 
                        threshold=0.4, patch_size=256, overlap=32):
    """
    Main function to predict fire/no-fire for next day
    Returns binary raster map at 30m resolution
    Updated default threshold to 0.3 for better precision/recall balance
    Updated patch_size to 256 and overlap to 32 for memory efficiency
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
    predict_with_confidence_zones(
        input_tif_path, output_dir, results
    )
    
    print(f"\n🎉 PREDICTION COMPLETE!")
    print(f"📁 Output files saved to: {output_dir}")
    print(f"   - fire_binary_map.tif (Binary fire/no-fire)")
    print(f"   - fire_probability_map.tif (Probability 0-1)")
    print(f"   - fire_confidence_zones.tif (Confidence zones)")
    print(f"   - fire_binary_map_0.1.tif, fire_binary_map_0.2.tif, fire_binary_map_0.3.tif, fire_binary_map_0.4.tif, fire_binary_map_0.5.tif (Multiple thresholds)")
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
    
    # Call the main prediction function with updated default threshold
    return predict_fire_nextday(
        model_path=model_path,
        input_tif_path=input_path,
        output_dir=output_dir,
        threshold=kwargs.get('threshold', 0.4),  # Updated default threshold from 0.3 to 0.4
        patch_size=kwargs.get('patch_size', 256),  # Reduced from 256 to 256 for memory efficiency
        overlap=kwargs.get('overlap', 32)  # Reduced from 64 to 32 for memory efficiency
    )

# Test and example usage
if __name__ == "__main__":
    # Configuration
    model_path = "/kaggle/working/forest_fire_ml/outputs/final_model.keras"
    input_tif_path = "/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked/stack_2016_04_26.tif"  # Replace with actual path
    output_dir = "outputs/fire_predictions"
    
    # Test model loading
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("🎉 Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        exit(1)
    
    # Example: Predict fire for next day with updated threshold
    if os.path.exists(input_tif_path):
        results = predict_fire_nextday(
            model_path=model_path,
            input_tif_path=input_tif_path,
            output_dir=output_dir,
            threshold=0.4  # Updated threshold for better precision/recall balance
        )
        
        print("\n📊 FINAL RESULTS:")
        print(f"Fire coverage (Uttarakhand): {results['metadata']['fire_percentage_uttarakhand']:.2f}%")
        print(f"Fire coverage (Total): {results['metadata']['fire_percentage_total']:.2f}%")
        
    else:
        print("⚠️ Input file not found. Please update the input_tif_path.")
        print("💡 Example usage:")
        print("python predict.py")
        print("   - Ensure input_tif_path points to your satellite image")
        print("   - Model will generate binary fire/no-fire map")
        print("   - Output will be saved as GeoTIFF with same CRS and resolution")

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
