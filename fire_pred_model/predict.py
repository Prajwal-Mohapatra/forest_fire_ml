# ====================
# 8. predict.py
# ====================
import numpy as np
import rasterio
import tensorflow as tf
from utils.preprocess import normalize_patch
from utils.metrics import focal_loss, iou_score, dice_coef

def predict_fire_probability(model_path, input_tif_path, output_path, patch_size=256, overlap=64):
    """
    Predict fire probability for entire region using sliding window approach
    """
    
    # Load model with custom objects
    custom_objects = {
        'focal_loss': focal_loss(),
        'iou_score': iou_score,
        'dice_coef': dice_coef
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        # Try loading without custom objects
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("⚠️ Model loaded without compilation (custom metrics disabled)")
        except Exception as e2:
            print(f"❌ Failed to load model even without compilation: {str(e2)}")
            raise e2
    
    # Read input image
    try:
        with rasterio.open(input_tif_path) as src:
            profile = src.profile
            img_data = src.read().astype(np.float32)
        print(f"✅ Input image loaded: {img_data.shape}")
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
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save prediction
    profile.update(count=1, dtype=rasterio.float32)
    try:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)
        print(f"✅ Fire probability map saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save prediction: {str(e)}")
        raise e
    
    return prediction
