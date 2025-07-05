# ====================
# 8. predict.py
# ====================
import numpy as np
import rasterio
import tensorflow as tf
from utils.preprocess import normalize_patch

def predict_fire_probability(model_path, input_tif_path, output_path, patch_size=256, overlap=64):
    """
    Predict fire probability for entire region using sliding window approach
    """
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Read input image
    with rasterio.open(input_tif_path) as src:
        profile = src.profile
        img_data = src.read().astype(np.float32)
    
    # Prepare data
    img_data = np.moveaxis(img_data, 0, -1)  # (H, W, C)
    features = normalize_patch(img_data[:, :, :9])  # First 9 bands
    
    h, w = features.shape[:2]
    
    # Initialize prediction array
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # Sliding window prediction
    stride = patch_size - overlap
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = features[y:y+patch_size, x:x+patch_size, :]
            patch = np.expand_dims(patch, axis=0)  # Add batch dimension
            
            # Predict
            pred_patch = model.predict(patch, verbose=0)[0, :, :, 0]
            
            # Add to prediction map
            prediction[y:y+patch_size, x:x+patch_size] += pred_patch
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping predictions
    prediction = np.divide(prediction, count_map, out=np.zeros_like(prediction), where=count_map!=0)
    
    # Save prediction
    profile.update(count=1, dtype=rasterio.float32)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction, 1)
    
    print(f"Fire probability map saved to {output_path}")
    return prediction