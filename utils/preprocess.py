# preprocess.py
# ====================
import numpy as np
import cv2

def normalize_patch(patch, lulc_band_idx=8, n_lulc_classes=4, nodata_value=-9999):
    """
    Robust per-band normalization for a 9-band input patch with LULC one-hot encoding.
    Applies percentile-based normalization to handle outliers and nodata masking.
    Input shape: (H, W, 9) -> Output shape: (H, W, 12) after LULC encoding
    """
    # Ensure input is float32 and contiguous
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    
    # Handle nodata values by masking them
    nodata_mask = (patch == nodata_value) | np.isnan(patch) | np.isinf(patch) | (patch < -1e6)
    patch[nodata_mask] = np.nan
    
    # Separate LULC band for one-hot encoding
    lulc_band = patch[:, :, lulc_band_idx].astype(np.int32)
    lulc_band = np.nan_to_num(lulc_band, nan=0)  # Set nodata LULC to class 0
    other_bands = np.concatenate([patch[:, :, :lulc_band_idx], 
                                 patch[:, :, lulc_band_idx+1:]], axis=-1)
    
    # Normalize non-LULC bands
    norm_patch = np.zeros_like(other_bands, dtype=np.float32)
    
    for b in range(other_bands.shape[-1]):
        band = other_bands[:, :, b].astype(np.float32)
        
        # Mask nodata values before normalization
        valid_mask = ~(np.isnan(band) | np.isinf(band) | (band == nodata_value))
        valid_data = band[valid_mask]
        
        if len(valid_data) > 0:
            # Use percentile-based normalization only on valid data
            p2, p98 = np.percentile(valid_data, (2, 98))
            
            # Ensure percentiles are scalars
            p2, p98 = float(p2), float(p98)
            
            if p98 > p2:
                band = np.clip(band, p2, p98)
                norm_patch[:, :, b] = (band - p2) / (p98 - p2)
                # Set nodata areas to 0 after normalization
                norm_patch[~valid_mask, b] = 0.0
            else:
                norm_patch[:, :, b] = 0.0
        else:
            norm_patch[:, :, b] = 0.0
    
    # One-hot encode LULC band
    lulc_encoded = encode_lulc_onehot(lulc_band, n_lulc_classes)
    
    # Concatenate normalized bands with one-hot encoded LULC
    final_patch = np.concatenate([norm_patch, lulc_encoded], axis=-1)
    
    return np.ascontiguousarray(final_patch, dtype=np.float32)

def encode_lulc_onehot(lulc_band, n_classes=4):
    """
    One-hot encode the LULC (fuel) band
    Input: (H, W) with values 0-3
    Output: (H, W, 4) one-hot encoded
    """
    h, w = lulc_band.shape
    onehot = np.zeros((h, w, n_classes), dtype=np.float32)
    
    # Clip values to valid range
    lulc_band = np.clip(lulc_band, 0, n_classes - 1)
    
    # Create one-hot encoding
    for class_idx in range(n_classes):
        onehot[:, :, class_idx] = (lulc_band == class_idx).astype(np.float32)
    
    return onehot

def compute_fire_density_map(fire_mask, kernel_size=64):
    """
    Compute local fire density to identify fire-prone regions
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    fire_density = cv2.filter2D(fire_mask.astype(np.float32), -1, kernel)
    return fire_density

def get_fire_focused_coordinates(fire_mask, patch_size=256, n_patches=50, fire_ratio=0.9, min_fire_density=0.05):
    """
    Enhanced patch coordinate generation with focus on fire-prone areas and SMOTE-like augmentation
    Now ensures patches have minimum fire density
    """
    h, w = fire_mask.shape
    fire_density = compute_fire_density_map(fire_mask, kernel_size=32)  # Smaller kernel for more precision
    
    # Get fire-focused patches with multiple sensitivity levels
    n_fire_patches = int(n_patches * fire_ratio)
    fire_coords = []
    
    # Level 1: High fire density areas (>5% density) - Most important
    high_fire_indices = np.where(fire_density > 0.05)
    if len(high_fire_indices[0]) > 0:
        n_high_fire = min(n_fire_patches // 2, len(high_fire_indices[0]))
        # Add more patches from high density areas
        n_high_fire = min(n_fire_patches * 2 // 3, len(high_fire_indices[0]))
        
        for _ in range(n_high_fire):
            idx = np.random.randint(0, len(high_fire_indices[0]))
            cy, cx = int(high_fire_indices[0][idx]), int(high_fire_indices[1][idx])
            
            # Add small random offset for augmentation (SMOTE-like)
            offset_x = np.random.randint(-patch_size//8, patch_size//8)
            offset_y = np.random.randint(-patch_size//8, patch_size//8)
            
            y = max(0, min(h - patch_size, cy - patch_size // 2 + offset_y))
            x = max(0, min(w - patch_size, cx - patch_size // 2 + offset_x))
            
            # Verify this patch has sufficient fire density
            patch_fire = fire_mask[y:y+patch_size, x:x+patch_size]
            patch_density = np.mean(patch_fire) if patch_fire.size > 0 else 0
            
            if patch_density >= 0.005:  # Accept patches with at least 0.5% fire
                fire_coords.append((int(x), int(y)))
    
    # Level 2: Medium fire density areas (>2% density)
    med_fire_indices = np.where(fire_density > 0.02)
    remaining_fire_patches = n_fire_patches - len(fire_coords)
    if len(med_fire_indices[0]) > 0 and remaining_fire_patches > 0:
        n_med_fire = min(remaining_fire_patches, len(med_fire_indices[0]))
        for _ in range(n_med_fire):
            idx = np.random.randint(0, len(med_fire_indices[0]))
            cy, cx = int(med_fire_indices[0][idx]), int(med_fire_indices[1][idx])
            
            # Add small random offset for augmentation
            offset_x = np.random.randint(-patch_size//16, patch_size//16)
            offset_y = np.random.randint(-patch_size//16, patch_size//16)
            
            y = max(0, min(h - patch_size, cy - patch_size // 2 + offset_y))
            x = max(0, min(w - patch_size, cx - patch_size // 2 + offset_x))
            
            # Verify patch quality
            patch_fire = fire_mask[y:y+patch_size, x:x+patch_size]
            patch_density = np.mean(patch_fire) if patch_fire.size > 0 else 0
            
            if patch_density >= 0.002:  # Accept patches with at least 0.2% fire
                fire_coords.append((int(x), int(y)))
    
    # Level 3: Any fire activity (>0.1% density)
    any_fire_indices = np.where(fire_density > 0.001)
    remaining_fire_patches = n_fire_patches - len(fire_coords)
    if len(any_fire_indices[0]) > 0 and remaining_fire_patches > 0:
        n_any_fire = min(remaining_fire_patches, len(any_fire_indices[0]))
        for _ in range(n_any_fire):
            idx = np.random.randint(0, len(any_fire_indices[0]))
            cy, cx = int(any_fire_indices[0][idx]), int(any_fire_indices[1][idx])
            
            y = max(0, min(h - patch_size, cy - patch_size // 2))
            x = max(0, min(w - patch_size, cx - patch_size // 2))
            
            fire_coords.append((int(x), int(y)))
    
    # Add random patches for diversity (remaining slots)
    n_random_patches = n_patches - len(fire_coords)
    random_coords = []
    for _ in range(n_random_patches):
        x = int(np.random.randint(0, max(1, w - patch_size)))
        y = int(np.random.randint(0, max(1, h - patch_size)))
        random_coords.append((x, y))
    
    all_coords = fire_coords + random_coords
    print(f"🔥 Generated {len(fire_coords)} fire-focused coords and {len(random_coords)} random coords")
    print(f"   High fire density patches: {len([c for c in fire_coords if np.mean(fire_mask[c[1]:c[1]+patch_size, c[0]:c[0]+patch_size]) > 0.05])}")
    return all_coords
