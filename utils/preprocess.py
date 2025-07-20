# preprocess.py
# ====================
import numpy as np
import cv2

def normalize_patch(patch, lulc_band_idx=8, n_lulc_classes=4):
    """
    Robust per-band normalization for a 9-band input patch with LULC one-hot encoding.
    Applies percentile-based normalization to handle outliers.
    Input shape: (H, W, 9) -> Output shape: (H, W, 12) after LULC encoding
    """
    # Ensure input is float32 and contiguous
    patch = np.ascontiguousarray(patch, dtype=np.float32)
    
    # Separate LULC band for one-hot encoding
    lulc_band = patch[:, :, lulc_band_idx].astype(np.int32)
    other_bands = np.concatenate([patch[:, :, :lulc_band_idx], 
                                 patch[:, :, lulc_band_idx+1:]], axis=-1)
    
    # Normalize non-LULC bands
    norm_patch = np.zeros_like(other_bands, dtype=np.float32)
    
    for b in range(other_bands.shape[-1]):
        band = other_bands[:, :, b].astype(np.float32)
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use percentile-based normalization for robustness
        p2, p98 = np.percentile(band, (2, 98))
        
        # Ensure percentiles are scalars
        p2, p98 = float(p2), float(p98)
        
        if p98 > p2:
            band = np.clip(band, p2, p98)
            norm_patch[:, :, b] = (band - p2) / (p98 - p2)
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

def get_fire_focused_coordinates(fire_mask, patch_size=256, n_patches=50, fire_ratio=0.9):
    """
    Enhanced patch coordinate generation with focus on fire-prone areas and SMOTE-like augmentation
    """
    h, w = fire_mask.shape
    fire_density = compute_fire_density_map(fire_mask, kernel_size=32)  # Smaller kernel for more precision
    
    # Get fire-focused patches with multiple sensitivity levels
    n_fire_patches = int(n_patches * fire_ratio)
    fire_coords = []
    
    # Level 1: High fire density areas (>2% density)
    high_fire_indices = np.where(fire_density > 0.02)
    if len(high_fire_indices[0]) > 0:
        n_high_fire = min(n_fire_patches // 2, len(high_fire_indices[0]))
        for _ in range(n_high_fire):
            idx = np.random.randint(0, len(high_fire_indices[0]))
            cy, cx = int(high_fire_indices[0][idx]), int(high_fire_indices[1][idx])
            
            # Add small random offset for augmentation (SMOTE-like)
            offset_x = np.random.randint(-patch_size//8, patch_size//8)
            offset_y = np.random.randint(-patch_size//8, patch_size//8)
            
            y = max(0, min(h - patch_size, cy - patch_size // 2 + offset_y))
            x = max(0, min(w - patch_size, cx - patch_size // 2 + offset_x))
            
            fire_coords.append((int(x), int(y)))
    
    # Level 2: Medium fire density areas (>0.5% density)
    med_fire_indices = np.where(fire_density > 0.005)
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
            
            fire_coords.append((int(x), int(y)))
    
    # Level 3: Any fire activity (>0% density)
    any_fire_indices = np.where(fire_density > 0.0001)
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
    return all_coords
