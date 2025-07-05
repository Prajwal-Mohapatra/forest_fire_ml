# preprocess.py
# ====================
import numpy as np
import cv2

def normalize_patch(patch):
    """
    Robust per-band normalization for a 9-band input patch.
    Applies percentile-based normalization to handle outliers.
    Input shape: (H, W, 9)
    """
    norm_patch = np.zeros_like(patch, dtype=np.float32)
    for b in range(patch.shape[-1]):
        band = patch[:, :, b].astype(np.float32)
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use percentile-based normalization for robustness
        p2, p98 = np.percentile(band, (2, 98))
        if p98 > p2:
            band = np.clip(band, p2, p98)
            norm_patch[:, :, b] = (band - p2) / (p98 - p2)
        else:
            norm_patch[:, :, b] = 0.0
    return norm_patch

def compute_fire_density_map(fire_mask, kernel_size=64):
    """
    Compute local fire density to identify fire-prone regions
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    fire_density = cv2.filter2D(fire_mask.astype(np.float32), -1, kernel)
    return fire_density

def get_fire_focused_coordinates(fire_mask, patch_size=256, n_patches=50, fire_ratio=0.7):
    """
    Generate patch coordinates with focus on fire-prone areas
    """
    h, w = fire_mask.shape
    fire_density = compute_fire_density_map(fire_mask)
    
    # Get fire-focused patches
    n_fire_patches = int(n_patches * fire_ratio)
    fire_coords = []
    
    # Find regions with fire activity
    fire_indices = np.where(fire_density > 0.01)  # Areas with >1% fire density
    
    if len(fire_indices[0]) > 0:
        for _ in range(n_fire_patches):
            idx = np.random.randint(0, len(fire_indices[0]))
            cy, cx = fire_indices[0][idx], fire_indices[1][idx]
            
            # Ensure patch fits within image bounds
            y = max(0, min(h - patch_size, cy - patch_size // 2))
            x = max(0, min(w - patch_size, cx - patch_size // 2))
            fire_coords.append((x, y))
    
    # Add random patches for diversity
    n_random_patches = n_patches - len(fire_coords)
    random_coords = []
    for _ in range(n_random_patches):
        x = np.random.randint(0, max(1, w - patch_size))
        y = np.random.randint(0, max(1, h - patch_size))
        random_coords.append((x, y))
    
    return fire_coords + random_coords