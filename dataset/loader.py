# 5. dataset/loader.py
# ====================
import os
import numpy as np
import rasterio
from keras.utils import Sequence
from datetime import datetime, timedelta
import albumentations as A
from utils.preprocess import normalize_patch, get_fire_focused_coordinates

class FireDatasetGenerator(Sequence):
    def __init__(self, tif_paths, patch_size=256, batch_size=8, n_patches_per_img=50,
                 shuffle=True, augment=True, fire_focus_ratio=0.9, fire_patch_ratio=0.2, **kwargs):
        super().__init__(**kwargs)
        
        self.tif_paths = sorted(tif_paths)  # Ensure chronological order
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_patches_per_img = n_patches_per_img
        self.shuffle = shuffle
        self.fire_focus_ratio = fire_focus_ratio
        self.fire_patch_ratio = fire_patch_ratio  # Minimum ratio of fire-positive patches per batch
        
        # Setup augmentation pipeline - Multi-channel compatible (9-band satellite data)
        self.augment_fn = None
        if augment:
            self.augment_fn = A.Compose([
                # Geometric augmentations (work with any number of channels)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),  # scale_limit=0.1 means 90% to 110% scaling
                    translate_percent=(-0.1, 0.1),  # shift_limit=0.1 means ±10% translation
                    rotate=(-15, 15),  # rotate_limit=15 means ±15 degrees rotation
                    p=0.5
                ),  # Replaced ShiftScaleRotate with proper Affine parameters
                
                # Simple photometric augmentations (fixed parameters)
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.4),  # Fixed: tuple format
                A.RandomGamma(gamma_limit=(1.0, 1.3), p=0.3),  # Fixed: all values must be >= 1.0
                
                # Simple noise (using correct parameter)
                A.GaussNoise(noise_scale_factor=0.1, p=0.2),  # Fixed: noise_scale_factor instead of var_limit
            ])
        
        # Pre-compute patch coordinates for each day
        self.samples = self._generate_temporal_patches()
        self.on_epoch_end()
        
        print(f"✅ Dataset loaded successfully! {len(self.samples)} patches from {len(self.tif_paths)} days.")

    def _generate_temporal_patches(self):
        """Generate patches with temporal awareness and fire focus"""
        all_samples = []
        fire_samples = []
        no_fire_samples = []
        
        for day_idx, tif_path in enumerate(self.tif_paths):
            try:
                with rasterio.open(tif_path) as src:
                    # Read fire mask to identify fire-prone areas
                    fire_mask = src.read(10)  # Band 10 is fire mask
                    
                    # Get fire-focused coordinates
                    coords = get_fire_focused_coordinates(
                        fire_mask, self.patch_size, self.n_patches_per_img, self.fire_focus_ratio
                    )
                    
                    # Add temporal context and separate fire from no-fire samples
                    for x, y in coords:
                        # Ensure fire density is a scalar value
                        fire_patch = fire_mask[y:y+self.patch_size, x:x+self.patch_size]
                        fire_density = float(np.mean(fire_patch))  # Explicitly convert to Python float
                        
                        sample = {
                            'tif_path': tif_path,
                            'day_idx': day_idx,
                            'x': int(x),  # Ensure coordinates are Python ints
                            'y': int(y),
                            'fire_density': fire_density
                        }
                        
                        # Separate samples based on fire presence
                        if fire_density > 0.001:  # More sensitive threshold for fire detection
                            fire_samples.append(sample)
                        else:
                            no_fire_samples.append(sample)
                        
                        all_samples.append(sample)
                        
            except Exception as e:
                print(f"⚠️ Error processing {tif_path}: {e}")
                continue
        
        # Store separated samples for stratified sampling
        self.fire_samples = fire_samples
        self.no_fire_samples = no_fire_samples
        
        # Sort by fire density to prioritize fire-rich patches
        all_samples.sort(key=lambda x: x['fire_density'], reverse=True)
        print(f"📊 Dataset composition: {len(fire_samples)} fire patches, {len(no_fire_samples)} no-fire patches")
        return all_samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        # Stratified sampling: ensure minimum fire patches per batch
        n_fire_patches = max(1, int(self.batch_size * self.fire_patch_ratio))
        n_no_fire_patches = self.batch_size - n_fire_patches
        
        # Sample fire patches
        fire_batch = []
        if len(self.fire_samples) > 0:
            fire_indices = np.random.choice(len(self.fire_samples), 
                                          size=min(n_fire_patches, len(self.fire_samples)), 
                                          replace=True)
            fire_batch = [self.fire_samples[i] for i in fire_indices]
        
        # Sample no-fire patches
        no_fire_batch = []
        if len(self.no_fire_samples) > 0 and n_no_fire_patches > 0:
            no_fire_indices = np.random.choice(len(self.no_fire_samples), 
                                             size=min(n_no_fire_patches, len(self.no_fire_samples)), 
                                             replace=True)
            no_fire_batch = [self.no_fire_samples[i] for i in no_fire_indices]
        
        # Combine and shuffle
        batch_samples = fire_batch + no_fire_batch
        # Fill remaining slots if needed
        while len(batch_samples) < self.batch_size:
            if len(self.samples) > 0:
                batch_samples.append(np.random.choice(self.samples))
        
        # Trim to exact batch size and shuffle
        batch_samples = batch_samples[:self.batch_size]
        np.random.shuffle(batch_samples)
        
        X, Y = [], []
        for sample in batch_samples:
            try:
                # Load patch
                with rasterio.open(sample['tif_path']) as src:
                    # Ensure window coordinates are integers
                    x, y = int(sample['x']), int(sample['y'])
                    patch = src.read(
                        window=rasterio.windows.Window(
                            x, y, self.patch_size, self.patch_size
                        ),
                        boundless=True, fill_value=0
                    ).astype(np.float32)
                
                # Clean data and ensure proper shape
                patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
                patch = np.moveaxis(patch, 0, -1)  # (H, W, C)
                
                # Ensure patch has correct shape
                if patch.shape[2] != 10:
                    print(f"⚠️ Unexpected patch shape: {patch.shape}")
                    continue
                
                # Separate features and target
                img = normalize_patch(patch[:, :, :9])  # First 9 bands -> becomes 12 after LULC encoding
                mask = (patch[:, :, 9] > 0).astype(np.float32)  # Fire mask
                mask = np.expand_dims(mask, -1)
                
                # Ensure arrays are contiguous and proper dtype
                img = np.ascontiguousarray(img, dtype=np.float32)
                mask = np.ascontiguousarray(mask, dtype=np.float32)
                
                # Apply augmentation only if enabled and not in debug mode
                if self.augment_fn:
                    try:
                        # Ensure arrays are in correct format for albumentations
                        augmented = self.augment_fn(image=img, mask=mask)
                        img, mask = augmented['image'], augmented['mask']
                    except Exception as aug_error:
                        print(f"⚠️ Augmentation error: {aug_error}")
                        # Skip augmentation on error
                        pass
                
                X.append(img)
                Y.append(mask)
                
            except Exception as e:
                print(f"⚠️ Error loading patch: {e}")
                # Create dummy patch with correct shape (12 channels after LULC encoding)
                X.append(np.zeros((self.patch_size, self.patch_size, 12), dtype=np.float32))
                Y.append(np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32))
        
        return np.array(X), np.array(Y)

    def on_epoch_end(self):
        """Shuffle samples at end of epoch"""
        if self.shuffle:
            np.random.shuffle(self.samples)
