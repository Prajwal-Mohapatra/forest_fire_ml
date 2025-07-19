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
                 shuffle=True, augment=True, fire_focus_ratio=0.7, **kwargs):
        super().__init__(**kwargs)
        
        self.tif_paths = sorted(tif_paths)  # Ensure chronological order
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_patches_per_img = n_patches_per_img
        self.shuffle = shuffle
        self.fire_focus_ratio = fire_focus_ratio
        
        # Setup augmentation pipeline - Multi-channel compatible (9-band satellite data)
        self.augment_fn = None
        if augment:
            self.augment_fn = A.Compose([
                # Geometric augmentations (work with any number of channels)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(shear=(-10, 10), rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.3),
                
                # Photometric augmentations (multi-channel compatible)
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # Fire thermal variations
                # Removed: HueSaturationValue (only works with 1 or 3 channels)
                
                # Noise and blur (work with any number of channels)
                A.GaussNoise(var_limit=50, p=0.2),
                A.GaussianBlur(blur_limit=(1, 3), p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
                
                # Weather simulation (removed - RGB only)
                # Removed: RandomFog, RandomShadow (designed for RGB images)
            ])
        
        # Pre-compute patch coordinates for each day
        self.samples = self._generate_temporal_patches()
        self.on_epoch_end()
        
        print(f"✅ Dataset loaded successfully! {len(self.samples)} patches from {len(self.tif_paths)} days.")

    def _generate_temporal_patches(self):
        """Generate patches with temporal awareness and fire focus"""
        all_samples = []
        
        for day_idx, tif_path in enumerate(self.tif_paths):
            try:
                with rasterio.open(tif_path) as src:
                    # Read fire mask to identify fire-prone areas
                    fire_mask = src.read(10)  # Band 10 is fire mask
                    
                    # Get fire-focused coordinates
                    coords = get_fire_focused_coordinates(
                        fire_mask, self.patch_size, self.n_patches_per_img, self.fire_focus_ratio
                    )
                    
                    # Add temporal context
                    for x, y in coords:
                        all_samples.append({
                            'tif_path': tif_path,
                            'day_idx': day_idx,
                            'x': x,
                            'y': y,
                            'fire_density': np.mean(fire_mask[y:y+self.patch_size, x:x+self.patch_size])
                        })
                        
            except Exception as e:
                print(f"⚠️ Error processing {tif_path}: {e}")
                continue
        
        # Sort by fire density to prioritize fire-rich patches
        all_samples.sort(key=lambda x: x['fire_density'], reverse=True)
        return all_samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X, Y = [], []
        for sample in batch_samples:
            try:
                # Load patch
                with rasterio.open(sample['tif_path']) as src:
                    patch = src.read(
                        window=rasterio.windows.Window(
                            sample['x'], sample['y'], self.patch_size, self.patch_size
                        ),
                        boundless=True, fill_value=0
                    ).astype(np.float32)
                
                # Clean data
                patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
                patch = np.moveaxis(patch, 0, -1)  # (H, W, C)
                
                # Separate features and target
                img = normalize_patch(patch[:, :, :9])  # First 9 bands
                mask = (patch[:, :, 9] > 0).astype(np.float32)  # Fire mask
                mask = np.expand_dims(mask, -1)
                
                # Apply augmentation
                if self.augment_fn:
                    augmented = self.augment_fn(image=img, mask=mask)
                    img, mask = augmented['image'], augmented['mask']
                
                X.append(img)
                Y.append(mask)
                
            except Exception as e:
                print(f"⚠️ Error loading patch: {e}")
                # Create dummy patch
                X.append(np.zeros((self.patch_size, self.patch_size, 9), dtype=np.float32))
                Y.append(np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32))
        
        return np.array(X), np.array(Y)

    def on_epoch_end(self):
        """Shuffle samples at end of epoch"""
        if self.shuffle:
            np.random.shuffle(self.samples)
