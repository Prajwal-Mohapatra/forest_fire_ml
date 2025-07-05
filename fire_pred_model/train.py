# 6. train_optimized.py
# ====================
import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import datetime
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef, focal_loss
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Custom callback for monitoring
class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_metrics = []
        self.val_metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Log GPU memory usage
        if tf.config.experimental.list_physical_devices('GPU'):
            try:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                used_mb = mem_info['current'] / (1024**2)
                print(f"GPU Memory Used: {used_mb:.0f} MB")
            except:
                pass
        
        # Store metrics
        self.train_metrics.append({
            'epoch': epoch,
            'loss': logs.get('loss'),
            'iou': logs.get('iou_score'),
            'dice': logs.get('dice_coef')
        })
        
        self.val_metrics.append({
            'epoch': epoch,
            'val_loss': logs.get('val_loss'),
            'val_iou': logs.get('val_iou_score'),
            'val_dice': logs.get('val_dice_coef')
        })

def create_datasets(base_dir):
    """Create train/val/test splits with temporal awareness"""
    all_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_*.tif')))
    
    if not all_files:
        raise ValueError(f"No files found in {base_dir}")
    
    print(f"Found {len(all_files)} files")
    
    # Temporal split: April for training, early May for validation, late May for testing
    train_files = [f for f in all_files if '2016_04_' in f]
    val_files = [f for f in all_files if '2016_05_' in f and int(f.split('_')[-1].split('.')[0]) <= 15]
    test_files = [f for f in all_files if '2016_05_' in f and int(f.split('_')[-1].split('.')[0]) > 15]
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    return train_files, val_files, test_files

def main():
    # Configuration
    CONFIG = {
        'patch_size': 256,
        'batch_size': 8,
        'n_patches_per_img': 30,
        'epochs': 50,
        'learning_rate': 1e-4,
        'fire_focus_ratio': 0.8,
    }
    
    # Paths
    base_dir = '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked'
    output_dir = 'outputs'
    
    # Create output directories
    os.makedirs(f'{output_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{output_dir}/logs', exist_ok=True)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Create datasets
    train_files, val_files, test_files = create_datasets(base_dir)
    
    # Data generators
    train_gen = FireDatasetGenerator(
        train_files, 
        patch_size=CONFIG['patch_size'],
        batch_size=CONFIG['batch_size'],
        n_patches_per_img=CONFIG['n_patches_per_img'],
        fire_focus_ratio=CONFIG['fire_focus_ratio'],
        augment=True
    )
    
    val_gen = FireDatasetGenerator(
        val_files,
        patch_size=CONFIG['patch_size'],
        batch_size=CONFIG['batch_size'],
        n_patches_per_img=CONFIG['n_patches_per_img'] // 2,
        fire_focus_ratio=CONFIG['fire_focus_ratio'],
        augment=False
    )
    
    # Build model
    model = build_resunet_a(input_shape=(CONFIG['patch_size'], CONFIG['patch_size'], 9))
    
    # Compile with focal loss for better class imbalance handling
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[iou_score, dice_coef]
    )
    
    print("Model compiled successfully!")
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'{output_dir}/checkpoints/best_model.h5',
            monitor='val_iou_score',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(f'{output_dir}/logs/training_log.csv'),
        TrainingMonitor()
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Save final model
    model.save(f'{output_dir}/final_model.h5')
    print(f"Training completed! Model saved to {output_dir}/final_model.h5")

def plot_training_history(history, output_dir):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU Score
    axes[0, 1].plot(history.history['iou_score'], label='Train IoU')
    axes[0, 1].plot(history.history['val_iou_score'], label='Val IoU')
    axes[0, 1].set_title('IoU Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice Coefficient
    axes[1, 0].plot(history.history['dice_coef'], label='Train Dice')
    axes[1, 0].plot(history.history['val_dice_coef'], label='Val Dice')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()