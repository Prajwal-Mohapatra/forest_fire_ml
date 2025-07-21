import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from datetime import datetime
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef, focal_loss, fire_recall, fire_precision
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, LearningRateScheduler

from sklearn.model_selection import train_test_split


# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Custom callback for monitoring
class TrainingMonitor(keras.callbacks.Callback):
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
        
    # # Old Temporal split: April (1-30) for training, early May (1-15) for validation, late May (16-29) for testing
    # train_files = [f for f in all_files if '2016_04_' in f]
    # val_files = [f for f in all_files if '2016_05_' in f and int(f.split('_')[-1].split('.')[0]) <= 15]
    # test_files = [f for f in all_files if '2016_05_' in f and int(f.split('_')[-1].split('.')[0]) > 15]

    # New Temporal split: Early April (1-20) for training, late April for validation, full May for testing
    train_files = [f for f in all_files if '2016_04_' in f and int(f.split('_')[-1].split('.')[0]) <= 20]
    val_files = [f for f in all_files if '2016_04_' in f and int(f.split('_')[-1].split('.')[0]) > 20]
    test_files = [f for f in all_files if '2016_05_' in f]

    # # Random split: 80% training + validation, 20% testing; 80% -> 80% training & 20% validation
    # train_val_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    # train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    return train_files, val_files, test_files

def warmup_scheduler(epoch, lr, CONFIG):
    """Learning rate warmup scheduler"""
    if epoch < CONFIG['warmup_epochs']:
        # Linear warmup from initial LR to max LR
        return CONFIG['learning_rate'] + (CONFIG['max_learning_rate'] - CONFIG['learning_rate']) * (epoch / CONFIG['warmup_epochs'])
    else:
        return lr  # Use normal scheduler after warmup

def main():
    # Configuration - Optimized for improved fire detection with aggressive class imbalance handling
    CONFIG = {
        'patch_size': 256,
        'batch_size': 8,
        'n_patches_per_img': 30,      # Increased from 25 for more diversity
        'epochs': 20,                 # Increased from 10 for better learning
        'learning_rate': 1e-5,        # Start with lower LR for warmup
        'max_learning_rate': 1e-4,    # Target LR after warmup
        'warmup_epochs': 5,           # Warmup epochs for LR scheduler
        'fire_focus_ratio': 0.9,      # Increased from 0.8 - even more fire examples
        'fire_patch_ratio': 0.3,      # Increased from 0.2 - more fire-positive patches per batch
        'focal_gamma': 2.0,           # Hard example focus
        'focal_alpha': 0.6,           # Reduced from 0.75 - better precision/recall balance
        'dropout_rate': 0.2,          # Increased from 0.1 for better regularization
        'weight_decay': 1e-5,         # L2 regularization

        # Enhanced Callback Configuration
        'patience': 10,               # Increased patience from 8 for longer training
        'factor': 0.5,                # Learning rate reduction factor
        'min_lr': 1e-7,               # Minimum learning rate
        'monitor_metric': 'val_dice_coef',  # Changed from val_fire_recall to balance precision/recall

        'debug_mode': False,          # Set to True to disable all augmentation
    }
    
    print("🔥 Starting Fire Prediction Model Training...")
    print(f"Configuration: {CONFIG}")
    
    # Paths
    base_dir = '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked'
    output_dir = '/kaggle/working/forest_fire_ml/outputs'
    
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
        fire_patch_ratio=CONFIG['fire_patch_ratio'],  # Add stratified sampling
        augment=not CONFIG['debug_mode']  # Disable augmentation in debug mode
    )
    
    val_gen = FireDatasetGenerator(
        val_files,
        patch_size=CONFIG['patch_size'],
        batch_size=CONFIG['batch_size'],
        n_patches_per_img=CONFIG['n_patches_per_img'] // 2,
        fire_focus_ratio=CONFIG['fire_focus_ratio'],
        fire_patch_ratio=CONFIG['fire_patch_ratio'],  # Add stratified sampling for validation
        augment=False  # Never augment validation data
    )
    
    # Build model with enhanced dropout and L2 regularization
    model = build_resunet_a(
        input_shape=(CONFIG['patch_size'], CONFIG['patch_size'], 12),  # 12 channels after LULC encoding
        dropout_rate=CONFIG['dropout_rate'],
        weight_decay=CONFIG['weight_decay']  # Pass L2 regularization parameter
    )
    
    # Compile with focal loss and enhanced metrics, L2 regularization through optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']  # Add L2 regularization
    )
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=CONFIG['focal_gamma'], alpha=CONFIG['focal_alpha']),
        metrics=[iou_score, dice_coef, fire_recall, fire_precision]
    )
    
    print("Model compiled successfully!")
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'{output_dir}/checkpoints/best_model.keras',
            monitor=CONFIG['monitor_metric'],  # Now monitoring val_dice_coef for better balance
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor=CONFIG['monitor_metric'],  # Use same metric as ModelCheckpoint
            patience=CONFIG['patience'],  # Increased patience for 20-epoch training
            mode='max',  # Added mode='max' since val_dice_coef should be maximized
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=CONFIG['monitor_metric'],  # Monitor val_dice_coef instead of val_loss
            factor=CONFIG['factor'],
            patience=CONFIG['patience'] // 2,  # Reduce LR more quickly
            mode='max',  # Added mode='max' since val_dice_coef should be maximized
            min_lr=CONFIG['min_lr'],
            verbose=1
        ),
        LearningRateScheduler(
            lambda epoch, lr: warmup_scheduler(epoch, lr, CONFIG),
            verbose=1
        ),
        CSVLogger(f'{output_dir}/logs/training_log.csv'),
        TensorBoard(
            log_dir=f'{output_dir}/logs/tensorboard',
            histogram_freq=1,
            write_graph=True,
            write_images=False,  # Disable image logging to save space
            update_freq='epoch'
        ),
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
    model.save(f'{output_dir}/final_model.keras', save_format="keras_v3") # model.export = bad, saves to a directory structured, SavedModel format
    print(f"Training completed! Model saved to {output_dir}/final_model.keras")

def plot_training_history(history, output_dir):
    """Plot and save enhanced training history with fire-specific metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
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
    axes[1, 0].set_title('Dice Coefficient (Monitoring Metric)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1-Score (harmonic mean of precision and recall)
    if 'fire_recall' in history.history and 'fire_precision' in history.history:
        train_f1 = []
        val_f1 = []
        
        for i in range(len(history.history['fire_recall'])):
            train_prec = history.history['fire_precision'][i]
            train_rec = history.history['fire_recall'][i]
            val_prec = history.history['val_fire_precision'][i] if 'val_fire_precision' in history.history else 0
            val_rec = history.history['val_fire_recall'][i] if 'val_fire_recall' in history.history else 0
            
            # Calculate F1-score
            train_f1_val = 2 * train_prec * train_rec / (train_prec + train_rec) if (train_prec + train_rec) > 0 else 0
            val_f1_val = 2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0
            
            train_f1.append(train_f1_val)
            val_f1.append(val_f1_val)
        
        axes[1, 1].plot(train_f1, label='Train F1-Score', color='green')
        axes[1, 1].plot(val_f1, label='Val F1-Score', color='darkgreen')
        axes[1, 1].set_title('F1-Score (Precision-Recall Balance)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'F1-Score\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('F1-Score')
    
    # Fire Recall
    if 'fire_recall' in history.history:
        axes[2, 0].plot(history.history['fire_recall'], label='Train Fire Recall', color='red', linewidth=2)
        axes[2, 0].plot(history.history['val_fire_recall'], label='Val Fire Recall', color='darkred', linewidth=2)
        axes[2, 0].set_title('Fire Recall')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Recall')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
    else:
        axes[2, 0].text(0.5, 0.5, 'Fire Recall\nNot Available', ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Fire Recall')
    
    # Learning Rate
    if 'lr' in history.history:
        axes[2, 1].plot(history.history['lr'], label='Learning Rate', color='green')
        axes[2, 1].set_title('Learning Rate (with Warmup)')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('LR')
        axes[2, 1].set_yscale('log')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
    else:
        axes[2, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/training_history.png', dpi=300, bbox_inches='tight')
    
    # Also create a focused plot on fire-specific metrics with F1-score
    if 'fire_recall' in history.history and 'fire_precision' in history.history:
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.plot(history.history['fire_recall'], label='Train Fire Recall', color='red', linewidth=2)
        ax.plot(history.history['val_fire_recall'], label='Val Fire Recall', color='darkred', linewidth=2)
        ax.plot(history.history['fire_precision'], label='Train Fire Precision', color='orange', linewidth=2)
        ax.plot(history.history['val_fire_precision'], label='Val Fire Precision', color='darkorange', linewidth=2)
        
        # Add F1-score
        train_f1 = []
        val_f1 = []
        for i in range(len(history.history['fire_recall'])):
            train_prec = history.history['fire_precision'][i]
            train_rec = history.history['fire_recall'][i]
            val_prec = history.history['val_fire_precision'][i] if 'val_fire_precision' in history.history else 0
            val_rec = history.history['val_fire_recall'][i] if 'val_fire_recall' in history.history else 0
            
            train_f1_val = 2 * train_prec * train_rec / (train_prec + train_rec) if (train_prec + train_rec) > 0 else 0
            val_f1_val = 2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0
            
            train_f1.append(train_f1_val)
            val_f1.append(val_f1_val)
        
        ax.plot(train_f1, label='Train F1-Score', color='green', linewidth=2, linestyle='--')
        ax.plot(val_f1, label='Val F1-Score', color='darkgreen', linewidth=2, linestyle='--')
        
        ax.set_title('Fire Detection Metrics (Recall, Precision & F1-Score)', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/fire_metrics_focus.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()
