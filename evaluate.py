import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from dataset.loader import FireDatasetGenerator
# from utils.metrics import iou_score, dice_coef, focal_loss

import tensorflow as tf
import keras
import keras.backend as K

@keras.saving.register_keras_serializable()
def iou_score(y_true, y_pred, threshold=0.05, smooth=1e-6):
    """Intersection over Union metric for binary segmentation with configurable threshold"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, threshold=0.05, smooth=1e-6):
    """Dice coefficient for binary segmentation with configurable threshold"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@keras.saving.register_keras_serializable()
def fire_recall(y_true, y_pred, threshold=0.05, smooth=1e-6):
    """Fire-specific recall metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

@keras.saving.register_keras_serializable()
def fire_precision(y_true, y_pred, threshold=0.05, smooth=1e-6):
    """Fire-specific precision metric"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return (true_positives + smooth) / (predicted_positives + smooth)

@keras.saving.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.6):
    """Focal loss for handling class imbalance - Updated alpha to match training"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fixed

# Custom objects for loading model
custom_objects = {
        "focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.6),  # Updated parameters to match training      
        'iou_score': iou_score,
        'dice_coef': dice_coef,
        'fire_recall': fire_recall,
        'fire_precision': fire_precision,
    }

def evaluate_model(model_path, test_files, output_dir='outputs'):
    """Evaluate trained model on test data"""
    
    # Load model safely
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Create test generator
    try:
        test_gen = FireDatasetGenerator(
            test_files,
            patch_size=256,
            batch_size=8,
            n_patches_per_img=20,
            fire_focus_ratio=0.9,  # Match training configuration
            fire_patch_ratio=0.2,  # Add stratified sampling
            augment=False,
            shuffle=False
        )
        print(f"✅ Test generator created with {len(test_gen)} batches")
    except Exception as e:
        print(f"❌ Failed to create test generator: {str(e)}")
        raise e
    
    # Evaluate - handle case where model wasn't compiled
    print("📊 Evaluating model...")
    results = None
    
    try:
        if hasattr(model, 'compiled_loss') and model.compiled_loss is not None:
            # Model is compiled, can use evaluate
            results = model.evaluate(test_gen, verbose=1)
            print(f"✅ Evaluation completed")
            
            # Print results with metric names
            if hasattr(model, 'metrics_names'):
                for name, value in zip(model.metrics_names, results):
                    print(f"  {name}: {value:.4f}")
        else:
            print("⚠️ Model not compiled, skipping built-in evaluation")
            
    except Exception as e:
        print(f"❌ Built-in evaluation failed: {str(e)}")
        print("⚠️ Continuing with manual evaluation...")
    
    # Generate predictions for visualization and manual metrics
    print("🔮 Generating predictions...")
    try:
        predictions = model.predict(test_gen, verbose=1)
        print(f"✅ Predictions generated: {predictions.shape}")
        
        # Get ground truth
        print("📋 Collecting ground truth...")
        y_true = []
        for i in range(len(test_gen)):
            _, masks = test_gen[i]
            y_true.append(masks)
        y_true = np.concatenate(y_true, axis=0)
        print(f"✅ Ground truth collected: {y_true.shape}")
        
        # Calculate manual metrics with multiple thresholds
        print("📊 Calculating manual metrics at multiple thresholds...")
        thresholds = [0.1, 0.3, 0.5]
        
        for threshold in thresholds:
            manual_metrics = calculate_additional_metrics(y_true, predictions, threshold=threshold)
            print(f"\n📈 Manual Metrics (threshold={threshold}):")
            for name, value in manual_metrics.items():
                if name not in ['true_positives', 'false_positives', 'false_negatives', 'true_negatives']:
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {int(value)}")
        
        # Print prediction statistics for debugging
        print(f"\n📊 Prediction Statistics:")
        print(f"  Min: {predictions.min():.6f}")
        print(f"  Max: {predictions.max():.6f}")
        print(f"  Mean: {predictions.mean():.6f}")
        print(f"  Std: {predictions.std():.6f}")
        for threshold in [0.05, 0.1, 0.3, 0.5]:
            count = (predictions > threshold).sum()
            print(f"  Pixels > {threshold}: {count:,} ({count/predictions.size*100:.2f}%)")
        
        # Ensure output directory exists
        os.makedirs(f'{output_dir}/plots', exist_ok=True)
        
        # Visualize results
        visualize_predictions(y_true, predictions, output_dir)
        
    except Exception as e:
        print(f"❌ Prediction generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results

def visualize_predictions(y_true, y_pred, output_dir, n_samples=8):
    """Visualize model predictions"""
    
    print(f"📊 Creating visualization with {n_samples} samples...")
    
    # Ensure we don't exceed available samples
    n_samples = min(n_samples, len(y_true))
    
    fig, axes = plt.subplots(3, n_samples, figsize=(20, 8))
    
    # Handle case where n_samples = 1 (axes won't be 2D)
    if n_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(n_samples):
        if i < len(y_true):
            # Ground truth
            axes[0, i].imshow(y_true[i, :, :, 0], cmap='Reds', vmin=0, vmax=1)
            axes[0, i].set_title('Ground Truth')
            axes[0, i].axis('off')
            
            # Prediction
            axes[1, i].imshow(y_pred[i, :, :, 0], cmap='Reds', vmin=0, vmax=1)
            axes[1, i].set_title('Prediction')
            axes[1, i].axis('off')
            
            # Difference
            diff = np.abs(y_true[i, :, :, 0] - y_pred[i, :, :, 0])
            axes[2, i].imshow(diff, cmap='Blues', vmin=0, vmax=1)
            axes[2, i].set_title('Difference')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_path = f'{output_dir}/plots/predictions_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    plt.show()

def calculate_additional_metrics(y_true, y_pred, threshold=0.05):
    """Calculate additional evaluation metrics"""
    
    # Convert to binary predictions
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    
    # Calculate pixel-wise metrics
    tp = np.sum(y_true * y_pred_binary)
    fp = np.sum((1 - y_true) * y_pred_binary)
    fn = np.sum(y_true * (1 - y_pred_binary))
    tn = np.sum((1 - y_true) * (1 - y_pred_binary))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate IoU and Dice manually
    intersection = np.sum(y_true * y_pred_binary)
    union = np.sum(y_true) + np.sum(y_pred_binary) - intersection
    iou = intersection / union if union > 0 else 0
    
    dice = (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_binary)) if (np.sum(y_true) + np.sum(y_pred_binary)) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou_manual': iou,
        'dice_manual': dice,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }
    
    return metrics

# Test function to run evaluation
if __name__ == "__main__":
    # Example usage
    model_path = "/kaggle/working/forest_fire_ml/outputs/final_model.keras"
    test_files = []  # Add your test files here
    
    if test_files:
        results = evaluate_model(model_path, test_files)
        print("🎉 Evaluation completed!")
    else:
        print("⚠️ No test files specified. Please add test file paths.")
