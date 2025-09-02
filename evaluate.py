import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from dataset.loader import FireDatasetGenerator

# Configure TensorFlow to manage GPU memory growth dynamically
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

@keras.saving.register_keras_serializable()
def iou_score(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Intersection over Union metric for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Dice coefficient for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@keras.saving.register_keras_serializable()
def fire_recall(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Fire-specific recall metric."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    possible_positives = tf.reduce_sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

@keras.saving.register_keras_serializable()
def fire_precision(y_true, y_pred, threshold=0.4, smooth=1e-6):
    """Fire-specific precision metric."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    return (true_positives + smooth) / (predicted_positives + smooth)

@keras.saving.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.6):
    """Focal loss for handling class imbalance."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss_val = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss_val)
    return focal_loss_fixed

def evaluate_model(model_path, test_files, output_dir='outputs'):
    """Loads a trained model and evaluates its performance on the test dataset."""
    
    custom_objects = {
        "focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.6),
        'iou_score': iou_score,
        'dice_coef': dice_coef,
        'fire_recall': fire_recall,
        'fire_precision': fire_precision,
    }

    print(f"🔄 Loading model from: {model_path}")
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    try:
        test_gen = FireDatasetGenerator(
            test_files,
            patch_size=256,
            batch_size=8,
            n_patches_per_img=20,
            fire_focus_ratio=0.9,
            fire_patch_ratio=0.2,
            augment=False,
            shuffle=False
        )
        print(f"✅ Test generator created with {len(test_gen)} batches.")
    except Exception as e:
        print(f"❌ Failed to create test generator: {str(e)}")
        raise

    print("📊 Evaluating model...")
    results = None
    try:
        if hasattr(model, 'compiled_loss') and model.compiled_loss is not None:
            results = model.evaluate(test_gen, verbose=1)
            print("✅ Built-in evaluation completed.")
            if hasattr(model, 'metrics_names'):
                for name, value in zip(model.metrics_names, results):
                    print(f"  {name}: {value:.4f}")
        else:
            print("⚠️ Model not compiled, skipping built-in Keras evaluation.")
    except Exception as e:
        print(f"❌ Built-in evaluation failed: {str(e)}. Proceeding with manual evaluation.")

    print("🔮 Generating predictions for manual metrics and visualization...")
    try:
        predictions = model.predict(test_gen, verbose=1)
        
        print("📋 Collecting ground truth masks...")
        y_true = np.concatenate([test_gen[i][1] for i in range(len(test_gen))], axis=0)
        
        print("📊 Calculating manual metrics at multiple thresholds...")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        for threshold in thresholds:
            manual_metrics = calculate_additional_metrics(y_true, predictions, threshold=threshold)
            print(f"\nManual Metrics (threshold={threshold}):")
            for name, value in manual_metrics.items():
                print(f"  {name}: {value:.4f}" if not name.startswith(('true_', 'false_')) else f"  {name}: {int(value)}")

        print("\n📊 Prediction Statistics:")
        print(f"  Min: {predictions.min():.6f}, Max: {predictions.max():.6f}, Mean: {predictions.mean():.6f}")

        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        visualize_predictions(y_true, predictions, output_dir)
        
    except Exception as e:
        import traceback
        print(f"❌ Prediction or manual evaluation failed: {str(e)}")
        traceback.print_exc()
    
    tf.keras.backend.clear_session()
    return results

def visualize_predictions(y_true, y_pred, output_dir, n_samples=8):
    """Saves a plot comparing ground truth, predictions, and their difference."""
    print(f"Creating prediction visualization for {n_samples} samples...")
    n_samples = min(n_samples, len(y_true))
    if n_samples == 0:
        print("⚠️ No samples to visualize.")
        return

    fig, axes = plt.subplots(3, n_samples, figsize=(20, 8))
    axes = axes.reshape(3, n_samples) if n_samples > 1 else axes.reshape(3, 1)

    for i in range(n_samples):
        axes[0, i].imshow(y_true[i, :, :, 0], cmap='Reds', vmin=0, vmax=1)
        axes[0, i].set_title('Ground Truth')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(y_pred[i, ..., 0], cmap='Reds', vmin=0, vmax=1)
        axes[1, i].set_title('Prediction')
        axes[1, i].axis('off')
        
        diff = np.abs(y_true[i, ..., 0] - y_pred[i, ..., 0])
        axes[2, i].imshow(diff, cmap='Blues', vmin=0, vmax=1)
        axes[2, i].set_title('Difference')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'plots', 'predictions_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    # plt.show() # Commented out for non-interactive script execution

def calculate_additional_metrics(y_true, y_pred, threshold=0.4):
    """Calculates pixel-wise precision, recall, F1, IoU, and Dice score."""
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    
    tp = np.sum(y_true * y_pred_binary)
    fp = np.sum((1 - y_true) * y_pred_binary)
    fn = np.sum(y_true * (1 - y_pred_binary))
    tn = np.sum((1 - y_true) * (1 - y_pred_binary))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    intersection = np.sum(y_true * y_pred_binary)
    union = np.sum(y_true) + np.sum(y_pred_binary) - intersection
    iou = intersection / union if union > 0 else 0.0
    
    dice_numerator = 2 * intersection
    dice_denominator = np.sum(y_true) + np.sum(y_pred_binary)
    dice = dice_numerator / dice_denominator if dice_denominator > 0 else 0.0
    
    return {
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'iou_manual': iou, 'dice_manual': dice,
        'true_positives': tp, 'false_positives': fp,
        'false_negatives': fn, 'true_negatives': tn
    }

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Define the path to your trained model
    MODEL_PATH = "final_model.h5"
    
    # IMPORTANT: Add paths to your test image files here.
    # For example: TEST_FILES = ['/path/to/image1.tif', '/path/to/image2.tif']
    TEST_FILES = []
    
    # --- EXECUTION ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'")
    elif not TEST_FILES:
        print("⚠️ Warning: The 'TEST_FILES' list is empty. Please add file paths to evaluate the model.")
    else:
        print("Starting model evaluation...")
        evaluate_model(MODEL_PATH, TEST_FILES)
        print("🎉 Evaluation script finished!")
