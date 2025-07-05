# ====================
# 7. evaluate.py
# ====================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef, focal_loss
import seaborn as sns

def evaluate_model(model_path, test_files, output_dir='outputs'):
    """Evaluate trained model on test data"""
    
    # Load model with ALL custom objects
    custom_objects = {
        'focal_loss': focal_loss(),  # ✅ Added missing focal_loss
        'iou_score': iou_score,
        'dice_coef': dice_coef
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model with custom objects: {str(e)}")
        try:
            # Fallback: load without compilation
            model = tf.keras.models.load_model(model_path, compile=False)
            print("⚠️ Model loaded without compilation (metrics disabled)")
        except Exception as e2:
            print(f"❌ Failed to load model completely: {str(e2)}")
            raise e2
    
    # Create test generator
    try:
        test_gen = FireDatasetGenerator(
            test_files,
            patch_size=256,
            batch_size=8,
            n_patches_per_img=20,
            fire_focus_ratio=0.8,
            augment=False,
            shuffle=False
        )
        print(f"✅ Test generator created with {len(test_gen)} batches")
    except Exception as e:
        print(f"❌ Failed to create test generator: {str(e)}")
        raise e
    
    # Evaluate
    print("📊 Evaluating model...")
    try:
        results = model.evaluate(test_gen, verbose=1)
        print(f"✅ Evaluation completed")
        
        # Print results with metric names
        if hasattr(model, 'metrics_names'):
            for name, value in zip(model.metrics_names, results):
                print(f"  {name}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        results = None
    
    # Generate predictions for visualization
    print("🔮 Generating predictions...")
    try:
        predictions = model.predict(test_gen, verbose=1)
        print(f"✅ Predictions generated: {predictions.shape}")
        
        # Get ground truth
        print("📋 Collecting ground truth...")
        y_true = []
        for i in range(len(test_gen)):
            _, masks = test_gen[i]  # ✅ Fixed syntax error (test*gen -> test_gen)
            y_true.append(masks)
        y_true = np.concatenate(y_true, axis=0)
        print(f"✅ Ground truth collected: {y_true.shape}")
        
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

def calculate_additional_metrics(y_true, y_pred, threshold=0.5):
    """Calculate additional evaluation metrics"""
    
    # Convert to binary predictions
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    
    # Calculate metrics
    tp = np.sum(y_true * y_pred_binary)
    fp = np.sum((1 - y_true) * y_pred_binary)
    fn = np.sum(y_true * (1 - y_pred_binary))
    tn = np.sum((1 - y_true) * (1 - y_pred_binary))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }
    
    return metrics
