# ====================
# 7. evaluate.py (FIXED)
# ====================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef, focal_loss
import seaborn as sns

def load_model_safe(model_path):
    """Safely load model with proper custom objects handling"""
    
    # Define all possible custom objects the model might need
    custom_objects = {
        # Loss function (the one that works)
        'focal_loss_fixed': focal_loss(),
        
        # Metrics - try different possible names
        'iou_score': iou_score,
        'dice_coef': dice_coef,
        
        # Alternative names that might have been saved
        'iou_score_1': iou_score,
        'dice_coef_1': dice_coef,
        'iou_score_2': iou_score,
        'dice_coef_2': dice_coef,
        
        # Also include the factory function
        'focal_loss': focal_loss,
    }
    
    print(f"🔄 Attempting to load model from: {model_path}")
    
    # Try different loading strategies
    strategies = [
        # Strategy 1: Load with focal_loss_fixed only (we know this works)
        {
            'focal_loss_fixed': focal_loss(),
        },
        
        # Strategy 2: Load without compilation (fallback)
        None,
        
        # Strategy 3: Load with all custom objects
        custom_objects,
    ]
    
    model = None
    for i, strategy in enumerate(strategies, 1):
        try:
            if strategy is None:
                # Load without compilation
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"✅ Strategy {i}: Loaded without compilation")
                break
            else:
                # Load with custom objects
                model = tf.keras.models.load_model(model_path, custom_objects=strategy)
                print(f"✅ Strategy {i}: Loaded with custom objects")
                break
                
        except Exception as e:
            print(f"❌ Strategy {i} failed: {str(e)}")
            continue
    
    if model is None:
        raise Exception("Failed to load model with any strategy")
    
    return model

def evaluate_model(model_path, test_files, output_dir='outputs'):
    """Evaluate trained model on test data"""
    
    # Load model safely
    model = load_model_safe(model_path)
    
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
        
        # Calculate manual metrics
        print("📊 Calculating manual metrics...")
        manual_metrics = calculate_additional_metrics(y_true, predictions)
        
        print("\n📈 Manual Metrics:")
        for name, value in manual_metrics.items():
            print(f"  {name}: {value:.4f}")
        
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
    model_path = "/kaggle/working/forest_fire_ml/outputs/final_model.h5"
    test_files = []  # Add your test files here
    
    if test_files:
        results = evaluate_model(model_path, test_files)
        print("🎉 Evaluation completed!")
    else:
        print("⚠️ No test files specified. Please add test file paths.")
