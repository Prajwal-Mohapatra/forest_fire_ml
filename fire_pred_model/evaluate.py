# ====================
# 7. evaluate.py
# ====================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef
import seaborn as sns

def evaluate_model(model_path, test_files, output_dir='outputs'):
    """Evaluate trained model on test data"""
    
    # Load model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'iou_score': iou_score,
            'dice_coef': dice_coef
        }
    )
    
    # Create test generator
    test_gen = FireDatasetGenerator(
        test_files,
        patch_size=256,
        batch_size=8,
        n_patches_per_img=20,
        fire_focus_ratio=0.8,
        augment=False,
        shuffle=False
    )
    
    # Evaluate
    print("Evaluating model...")
    results = model.evaluate(test_gen, verbose=1)
    
    # Generate predictions for visualization
    print("Generating predictions...")
    predictions = model.predict(test_gen, verbose=1)
    
    # Get ground truth
    y_true = []
    for i in range(len(test_gen)):
        _, masks = test_gen[i]
        y_true.append(masks)
    y_true = np.concatenate(y_true, axis=0)
    
    # Visualize results
    visualize_predictions(y_true, predictions, output_dir)
    
    return results

def visualize_predictions(y_true, y_pred, output_dir, n_samples=8):
    """Visualize model predictions"""
    
    fig, axes = plt.subplots(3, n_samples, figsize=(20, 8))
    
    for i in range(n_samples):
        if i < len(y_true):
            # Ground truth
            axes[0, i].imshow(y_true[i, :, :, 0], cmap='Reds')
            axes[0, i].set_title('Ground Truth')
            axes[0, i].axis('off')
            
            # Prediction
            axes[1, i].imshow(y_pred[i, :, :, 0], cmap='Reds')
            axes[1, i].set_title('Prediction')
            axes[1, i].axis('off')
            
            # Difference
            diff = np.abs(y_true[i, :, :, 0] - y_pred[i, :, :, 0])
            axes[2, i].imshow(diff, cmap='Blues')
            axes[2, i].set_title('Difference')
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()