# ====================
# 9. utils/visualization.py
# ====================
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap

def create_fire_colormap():
    """Create custom colormap for fire probability visualization"""
    colors = ['#000033', '#000055', '#000077', '#003399', '#0066CC', 
              '#3399FF', '#66CCFF', '#99FFFF', '#FFFF99', '#FFCC66', 
              '#FF9933', '#FF6600', '#FF3300', '#CC0000', '#990000']
    return LinearSegmentedColormap.from_list('fire_prob', colors)

def visualize_fire_prediction(prediction_path, output_path=None):
    """Visualize fire probability prediction"""
    
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot prediction with custom colormap
    im = ax.imshow(prediction, cmap=create_fire_colormap(), vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Fire Probability', rotation=270, labelpad=15)
    
    # Styling
    ax.set_title('Fire Probability Prediction Map', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude (pixels)', fontsize=12)
    ax.set_ylabel('Latitude (pixels)', fontsize=12)
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def compare_predictions(ground_truth_path, prediction_path, output_path=None):
    """Compare ground truth with predictions"""
    
    with rasterio.open(ground_truth_path) as src:
        ground_truth = src.read(10)  # Fire mask band
    
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ground truth
    axes[0].imshow(ground_truth, cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth Fire Mask', fontsize=14)
    axes[0].axis('off')
    
    # Prediction
    im1 = axes[1].imshow(prediction, cmap=create_fire_colormap(), vmin=0, vmax=1)
    axes[1].set_title('Predicted Fire Probability', fontsize=14)
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(ground_truth - prediction)
    im2 = axes[2].imshow(diff, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title('Absolute Difference', fontsize=14)
    axes[2].axis('off')
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to {output_path}")
    
    plt.show()