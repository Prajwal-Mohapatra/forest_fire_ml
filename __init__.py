# Forest Fire ML Package
"""
Forest Fire Machine Learning Module

This package contains the ML model, utilities, and prediction functions
for forest fire probability prediction using ResUNet-A architecture.
"""

__version__ = "1.0.0"
__author__ = "Forest Fire Simulation Team"

# Import main functions for easy access
try:
    from .predict import predict_fire_map, load_model_safe
    from .model.resunet_a import build_resunet_a
    from .utils.metrics import iou_score, dice_coef, focal_loss
except ImportError:
    # Graceful degradation if imports fail
    pass
