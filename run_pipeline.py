"""
Complete pipeline script for fire prediction model training and evaluation
"""

import os
import sys
import glob
import argparse
from datetime import datetime

def setup_environment():
    """Setup directories and environment"""
    
    # Create directory structure
    directories = [
        'outputs/checkpoints',
        'outputs/logs',
        'outputs/plots',
        'outputs/predictions',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Environment setup complete!")

def run_training(data_dir, config=None):
    """Run model training"""
    
    # Import and run training
    from train import main
    main()
    
    print("✅ Training completed successfully!")

def run_evaluation(model_path, test_data_dir):
    """Run model evaluation"""
    
    print("📊 Starting Model Evaluation...")
    
    # Get test files
    test_files = sorted(glob.glob(os.path.join(test_data_dir, 'stack_2016_05_2*.tif')))
    
    from evaluate import evaluate_model
    results = evaluate_model(model_path, test_files)
    
    print(f"✅ Evaluation completed! Results: {results}")

def run_prediction(model_path, input_tif, output_path):
    """Run comprehensive fire probability prediction with confidence zones"""
    
    print("🔮 Starting Comprehensive Fire Prediction...")
    
    from predict import predict_fire_nextday
    
    # Check if output_path is a directory or file
    if os.path.isdir(output_path) or not output_path.endswith('.tif'):
        # If it's a directory, create output directory
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        
        # Use predict_fire_nextday for comprehensive prediction (includes confidence zones)
        predict_fire_nextday(
            model_path=model_path,
            input_tif_path=input_tif,
            output_dir=output_dir,
            threshold=0.3,  # Use optimized threshold
            patch_size=256,
            overlap=64
        )
        
        # Use the binary map for visualization
        binary_map_path = os.path.join(output_dir, 'fire_binary_map.tif')
        if os.path.exists(binary_map_path):
            # Visualize results
            from utils.visualization import visualize_fire_prediction
            visualize_fire_prediction(binary_map_path, os.path.join(output_dir, 'fire_prediction_visualization.png'))
    else:
        # If it's a file path, use the directory for output
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Use predict_fire_nextday for comprehensive prediction (includes confidence zones)
        prediction = predict_fire_nextday(
            model_path=model_path,
            input_tif_path=input_tif,
            output_dir=output_dir,
            threshold=0.3,  # Use optimized threshold
            patch_size=256,
            overlap=64
        )
        
        # Visualize results - use binary map if available, otherwise the specified output path
        binary_map_path = os.path.join(output_dir, 'fire_binary_map.tif')
        if os.path.exists(binary_map_path):
            from utils.visualization import visualize_fire_prediction
            visualize_fire_prediction(binary_map_path, f"{output_path.replace('.tif', '_visualization.png')}")
    
    print("✅ Prediction completed successfully!")

def main():
    """Main pipeline execution"""
    
    parser = argparse.ArgumentParser(description='Fire Prediction ML Pipeline')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict', 'full'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing stacked TIF files')
    parser.add_argument('--model_path', type=str, default='outputs/checkpoints/best_model.keras',
                       help='Path to trained model')
    parser.add_argument('--input_tif', type=str, help='Input TIF file for prediction')
    parser.add_argument('--output_path', type=str, default='outputs/predictions/fire_probability.tif',
                       help='Output path for predictions')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Log start time
    start_time = datetime.now()
    print(f"🚀 Pipeline started at {start_time}")
    
    try:
        if args.mode in ['train', 'full']:
            run_training(args.data_dir)
        
        if args.mode in ['evaluate', 'full']:
            run_evaluation(args.model_path, args.data_dir)
        
        if args.mode in ['predict', 'full']:
            if args.input_tif:
                run_prediction(args.model_path, args.input_tif, args.output_path)
            else:
                print("⚠️ No input TIF specified for prediction")
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"✅ Pipeline completed successfully!")
        print(f"⏱️ Total execution time: {duration}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
