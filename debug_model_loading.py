import tensorflow as tf
import h5py
import json

def inspect_saved_model(model_path):
    """Inspect what's saved in the model file"""
    print(f"🔍 Inspecting model: {model_path}")
    
    try:
        # Try to read the HDF5 file structure
        with h5py.File(model_path, 'r') as f:
            print("\n📁 HDF5 file structure:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
            f.visititems(print_structure)
            
            # Check for model config
            if 'model_config' in f.attrs:
                config = f.attrs['model_config']
                if isinstance(config, bytes):
                    config = config.decode('utf-8')
                print(f"\n📋 Model config (first 500 chars):")
                print(config[:500])
                
                # Try to parse as JSON
                try:
                    config_dict = json.loads(config)
                    print("\n🎯 Loss function info:")
                    if 'config' in config_dict and 'loss' in config_dict['config']:
                        print(f"  Loss: {config_dict['config']['loss']}")
                    if 'config' in config_dict and 'metrics' in config_dict['config']:
                        print(f"  Metrics: {config_dict['config']['metrics']}")
                except:
                    print("  Could not parse config as JSON")
                    
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")
    
    # Try loading with different approaches
    print("\n🧪 Testing different loading approaches:")
    
    # 1. Load without any custom objects
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ 1. Load without compilation: SUCCESS")
        print(f"   Model summary: {len(model.layers)} layers")
        del model
    except Exception as e:
        print(f"❌ 1. Load without compilation: FAILED - {e}")
    
    # 2. Load with minimal custom objects
    try:
        from utils.metrics import iou_score, dice_coef, focal_loss
        custom_objects = {
            'iou_score': iou_score,
            'dice_coef': dice_coef,
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ 2. Load with iou_score + dice_coef: SUCCESS")
        del model
    except Exception as e:
        print(f"❌ 2. Load with iou_score + dice_coef: FAILED - {e}")
    
    # 3. Load with focal_loss function
    try:
        from utils.metrics import iou_score, dice_coef, focal_loss
        custom_objects = {
            'iou_score': iou_score,
            'dice_coef': dice_coef,
            'focal_loss': focal_loss,
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ 3. Load with focal_loss function: SUCCESS")
        del model
    except Exception as e:
        print(f"❌ 3. Load with focal_loss function: FAILED - {e}")
    
    # 4. Load with focal_loss_fixed (the actual loss function)
    try:
        from utils.metrics import iou_score, dice_coef, focal_loss
        custom_objects = {
            'iou_score': iou_score,
            'dice_coef': dice_coef,
            'focal_loss_fixed': focal_loss(),
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ 4. Load with focal_loss_fixed: SUCCESS")
        del model
    except Exception as e:
        print(f"❌ 4. Load with focal_loss_fixed: FAILED - {e}")
    
    # 5. Load with all possible combinations
    try:
        from utils.metrics import iou_score, dice_coef, focal_loss
        custom_objects = {
            'iou_score': iou_score,
            'dice_coef': dice_coef,
            'focal_loss': focal_loss,
            'focal_loss_fixed': focal_loss(),
            'focal_loss_fixed_inner': focal_loss()
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✅ 5. Load with all combinations: SUCCESS")
        del model
    except Exception as e:
        print(f"❌ 5. Load with all combinations: FAILED - {e}")

if __name__ == "__main__":
    model_path = "/kaggle/working/forest_fire_ml/outputs/final_model.h5"
    inspect_saved_model(model_path)
