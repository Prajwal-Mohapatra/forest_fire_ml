# ====================
# run_fire_simulation_demo.py
# Complete Fire Spread Simulation Demo
# ====================

import os
import sys
import argparse
from datetime import datetime
import json

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def run_demo_simulation():
    """Run a complete demo simulation"""
    print("🔥 FOREST FIRE SPREAD SIMULATION DEMO")
    print("="*50)
    
    # Configuration
    MODEL_PATH = "outputs/final_model.h5"
    DATA_DIRECTORY = "../../refined_dataset_stacking"  # Adjust as needed
    
    try:
        # Import CA system
        from cellular_automata.integration import quick_integrated_simulation
        
        print("✅ CA system imported successfully")
        
        # Demo parameters
        demo_date = "2016_04_15"  # April 15, 2016
        demo_ignition_points = [(100, 100), (150, 120)]  # Example coordinates
        demo_simulation_hours = 6
        
        print(f"📊 Demo Configuration:")
        print(f"   Date: {demo_date}")
        print(f"   Ignition points: {demo_ignition_points}")
        print(f"   Duration: {demo_simulation_hours} hours")
        
        # Check if model and data exist
        model_path = os.path.join(current_dir, MODEL_PATH)
        data_path = os.path.join(current_dir, DATA_DIRECTORY)
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}")
            print("   Creating mock simulation instead...")
            return run_mock_simulation()
        
        if not os.path.exists(data_path):
            print(f"⚠️ Data directory not found at {data_path}")
            print("   Creating mock simulation instead...")
            return run_mock_simulation()
        
        print(f"🚀 Running integrated simulation...")
        
        # Run the integrated simulation
        results = quick_integrated_simulation(
            model_path=model_path,
            data_directory=data_path,
            date=demo_date,
            ignition_points=demo_ignition_points,
            simulation_hours=demo_simulation_hours
        )
        
        print(f"✅ Simulation completed successfully!")
        
        # Display results summary
        print(f"\n📊 Results Summary:")
        ca_results = results['ca_simulation']
        stats = ca_results['statistics']
        
        print(f"   Total frames generated: {stats['total_frames']}")
        
        if stats['frame_statistics']:
            final_stats = stats['frame_statistics'][-1]
            print(f"   Final burned area: {final_stats['burned_area_hectares']:.2f} hectares")
            print(f"   Total fire pixels: {final_stats['total_fire_pixels']}")
            print(f"   Max fire intensity: {final_stats['max_intensity']:.3f}")
        
        print(f"   Output directory: {results['output_directory']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import CA system: {str(e)}")
        print("   Running mock simulation instead...")
        return run_mock_simulation()
        
    except Exception as e:
        print(f"❌ Simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_mock_simulation():
    """Run a mock simulation for testing"""
    print("\n🎭 Running Mock Simulation...")
    
    try:
        import numpy as np
        
        # Create mock data
        map_size = 200
        probability_map = np.random.random((map_size, map_size)) * 0.4
        probability_map[80:120, 80:120] = 0.8  # High risk area
        
        # Initialize fire state
        fire_state = np.zeros((map_size, map_size))
        
        # Set ignition points
        ignition_points = [(100, 100), (110, 110)]
        for x, y in ignition_points:
            fire_state[y, x] = 1.0
        
        print(f"📊 Mock simulation setup:")
        print(f"   Map size: {map_size}x{map_size}")
        print(f"   Ignition points: {ignition_points}")
        
        # Simple spread simulation
        frames = [fire_state.copy()]
        
        for step in range(6):  # 6 hours
            # Simple neighbor-based spread using scipy if available
            try:
                from scipy import ndimage
                
                kernel = np.array([
                    [0.1, 0.2, 0.1],
                    [0.2, 0.0, 0.2], 
                    [0.1, 0.2, 0.1]
                ])
                
                neighbor_influence = ndimage.convolve(fire_state, kernel, mode='constant')
                spread_prob = 0.15 * probability_map * (1.0 + neighbor_influence)
                
                random_values = np.random.random(fire_state.shape)
                new_ignitions = (random_values < spread_prob).astype(np.float32)
                
                fire_state = np.maximum(fire_state, new_ignitions)
                frames.append(fire_state.copy())
                
                fire_pixels = np.sum(fire_state > 0)
                print(f"   Hour {step + 1}: {fire_pixels} fire pixels")
                
            except ImportError:
                # Fallback without scipy
                print(f"   Hour {step + 1}: Mock step (scipy not available)")
                fire_state[fire_state > 0] = 1.0  # Keep existing fire
                frames.append(fire_state.copy())
        
        # Mock results
        mock_results = {
            'frames': frames,
            'total_frames': len(frames),
            'final_fire_pixels': int(np.sum(fire_state > 0)),
            'burned_area_hectares': float(np.sum(fire_state > 0) * 0.09),  # 30m = 900m² = 0.09ha
            'status': 'mock_completed'
        }
        
        print(f"✅ Mock simulation completed!")
        print(f"   Total frames: {mock_results['total_frames']}")
        print(f"   Final fire pixels: {mock_results['final_fire_pixels']}")
        print(f"   Burned area: {mock_results['burned_area_hectares']:.2f} hectares")
        
        # Save mock results
        output_dir = "outputs/mock_simulation"
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, "mock_results.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                'total_frames': mock_results['total_frames'],
                'final_fire_pixels': mock_results['final_fire_pixels'], 
                'burned_area_hectares': mock_results['burned_area_hectares'],
                'status': mock_results['status'],
                'timestamp': datetime.now().isoformat()
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Mock results saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock simulation failed: {str(e)}")
        return False

def start_web_server():
    """Start the web API server"""
    print("\n🌐 Starting Web API Server...")
    
    try:
        # Import and start the Flask app
        from web_api.app import app, initialize_integration
        
        print("✅ Web API imported successfully")
        
        if initialize_integration():
            print("🚀 Starting server on http://localhost:5000")
            print("   Available endpoints:")
            print("   - GET  /api/health")
            print("   - GET  /api/available-dates")
            print("   - POST /api/run-simulation")
            print("   - GET  /api/config")
            print("\n   Press Ctrl+C to stop the server")
            
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            print("❌ Failed to initialize integration - server not started")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import web API: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Failed to start web server: {str(e)}")
        return False

def run_tests():
    """Run the CA system tests"""
    print("\n🧪 Running CA System Tests...")
    
    try:
        from test_ca_system import run_comprehensive_test
        return run_comprehensive_test()
    except ImportError as e:
        print(f"❌ Failed to import test module: {str(e)}")
        return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Forest Fire Spread Simulation Demo')
    parser.add_argument('command', choices=['demo', 'test', 'server'], 
                       help='Command to run')
    parser.add_argument('--mock', action='store_true', 
                       help='Force mock simulation even if real data available')
    
    args = parser.parse_args()
    
    print(f"🔥 Forest Fire Simulation System")
    print(f"   Command: {args.command}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    if args.command == 'demo':
        if args.mock:
            success = run_mock_simulation()
        else:
            success = run_demo_simulation()
    elif args.command == 'test':
        success = run_tests()
    elif args.command == 'server':
        success = start_web_server()
    
    if success:
        print(f"\n✅ {args.command.title()} completed successfully!")
    else:
        print(f"\n❌ {args.command.title()} failed!")
        sys.exit(1)

if __name__ == "__main__":
    # If no arguments provided, run demo
    if len(sys.argv) == 1:
        print("🔥 No command specified, running demo simulation...")
        sys.argv.append('demo')
    
    main()
