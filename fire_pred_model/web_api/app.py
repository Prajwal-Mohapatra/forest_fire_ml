# ====================
# web_api/app.py
# Flask API for Forest Fire Simulation Web Interface
# ====================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import traceback
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

# Import our CA integration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellular_automata.integration import MLCAIntegration, get_web_simulation_data
from cellular_automata.config import SIMULATION_SCENARIOS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
MODEL_PATH = "outputs/final_model.h5"
DATA_DIRECTORY = "../../../refined_dataset_stacking"  # Adjust path as needed
CACHE_DIRECTORY = "cache"

# Global variables
ml_ca_integration = None
cached_predictions = {}

def initialize_integration():
    """Initialize the ML-CA integration system"""
    global ml_ca_integration
    
    try:
        # Adjust paths based on actual file structure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", MODEL_PATH)
        data_dir = os.path.join(current_dir, DATA_DIRECTORY)
        
        # Check if paths exist
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}")
            # Try alternative location
            model_path = os.path.join(current_dir, "..", "outputs", "final_model.h5")
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found at alternative location: {model_path}")
                return False
        
        ml_ca_integration = MLCAIntegration(
            model_path=model_path,
            data_directory=data_dir,
            output_directory="outputs/web_api"
        )
        
        print("✅ ML-CA Integration initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize ML-CA Integration: {str(e)}")
        traceback.print_exc()
        return False

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_ca_integration': ml_ca_integration is not None
    })

@app.route('/api/available-dates', methods=['GET'])
def get_available_dates():
    """Get list of available simulation dates"""
    try:
        if not ml_ca_integration:
            return jsonify({'error': 'ML-CA integration not initialized'}), 500
        
        dates = ml_ca_integration.get_available_dates()
        
        # Convert to more readable format
        formatted_dates = []
        for date_str in dates:
            try:
                # Convert 2016_04_15 to readable format
                year, month, day = date_str.split('_')
                date_obj = datetime(int(year), int(month), int(day))
                formatted_dates.append({
                    'value': date_str,
                    'label': date_obj.strftime('%B %d, %Y'),
                    'iso': date_obj.isoformat()
                })
            except:
                # If parsing fails, use original format
                formatted_dates.append({
                    'value': date_str,
                    'label': date_str.replace('_', '-'),
                    'iso': date_str
                })
        
        return jsonify({
            'dates': formatted_dates,
            'total_count': len(formatted_dates)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulation-scenarios', methods=['GET'])
def get_simulation_scenarios():
    """Get predefined simulation scenarios"""
    return jsonify({
        'scenarios': SIMULATION_SCENARIOS
    })

@app.route('/api/run-simulation', methods=['POST'])
def run_simulation():
    """Run fire spread simulation"""
    try:
        if not ml_ca_integration:
            return jsonify({'error': 'ML-CA integration not initialized'}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['date', 'ignition_points']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters
        date = data['date']
        ignition_points = data['ignition_points']
        simulation_hours = data.get('simulation_hours', 6)
        weather_data = data.get('weather_data', {
            'wind_speed': 5.0,
            'wind_direction': 45.0,
            'humidity': 40.0,
            'temperature': 25.0
        })
        
        # Validate ignition points format
        try:
            ignition_points = [(int(point['x']), int(point['y'])) for point in ignition_points]
        except (KeyError, ValueError, TypeError):
            return jsonify({'error': 'Invalid ignition_points format. Expected: [{"x": int, "y": int}, ...]'}), 400
        
        print(f"🔥 Running simulation for date: {date}")
        print(f"   Ignition points: {ignition_points}")
        print(f"   Duration: {simulation_hours} hours")
        
        # Run simulation
        web_data = get_web_simulation_data(
            model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", MODEL_PATH),
            data_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIRECTORY),
            date=date,
            ignition_points=ignition_points,
            simulation_hours=simulation_hours
        )
        
        print(f"✅ Simulation completed successfully")
        
        return jsonify({
            'success': True,
            'simulation_data': web_data,
            'parameters': {
                'date': date,
                'ignition_points': [{'x': x, 'y': y} for x, y in ignition_points],
                'simulation_hours': simulation_hours,
                'weather_data': weather_data
            }
        })
        
    except Exception as e:
        print(f"❌ Simulation failed: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-prediction', methods=['POST'])
def generate_prediction():
    """Generate ML fire probability prediction for a specific date"""
    try:
        if not ml_ca_integration:
            return jsonify({'error': 'ML-CA integration not initialized'}), 500
        
        data = request.get_json()
        date = data.get('date')
        
        if not date:
            return jsonify({'error': 'Missing required field: date'}), 400
        
        print(f"🧠 Generating ML prediction for date: {date}")
        
        # Generate prediction
        prediction_path = ml_ca_integration.generate_daily_prediction(date)
        
        return jsonify({
            'success': True,
            'prediction_path': prediction_path,
            'date': date
        })
        
    except Exception as e:
        print(f"❌ Prediction generation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multiple-scenarios', methods=['POST'])
def run_multiple_scenarios():
    """Run multiple fire scenarios for comparison"""
    try:
        if not ml_ca_integration:
            return jsonify({'error': 'ML-CA integration not initialized'}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['date', 'scenarios']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        date = data['date']
        scenarios = data['scenarios']
        
        # Validate scenarios format
        for i, scenario in enumerate(scenarios):
            if 'ignition_points' not in scenario:
                return jsonify({'error': f'Scenario {i+1} missing ignition_points'}), 400
        
        print(f"🎭 Running {len(scenarios)} scenarios for date: {date}")
        
        # Run multiple scenarios
        results = ml_ca_integration.run_multiple_scenarios(
            date=date,
            scenario_configs=scenarios,
            base_simulation_hours=data.get('simulation_hours', 6)
        )
        
        return jsonify({
            'success': True,
            'comparison_results': results
        })
        
    except Exception as e:
        print(f"❌ Multiple scenarios failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulation-cache/<date>', methods=['GET'])
def get_cached_simulation(date):
    """Get cached simulation results for a date"""
    try:
        cache_path = os.path.join(CACHE_DIRECTORY, f"simulation_{date}.json")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            return jsonify({
                'success': True,
                'cached': True,
                'data': cached_data
            })
        else:
            return jsonify({
                'success': True,
                'cached': False,
                'message': 'No cached data found'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-results/<simulation_id>', methods=['GET'])
def export_simulation_results(simulation_id):
    """Export simulation results as downloadable file"""
    try:
        # Implementation for exporting results
        # This would typically create a zip file with all simulation outputs
        
        return jsonify({
            'message': 'Export functionality coming soon',
            'simulation_id': simulation_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Utility functions for the API

def validate_coordinates(x, y, max_x, max_y):
    """Validate ignition point coordinates"""
    if not (0 <= x < max_x and 0 <= y < max_y):
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds. Max: ({max_x}, {max_y})")

def cache_simulation_results(date, results):
    """Cache simulation results for faster retrieval"""
    try:
        os.makedirs(CACHE_DIRECTORY, exist_ok=True)
        cache_path = os.path.join(CACHE_DIRECTORY, f"simulation_{date}.json")
        
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Cached simulation results for {date}")
        
    except Exception as e:
        print(f"⚠️ Failed to cache results: {str(e)}")

# Configuration endpoint for frontend
@app.route('/api/config', methods=['GET'])
def get_api_config():
    """Get API configuration for frontend"""
    return jsonify({
        'api_version': '1.0.0',
        'max_simulation_hours': 24,
        'max_ignition_points': 10,
        'supported_formats': ['GeoTIFF', 'JSON'],
        'default_weather': {
            'wind_speed': 5.0,
            'wind_direction': 45.0,
            'humidity': 40.0,
            'temperature': 25.0
        },
        'resolution_meters': 30,
        'coordinate_system': 'Image coordinates (pixel-based)'
    })

if __name__ == '__main__':
    print("🌐 Starting Forest Fire Simulation API...")
    
    # Initialize the ML-CA integration
    if initialize_integration():
        print("🚀 API server starting on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start API server - initialization failed")
        exit(1)
