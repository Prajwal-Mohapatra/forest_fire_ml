# 🌉 ML-CA Integration Bridge Documentation

## Overview

The ML-CA Integration Bridge (`cellular_automata/integration/ml_ca_bridge.py`) orchestrates the seamless flow of data from ResUNet-A ML predictions to the Cellular Automata simulation engine. It handles prediction generation, data validation, scenario management, and complete pipeline execution.

## Architecture Design

### Integration Workflow

```
Input Data → ML Prediction → Data Validation → CA Simulation → Results Package
     ↓             ↓              ↓              ↓              ↓
Stacked TIFF   Probability    Consistency    Fire Spread    Export Ready
(9 bands)      Map (0-1)      Checks         Animation      JSON/GeoTIFF
```

### Core Components

```python
class MLCABridge:
    """
    Main integration class coordinating ML and CA components
    """
    def __init__(self, ml_model_path, ca_output_dir):
        self.ml_model_path = ml_model_path
        self.ca_output_dir = ca_output_dir
        self.ml_available = os.path.exists(ml_model_path)

    # Core Methods:
    # - generate_ml_prediction()      # ML model inference
    # - run_ca_simulation()           # CA engine execution
    # - run_complete_pipeline()       # End-to-end orchestration
    # - validate_data_consistency()   # Quality assurance
    # - create_demo_scenario()        # Testing and demos
```

## Core Functionality

### 1. ML Prediction Generation

```python
def generate_ml_prediction(self, input_tif_path, output_path=None, date_str=None):
    """
    Generate fire probability prediction using ResUNet-A model

    Args:
        input_tif_path: Path to 9-band stacked environmental data
        output_path: Where to save probability map (auto-generated if None)
        date_str: Date string for file naming (YYYY_MM_DD format)

    Returns:
        Path to generated probability map or None if failed

    Process:
        1. Validate input file exists and is readable
        2. Import ML prediction function from predict.py
        3. Run sliding window prediction across full image
        4. Save probability map as GeoTIFF
        5. Return path for CA engine consumption
    """

    if not self.ml_available:
        print("❌ ML model not available")
        return None

    if not os.path.exists(input_tif_path):
        print(f"❌ Input data not found: {input_tif_path}")
        return None

    # Set default output path
    if output_path is None:
        if date_str is None:
            date_str = datetime.now().strftime("%Y_%m_%d")
        output_path = os.path.join(
            self.ca_output_dir, f"probability_map_{date_str}.tif"
        )

    try:
        # Import and execute ML prediction
        from predict import predict_fire_probability

        prediction = predict_fire_probability(
            model_path=self.ml_model_path,
            input_tif_path=input_tif_path,
            output_path=output_path,
            patch_size=256,
            overlap=64
        )

        if prediction and os.path.exists(output_path):
            print(f"✅ ML prediction generated: {output_path}")
            return output_path
        else:
            print("❌ ML prediction failed")
            return None

    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        return None
```

### 2. CA Simulation Execution

```python
def run_ca_simulation(self, probability_map_path, ignition_points,
                     weather_params, simulation_hours=6, scenario_name=None):
    """
    Execute cellular automata fire spread simulation

    Args:
        probability_map_path: Path to ML-generated probability map
        ignition_points: List of (longitude, latitude) ignition coordinates
        weather_params: Dictionary with wind_speed, wind_direction, temperature, humidity
        simulation_hours: Duration of simulation (1-24 hours)
        scenario_name: Optional name for scenario tracking

    Returns:
        Complete simulation results or None if failed

    Process:
        1. Validate probability map exists and is valid
        2. Import CA engine from cellular_automata module
        3. Initialize simulation with probability map as base state
        4. Execute hourly simulation steps
        5. Generate statistics and save frames
        6. Package results for consumption
    """

    if not os.path.exists(probability_map_path):
        print(f"❌ Probability map not found: {probability_map_path}")
        return None

    try:
        # Import CA engine
        from cellular_automata.ca_engine import run_quick_simulation

        print(f"🔥 Running CA simulation...")
        print(f"   Probability map: {probability_map_path}")
        print(f"   Ignition points: {len(ignition_points)}")
        print(f"   Duration: {simulation_hours} hours")

        # Set output directory
        if scenario_name is None:
            scenario_name = f"ml_ca_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir = os.path.join(self.ca_output_dir, scenario_name)

        # Execute simulation
        results = run_quick_simulation(
            probability_map_path=probability_map_path,
            ignition_points=ignition_points,
            weather_params=weather_params,
            simulation_hours=simulation_hours,
            output_dir=output_dir
        )

        if results:
            print(f"✅ CA simulation completed: {results['scenario_id']}")
            return results
        else:
            print("❌ CA simulation failed")
            return None

    except Exception as e:
        print(f"❌ CA simulation error: {e}")
        return None
```

### 3. Complete Pipeline Orchestration

```python
def run_complete_pipeline(self, input_data_path, ignition_points, weather_params,
                         simulation_hours=6, date_str=None, scenario_name=None):
    """
    Execute complete ML→CA pipeline from input data to final results

    Args:
        input_data_path: Path to stacked environmental data for ML input
        ignition_points: Fire ignition coordinates [(lon, lat), ...]
        weather_params: Weather conditions dict
        simulation_hours: Simulation duration
        date_str: Date for file naming
        scenario_name: Scenario identifier

    Returns:
        Complete pipeline results with metadata

    Pipeline Steps:
        1. Generate ML fire probability prediction
        2. Validate data consistency between ML output and CA input
        3. Execute CA simulation using probability map
        4. Package complete results with metadata
        5. Save comprehensive results file
    """

    print("🚀 Running complete ML→CA pipeline")
    print("=" * 50)

    # Step 1: Generate ML prediction
    print("\n📊 Step 1: ML Prediction Generation")
    probability_map_path = self.generate_ml_prediction(
        input_tif_path=input_data_path,
        date_str=date_str
    )

    if probability_map_path is None:
        print("❌ Pipeline failed at ML prediction step")
        return None

    # Step 2: Validate data consistency
    print("\n🔍 Step 2: Data Validation")
    if not self.validate_data_consistency(input_data_path, probability_map_path):
        print("⚠️ Data consistency validation failed, continuing anyway...")

    # Step 3: Run CA simulation
    print("\n🔥 Step 3: CA Simulation")
    ca_results = self.run_ca_simulation(
        probability_map_path=probability_map_path,
        ignition_points=ignition_points,
        weather_params=weather_params,
        simulation_hours=simulation_hours,
        scenario_name=scenario_name
    )

    if ca_results is None:
        print("❌ Pipeline failed at CA simulation step")
        return None

    # Step 4: Package complete results
    print("\n📦 Step 4: Results Packaging")
    complete_results = {
        'pipeline_id': f"ml_ca_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'input_data_path': input_data_path,
        'probability_map_path': probability_map_path,
        'ml_model_path': self.ml_model_path,
        'ca_results': ca_results,
        'simulation_parameters': {
            'ignition_points': ignition_points,
            'weather_params': weather_params,
            'simulation_hours': simulation_hours
        },
        'created_at': datetime.now().isoformat()
    }

    # Save complete results
    results_path = os.path.join(
        self.ca_output_dir,
        f"{complete_results['pipeline_id']}_complete_results.json"
    )

    with open(results_path, 'w') as f:
        json.dump(complete_results, f, indent=2, default=str)

    print(f"✅ Complete pipeline results saved: {results_path}")
    return complete_results
```

## Data Validation and Quality Assurance

### Spatial Consistency Validation

```python
def validate_data_consistency(self, input_data_path, probability_map_path):
    """
    Validate spatial and data consistency between input and output

    Validation Checks:
        1. Spatial dimensions match (height, width)
        2. Geographic bounds are consistent
        3. Coordinate reference system (CRS) matches
        4. Probability values are in valid range [0,1]
        5. No excessive missing data

    Returns:
        True if all validations pass, False otherwise
    """

    try:
        import rasterio

        # Check input data properties
        with rasterio.open(input_data_path) as src_input:
            input_shape = (src_input.height, src_input.width)
            input_bounds = src_input.bounds
            input_crs = src_input.crs

        # Check probability map properties
        with rasterio.open(probability_map_path) as src_prob:
            prob_shape = (src_prob.height, src_prob.width)
            prob_bounds = src_prob.bounds
            prob_crs = src_prob.crs
            prob_data = src_prob.read(1)

        # Validation 1: Shape consistency
        if input_shape != prob_shape:
            print(f"⚠️ Shape mismatch: input {input_shape} vs probability {prob_shape}")
            return False

        # Validation 2: Bounds consistency (allow small floating point differences)
        bounds_diff = max(abs(a - b) for a, b in zip(input_bounds, prob_bounds))
        if bounds_diff > 0.001:
            print(f"⚠️ Bounds mismatch: difference {bounds_diff}")
            return False

        # Validation 3: CRS consistency
        if input_crs != prob_crs:
            print(f"⚠️ CRS mismatch: {input_crs} vs {prob_crs}")
            return False

        # Validation 4: Probability value range
        if not (0 <= prob_data.min() <= prob_data.max() <= 1):
            print(f"⚠️ Invalid probability range: [{prob_data.min()}, {prob_data.max()}]")
            return False

        # Validation 5: Missing data check
        missing_percentage = np.isnan(prob_data).sum() / prob_data.size * 100
        if missing_percentage > 10:
            print(f"⚠️ Excessive missing data: {missing_percentage:.1f}%")
            return False

        print("✅ Data consistency validation passed")
        return True

    except Exception as e:
        print(f"❌ Data validation error: {e}")
        return False
```

## Demo and Testing Scenarios

### Predefined Demo Scenarios

```python
def create_demo_scenario(self, demo_date="2016_05_15", demo_location="dehradun"):
    """
    Create standardized demo scenarios for testing and presentation

    Available Locations:
        - dehradun: Dehradun city area (state capital)
        - rishikesh: Rishikesh spiritual town
        - haridwar: Haridwar pilgrimage city
        - nainital: Nainital hill station

    Demo Configurations:
        - Pre-defined ignition points for each location
        - Realistic weather conditions for May 2016
        - Standard 6-hour simulation duration
        - Full documentation and metadata
    """

    print(f"🎬 Creating demo scenario: {demo_location} on {demo_date}")

    # Demo ignition coordinates (major cities/areas in Uttarakhand)
    demo_ignition_points = {
        'dehradun': [(78.0322, 30.3165)],    # Dehradun city center
        'rishikesh': [(78.2676, 30.0869)],   # Rishikesh town center
        'haridwar': [(78.1642, 29.9457)],    # Haridwar city center
        'nainital': [(79.4633, 29.3803)]     # Nainital lake area
    }

    # Realistic weather conditions for May in Uttarakhand
    demo_weather = {
        'wind_direction': 225,    # Southwest (typical pre-monsoon pattern)
        'wind_speed': 18,         # 18 km/h (moderate wind)
        'temperature': 32,        # 32°C (hot pre-monsoon temperature)
        'relative_humidity': 35   # 35% (dry conditions favor fire spread)
    }

    ignition_points = demo_ignition_points.get(demo_location, demo_ignition_points['dehradun'])

    # Check for real input data
    input_data_path = os.path.join(
        project_root, "dataset collection", f"stacked_data_{demo_date}.tif"
    )

    if os.path.exists(input_data_path):
        # Run with real data
        scenario_name = f"demo_{demo_location}_{demo_date}"

        results = self.run_complete_pipeline(
            input_data_path=input_data_path,
            ignition_points=ignition_points,
            weather_params=demo_weather,
            simulation_hours=6,
            date_str=demo_date,
            scenario_name=scenario_name
        )

        return results
    else:
        print(f"⚠️ Real data not found: {input_data_path}")
        print("   Creating synthetic demo scenario...")

        # Fallback to synthetic data for demonstration
        return self._create_synthetic_demo(demo_date, demo_location, ignition_points, demo_weather)
```

### Synthetic Demo Creation

```python
def _create_synthetic_demo(self, demo_date, demo_location, ignition_points, weather_params):
    """
    Create synthetic demo when real data is not available

    Generates:
        - Synthetic probability map with realistic patterns
        - Proper geospatial metadata and projection
        - Compatible format for CA engine consumption
    """

    try:
        # Create synthetic probability map with realistic patterns
        from cellular_automata.test_ca_engine import create_synthetic_probability_map

        prob_map_path = os.path.join(
            self.ca_output_dir, f"synthetic_demo_{demo_location}_{demo_date}.tif"
        )

        if create_synthetic_probability_map(prob_map_path, width=400, height=400):
            scenario_name = f"synthetic_demo_{demo_location}_{demo_date}"

            results = self.run_ca_simulation(
                probability_map_path=prob_map_path,
                ignition_points=ignition_points,
                weather_params=weather_params,
                simulation_hours=6,
                scenario_name=scenario_name
            )

            return results
        else:
            print("❌ Synthetic demo creation failed")
            return None

    except Exception as e:
        print(f"❌ Synthetic demo creation error: {e}")
        return None
```

## Convenience Functions

### Quick Simulation Interface

```python
def quick_ml_ca_simulation(input_data_path, ignition_lat, ignition_lon,
                          wind_direction=45, wind_speed=15, simulation_hours=6):
    """
    Streamlined interface for quick simulations with minimal parameters

    Args:
        input_data_path: Path to stacked environmental data
        ignition_lat, ignition_lon: Single ignition point coordinates
        wind_direction: Wind direction in degrees (0-360)
        wind_speed: Wind speed in km/h
        simulation_hours: Simulation duration

    Returns:
        Complete simulation results

    Usage Example:
        results = quick_ml_ca_simulation(
            input_data_path="data/stack_2016_05_15.tif",
            ignition_lat=30.3165, ignition_lon=78.0322,  # Dehradun
            wind_direction=225, wind_speed=20,
            simulation_hours=6
        )
    """

    bridge = MLCABridge()

    weather_params = {
        'wind_direction': wind_direction,
        'wind_speed': wind_speed,
        'temperature': 30,        # Default temperature
        'relative_humidity': 40   # Default humidity
    }

    return bridge.run_complete_pipeline(
        input_data_path=input_data_path,
        ignition_points=[(ignition_lon, ignition_lat)],  # Note: lon, lat order
        weather_params=weather_params,
        simulation_hours=simulation_hours
    )
```

### Batch Demo Creation

```python
def create_demo_scenarios():
    """
    Create multiple demo scenarios for comprehensive testing and presentation

    Returns:
        List of demo scenario results for comparison and analysis
    """

    bridge = MLCABridge()

    demo_configs = [
        ('dehradun', '2016_05_15'),    # State capital scenario
        ('rishikesh', '2016_05_20'),   # Tourist area scenario
        ('nainital', '2016_05_25')     # Hill station scenario
    ]

    results = []

    for location, date in demo_configs:
        print(f"\n🎬 Creating demo: {location} on {date}")
        result = bridge.create_demo_scenario(date, location)
        if result:
            results.append(result)

    print(f"\n✅ Created {len(results)} demo scenarios")
    return results
```

## Integration with Web Interface

### API Endpoint Support

```python
def prepare_web_api_response(self, pipeline_results):
    """
    Format pipeline results for web interface consumption

    Converts:
        - Simulation frames to web-compatible format
        - Statistics to JSON-serializable format
        - File paths to web-accessible URLs
        - Metadata for frontend display
    """

    if not pipeline_results or 'ca_results' not in pipeline_results:
        return None

    ca_results = pipeline_results['ca_results']

    web_response = {
        'scenario_id': ca_results.get('scenario_id'),
        'status': 'completed',
        'simulation_hours': len(ca_results.get('hourly_statistics', [])),
        'final_statistics': ca_results.get('hourly_statistics', [{}])[-1],
        'frame_urls': [f"/api/frames/{os.path.basename(path)}"
                      for path in ca_results.get('frame_paths', [])],
        'animation_url': f"/api/animations/{ca_results.get('scenario_id')}.gif",
        'export_urls': {
            'geotiff': f"/api/export/{ca_results.get('scenario_id')}/frames.zip",
            'json': f"/api/export/{ca_results.get('scenario_id')}/results.json",
            'statistics': f"/api/export/{ca_results.get('scenario_id')}/stats.csv"
        },
        'metadata': {
            'ignition_points': pipeline_results['simulation_parameters']['ignition_points'],
            'weather_conditions': pipeline_results['simulation_parameters']['weather_params'],
            'created_at': pipeline_results['created_at']
        }
    }

    return web_response
```

## Error Handling and Logging

### Robust Error Management

```python
def handle_pipeline_errors(self, operation_name, error_details):
    """
    Centralized error handling with detailed logging and recovery strategies
    """

    error_log = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation_name,
        'error_type': type(error_details).__name__,
        'error_message': str(error_details),
        'stack_trace': traceback.format_exc()
    }

    # Log to file
    log_path = os.path.join(self.ca_output_dir, 'error_log.json')
    with open(log_path, 'a') as f:
        json.dump(error_log, f, indent=2)
        f.write('\n')

    # Determine recovery strategy
    if 'ML prediction' in operation_name:
        print("💡 Recovery suggestion: Check ML model path and input data format")
        return 'ml_fallback'
    elif 'CA simulation' in operation_name:
        print("💡 Recovery suggestion: Verify probability map format and CA engine setup")
        return 'ca_fallback'
    else:
        print("💡 Recovery suggestion: Review input parameters and file paths")
        return 'general_fallback'
```

## Performance Optimization

### Pipeline Performance Monitoring

```python
def monitor_pipeline_performance(self):
    """
    Track performance metrics for optimization
    """

    performance_metrics = {
        'ml_prediction_time': 0,
        'ca_simulation_time': 0,
        'data_validation_time': 0,
        'total_pipeline_time': 0,
        'memory_usage_peak': 0,
        'gpu_utilization': 0
    }

    # Implementation for tracking execution times and resource usage
    return performance_metrics
```

---

**Key Functions:**

- `generate_ml_prediction()`: ML model orchestration
- `run_ca_simulation()`: CA engine execution
- `run_complete_pipeline()`: End-to-end workflow
- `validate_data_consistency()`: Quality assurance
- `create_demo_scenario()`: Testing and demonstration

**Integration Points:**

- ML Model: `forest_fire_ml/predict.py`
- CA Engine: `cellular_automata/ca_engine/core.py`
- Web Interface: REST API endpoints for real-time control
- Export System: Results packaging for external analysis
