# ====================
# cellular_automata/core.py
# Core Cellular Automata Engine for Forest Fire Spread Simulation
# ====================

import numpy as np
import tensorflow as tf
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json

from .config import CAConfig, SimulationParams
from .utils import (
    load_raster_data, create_ignition_map, save_simulation_frame,
    create_fire_animation_data, apply_barriers
)
from .rules import FireSpreadRules, SimplifiedFireRules

class CellularAutomataEngine:
    """
    Main engine for forest fire spread simulation using cellular automata
    """
    
    def __init__(self, config: CAConfig = None):
        self.config = config or CAConfig()
        self.fire_rules = FireSpreadRules(self.config)
        self.simplified_rules = SimplifiedFireRules(self.config)
        
        # Initialize TensorFlow if using GPU
        if self.config.use_gpu:
            self._configure_tensorflow()
        
        self.simulation_data = {}
        self.current_state = None
        self.simulation_frames = []
        
    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal performance"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ TensorFlow configured with {len(gpus)} GPU(s)")
            else:
                print("⚠️ No GPU detected, falling back to CPU")
                self.config.use_gpu = False
        except Exception as e:
            print(f"⚠️ GPU configuration failed: {e}")
            self.config.use_gpu = False
    
    def load_simulation_data(self, params: SimulationParams):
        """
        Load all required data for simulation
        
        Args:
            params: Simulation parameters including data paths
        """
        print("📊 Loading simulation data...")
        
        # Load fire probability map (required)
        prob_data, prob_metadata = load_raster_data(params.probability_map_path)
        self.simulation_data['probability_map'] = prob_data
        self.simulation_data['metadata'] = prob_metadata
        
        print(f"✅ Loaded probability map: {prob_data.shape}")
        
        # Load optional datasets
        if params.dem_path and os.path.exists(params.dem_path):
            dem_data, _ = load_raster_data(params.dem_path, target_shape=prob_data.shape)
            self.simulation_data['dem'] = dem_data
            print(f"✅ Loaded DEM: {dem_data.shape}")
        
        if params.lulc_path and os.path.exists(params.lulc_path):
            lulc_data, _ = load_raster_data(params.lulc_path, target_shape=prob_data.shape)
            self.simulation_data['lulc'] = lulc_data
            print(f"✅ Loaded LULC: {lulc_data.shape}")
        
        if params.ghsl_path and os.path.exists(params.ghsl_path):
            ghsl_data, _ = load_raster_data(params.ghsl_path, target_shape=prob_data.shape)
            # Convert GHSL to barrier map (urban areas = barriers)
            barrier_map = (ghsl_data > 0).astype(np.float32)
            self.simulation_data['barriers'] = barrier_map
            print(f"✅ Loaded GHSL barriers: {barrier_map.shape}")
        
        # Create ignition map
        if params.ignition_points:
            ignition_map = create_ignition_map(
                prob_data.shape, 
                params.ignition_points, 
                params.ignition_intensities
            )
            self.simulation_data['ignition_map'] = ignition_map
            print(f"✅ Created ignition map with {len(params.ignition_points)} points")
        
        print("📊 Data loading complete!")
    
    def run_simulation(self, params: SimulationParams, use_simplified: bool = False) -> Dict:
        """
        Run complete fire spread simulation
        
        Args:
            params: Simulation parameters
            use_simplified: Whether to use simplified rules (faster)
            
        Returns:
            Dictionary with simulation results
        """
        print(f"🔥 Starting fire spread simulation...")
        print(f"   Duration: {params.simulation_hours} hours")
        print(f"   Output frequency: {params.output_frequency} hour(s)")
        print(f"   Method: {'Simplified' if use_simplified else 'TensorFlow'}")
        
        # Load data if not already loaded
        if not self.simulation_data:
            self.load_simulation_data(params)
        
        # Initialize simulation state
        self._initialize_simulation_state(params)
        
        # Run simulation loop
        simulation_results = self._run_simulation_loop(params, use_simplified)
        
        # Save results
        if params.output_dir:
            self._save_simulation_results(simulation_results, params)
        
        return simulation_results
    
    def _initialize_simulation_state(self, params: SimulationParams):
        """Initialize the simulation state"""
        shape = self.simulation_data['probability_map'].shape
        
        # Initialize fire state
        self.current_state = np.zeros(shape, dtype=np.float32)
        
        # Apply ignition points
        if 'ignition_map' in self.simulation_data:
            self.current_state = np.maximum(self.current_state, 
                                          self.simulation_data['ignition_map'])
        
        # Initialize frames list
        self.simulation_frames = [self.current_state.copy()]
        
        print(f"🎯 Initialized simulation state: {shape}")
        if np.sum(self.current_state > 0) > 0:
            print(f"   Initial fire pixels: {np.sum(self.current_state > 0)}")
    
    def _run_simulation_loop(self, params: SimulationParams, use_simplified: bool) -> Dict:
        """Main simulation loop"""
        
        weather = params.weather_data
        probability_map = self.simulation_data['probability_map']
        
        # Optional data
        dem = self.simulation_data.get('dem')
        lulc = self.simulation_data.get('lulc')
        barriers = self.simulation_data.get('barriers')
        
        total_steps = params.simulation_hours
        save_frequency = params.output_frequency
        
        print(f"🔄 Running simulation for {total_steps} time steps...")
        
        for step in range(1, total_steps + 1):
            print(f"   Step {step}/{total_steps}: ", end="")
            
            # Apply fire spread rules
            if use_simplified:
                new_state = self.simplified_rules.simple_spread(
                    self.current_state,
                    probability_map,
                    weather['wind_direction']
                )
            else:
                # Convert to TensorFlow tensors
                fire_state_tf = tf.constant(self.current_state)
                prob_map_tf = tf.constant(probability_map)
                dem_tf = tf.constant(dem) if dem is not None else None
                lulc_tf = tf.constant(lulc) if lulc is not None else None
                barriers_tf = tf.constant(barriers) if barriers is not None else None
                
                new_state = self.fire_rules.apply_spread_rules(
                    fire_state_tf,
                    prob_map_tf,
                    weather['wind_speed'],
                    weather['wind_direction'],
                    dem_tf,
                    lulc_tf,
                    barriers_tf
                ).numpy()
            
            # Update current state
            self.current_state = new_state
            
            # Save frame if needed
            if step % save_frequency == 0 or step == total_steps:
                self.simulation_frames.append(self.current_state.copy())
            
            # Print progress
            fire_pixels = np.sum(self.current_state > 0.1)
            print(f"Fire pixels: {fire_pixels}")
        
        # Prepare results
        results = {
            'frames': self.simulation_frames,
            'final_state': self.current_state,
            'metadata': self.simulation_data['metadata'],
            'parameters': {
                'simulation_hours': params.simulation_hours,
                'weather': weather,
                'ignition_points': params.ignition_points
            },
            'statistics': self._calculate_simulation_statistics()
        }
        
        print(f"✅ Simulation complete! Generated {len(self.simulation_frames)} frames")
        return results
    
    def _calculate_simulation_statistics(self) -> Dict:
        """Calculate statistics for the simulation"""
        stats = {
            'total_frames': len(self.simulation_frames),
            'frame_statistics': []
        }
        
        for i, frame in enumerate(self.simulation_frames):
            frame_stats = {
                'time_step': i,
                'total_fire_pixels': int(np.sum(frame > 0.1)),
                'max_intensity': float(np.max(frame)),
                'mean_intensity': float(np.mean(frame[frame > 0.1])) if np.sum(frame > 0.1) > 0 else 0.0,
                'burned_area_hectares': float(np.sum(frame > 0.1) * (self.config.resolution/100)**2)
            }
            stats['frame_statistics'].append(frame_stats)
        
        return stats
    
    def _save_simulation_results(self, results: Dict, params: SimulationParams):
        """Save simulation results to disk"""
        print("💾 Saving simulation results...")
        
        # Create output directory
        os.makedirs(params.output_dir, exist_ok=True)
        
        # Save individual frames as GeoTIFF
        for i, frame in enumerate(results['frames']):
            frame_path = os.path.join(params.output_dir, f"fire_state_frame_{i:03d}.tif")
            save_simulation_frame(
                frame,
                self.simulation_data['probability_map'],
                frame_path,
                results['metadata'],
                i
            )
        
        # Save animation data for web interface
        animation_data = create_fire_animation_data(
            results['frames'],
            results['metadata']
        )
        
        animation_path = os.path.join(params.output_dir, "animation_data.json")
        with open(animation_path, 'w') as f:
            json.dump(animation_data, f, indent=2)
        
        # Save simulation metadata
        metadata_path = os.path.join(params.output_dir, "simulation_metadata.json")
        with open(metadata_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Results saved to {params.output_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    def run_quick_demo(self, 
                      probability_map_path: str,
                      ignition_points: List[Tuple[int, int]],
                      simulation_hours: int = 3) -> Dict:
        """
        Run a quick demo simulation with minimal setup
        
        Args:
            probability_map_path: Path to fire probability map
            ignition_points: List of (x, y) ignition coordinates
            simulation_hours: Duration of simulation
            
        Returns:
            Simulation results
        """
        # Create simple parameters
        params = SimulationParams(
            probability_map_path=probability_map_path,
            ignition_points=ignition_points,
            simulation_hours=simulation_hours,
            output_frequency=1,
            output_dir="outputs/quick_demo"
        )
        
        # Run with simplified rules for speed
        return self.run_simulation(params, use_simplified=True)

# Convenience function for easy access
def run_fire_simulation(probability_map_path: str,
                       ignition_points: List[Tuple[int, int]],
                       simulation_hours: int = 6,
                       output_dir: str = "outputs/simulation",
                       weather_data: Dict = None) -> Dict:
    """
    Convenience function to run fire simulation
    
    Args:
        probability_map_path: Path to ML fire probability map
        ignition_points: List of (x, y) ignition coordinates
        simulation_hours: Duration of simulation
        output_dir: Directory to save results
        weather_data: Weather parameters
        
    Returns:
        Simulation results dictionary
    """
    
    # Create CA engine
    engine = CellularAutomataEngine()
    
    # Create parameters
    params = SimulationParams(
        probability_map_path=probability_map_path,
        ignition_points=ignition_points,
        simulation_hours=simulation_hours,
        output_dir=output_dir,
        weather_data=weather_data or {}
    )
    
    # Run simulation
    return engine.run_simulation(params, use_simplified=True)

if __name__ == "__main__":
    # Example usage
    print("🔥 Forest Fire Cellular Automata Engine")
    print("   This module provides fire spread simulation capabilities")
    print("   Use run_fire_simulation() for quick simulations")
    print("   Or create CellularAutomataEngine() for advanced control")
