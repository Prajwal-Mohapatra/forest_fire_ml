# ====================
# cellular_automata/integration.py
# Integration layer between ML predictions and CA simulation
# ====================

import os
import glob
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from .core import CellularAutomataEngine, run_fire_simulation
from .config import SimulationParams, CAConfig
from ..predict import predict_fire_probability


class MLCAIntegration:
    """
    Integration layer that connects ML fire prediction with CA simulation
    """
    
    def __init__(self, 
                 model_path: str,
                 data_directory: str,
                 output_directory: str = "outputs/integrated_simulation"):
        
        self.model_path = model_path
        self.data_directory = data_directory
        self.output_directory = output_directory
        self.ca_engine = CellularAutomataEngine()
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        print(f"🔗 ML-CA Integration initialized")
        print(f"   Model: {model_path}")
        print(f"   Data: {data_directory}")
        print(f"   Output: {output_directory}")
    
    def get_available_dates(self) -> List[str]:
        """
        Get list of available dates from the stacked dataset
        
        Returns:
            List of date strings in YYYY_MM_DD format
        """
        pattern = os.path.join(self.data_directory, "stack_2016_*.tif")
        files = glob.glob(pattern)
        
        dates = []
        for file in files:
            # Extract date from filename: stack_2016_04_15.tif -> 2016_04_15
            basename = os.path.basename(file)
            if basename.startswith("stack_"):
                date_part = basename.replace("stack_", "").replace(".tif", "")
                dates.append(date_part)
        
        return sorted(dates)
    
    def generate_daily_prediction(self, date: str) -> str:
        """
        Generate fire probability map for a specific date
        
        Args:
            date: Date string in YYYY_MM_DD format
            
        Returns:
            Path to generated probability map
        """
        # Find the input stack file for this date
        input_file = os.path.join(self.data_directory, f"stack_{date}.tif")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input stack file not found: {input_file}")
        
        # Generate output path
        output_file = os.path.join(self.output_directory, f"probability_{date}.tif")
        
        print(f"🧠 Generating ML prediction for {date}...")
        
        # Generate prediction using the ML model
        probability_map = predict_fire_probability(
            self.model_path,
            input_file,
            output_file,
            patch_size=256,
            overlap=64
        )
        
        print(f"✅ Prediction saved: {output_file}")
        return output_file
    
    def run_integrated_simulation(self,
                                 date: str,
                                 ignition_points: List[Tuple[int, int]],
                                 simulation_hours: int = 6,
                                 weather_data: Dict = None,
                                 include_terrain: bool = True) -> Dict:
        """
        Run complete integrated simulation: ML prediction + CA spread
        
        Args:
            date: Date for simulation (YYYY_MM_DD format)
            ignition_points: List of (x, y) ignition coordinates
            simulation_hours: Duration of fire spread simulation
            weather_data: Weather parameters for simulation
            include_terrain: Whether to include DEM/LULC data
            
        Returns:
            Complete simulation results
        """
        print(f"🔥 Running integrated simulation for {date}")
        
        # Step 1: Generate ML prediction
        probability_map_path = self.generate_daily_prediction(date)
        
        # Step 2: Prepare additional data paths
        additional_data = {}
        if include_terrain:
            # Look for DEM and LULC in the stack (assuming they're separate files)
            dem_pattern = os.path.join(self.data_directory, "*DEM*.tif")
            lulc_pattern = os.path.join(self.data_directory, "*LULC*.tif")
            ghsl_pattern = os.path.join(self.data_directory, "*GHSL*.tif")
            
            dem_files = glob.glob(dem_pattern)
            lulc_files = glob.glob(lulc_pattern)
            ghsl_files = glob.glob(ghsl_pattern)
            
            if dem_files:
                additional_data['dem_path'] = dem_files[0]
            if lulc_files:
                additional_data['lulc_path'] = lulc_files[0]
            if ghsl_files:
                additional_data['ghsl_path'] = ghsl_files[0]
        
        # Step 3: Create simulation parameters
        simulation_output_dir = os.path.join(self.output_directory, f"simulation_{date}")
        
        params = SimulationParams(
            probability_map_path=probability_map_path,
            ignition_points=ignition_points,
            simulation_hours=simulation_hours,
            output_dir=simulation_output_dir,
            weather_data=weather_data,
            **additional_data
        )
        
        # Step 4: Run CA simulation
        print(f"🔄 Running CA simulation...")
        ca_results = self.ca_engine.run_simulation(params, use_simplified=True)
        
        # Step 5: Combine results
        integrated_results = {
            'date': date,
            'ml_prediction_path': probability_map_path,
            'ca_simulation': ca_results,
            'parameters': {
                'ignition_points': ignition_points,
                'simulation_hours': simulation_hours,
                'weather_data': weather_data,
                'include_terrain': include_terrain
            },
            'output_directory': simulation_output_dir
        }
        
        # Save integrated results
        self._save_integrated_results(integrated_results, date)
        
        print(f"✅ Integrated simulation complete!")
        return integrated_results
    
    def run_multiple_scenarios(self,
                              date: str,
                              scenario_configs: List[Dict],
                              base_simulation_hours: int = 6) -> Dict:
        """
        Run multiple fire scenarios for comparison
        
        Args:
            date: Date for simulation
            scenario_configs: List of scenario configurations
            base_simulation_hours: Base simulation duration
            
        Returns:
            Results for all scenarios
        """
        print(f"🎭 Running multiple scenarios for {date}")
        
        # Generate ML prediction once
        probability_map_path = self.generate_daily_prediction(date)
        
        scenario_results = {}
        
        for i, config in enumerate(scenario_configs):
            print(f"   Scenario {i+1}/{len(scenario_configs)}: {config.get('name', f'Scenario_{i+1}')}")
            
            scenario_output_dir = os.path.join(
                self.output_directory, 
                f"scenarios_{date}", 
                f"scenario_{i+1}"
            )
            
            params = SimulationParams(
                probability_map_path=probability_map_path,
                ignition_points=config['ignition_points'],
                simulation_hours=config.get('simulation_hours', base_simulation_hours),
                output_dir=scenario_output_dir,
                weather_data=config.get('weather_data'),
            )
            
            ca_results = self.ca_engine.run_simulation(params, use_simplified=True)
            
            scenario_results[f"scenario_{i+1}"] = {
                'config': config,
                'results': ca_results,
                'output_dir': scenario_output_dir
            }
        
        # Save comparison data
        comparison_results = {
            'date': date,
            'ml_prediction_path': probability_map_path,
            'scenarios': scenario_results,
            'summary': self._create_scenario_summary(scenario_results)
        }
        
        comparison_path = os.path.join(
            self.output_directory, 
            f"scenarios_{date}", 
            "comparison_results.json"
        )
        
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"✅ Multiple scenarios complete! Results saved to {comparison_path}")
        return comparison_results
    
    def _create_scenario_summary(self, scenario_results: Dict) -> Dict:
        """Create summary comparison of scenarios"""
        summary = {
            'total_scenarios': len(scenario_results),
            'scenario_comparison': []
        }
        
        for scenario_id, scenario_data in scenario_results.items():
            stats = scenario_data['results']['statistics']
            final_stats = stats['frame_statistics'][-1] if stats['frame_statistics'] else {}
            
            scenario_summary = {
                'scenario_id': scenario_id,
                'total_burned_area_hectares': final_stats.get('burned_area_hectares', 0),
                'max_fire_intensity': final_stats.get('max_intensity', 0),
                'total_fire_pixels': final_stats.get('total_fire_pixels', 0),
                'ignition_points_count': len(scenario_data['config']['ignition_points'])
            }
            summary['scenario_comparison'].append(scenario_summary)
        
        return summary
    
    def _save_integrated_results(self, results: Dict, date: str):
        """Save integrated simulation results"""
        output_file = os.path.join(self.output_directory, f"integrated_results_{date}.json")
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Integrated results saved: {output_file}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return "numpy_array"  # Don't serialize large arrays
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    def create_web_api_data(self, date: str, results: Dict) -> Dict:
        """
        Create data structure optimized for web API consumption
        
        Args:
            date: Simulation date
            results: Simulation results
            
        Returns:
            Web-optimized data structure
        """
        ca_results = results['ca_simulation']
        
        # Extract animation frames
        frames_data = []
        for i, frame in enumerate(ca_results['frames']):
            # Get fire pixel locations and intensities
            fire_mask = frame > 0.1
            fire_coords = np.where(fire_mask)
            
            if len(fire_coords[0]) > 0:
                frame_data = {
                    'time_step': i,
                    'fire_locations': {
                        'x': fire_coords[1].tolist(),  # x coordinates
                        'y': fire_coords[0].tolist(),  # y coordinates
                        'intensity': frame[fire_mask].tolist()
                    },
                    'total_pixels': len(fire_coords[0]),
                    'max_intensity': float(np.max(frame))
                }
            else:
                frame_data = {
                    'time_step': i,
                    'fire_locations': {'x': [], 'y': [], 'intensity': []},
                    'total_pixels': 0,
                    'max_intensity': 0.0
                }
            
            frames_data.append(frame_data)
        
        # Create web API structure
        web_data = {
            'simulation_info': {
                'date': date,
                'total_frames': len(frames_data),
                'simulation_hours': results['parameters']['simulation_hours'],
                'ignition_points': results['parameters']['ignition_points']
            },
            'animation_frames': frames_data,
            'metadata': {
                'bounds': ca_results['metadata']['bounds'],
                'shape': ca_results['metadata']['shape'],
                'resolution_meters': 30
            },
            'statistics': ca_results['statistics']
        }
        
        return web_data


# Convenience functions for easy integration

def quick_integrated_simulation(model_path: str,
                               data_directory: str,
                               date: str,
                               ignition_points: List[Tuple[int, int]],
                               simulation_hours: int = 6) -> Dict:
    """
    Quick function for running integrated ML + CA simulation
    
    Args:
        model_path: Path to trained ML model
        data_directory: Directory containing stacked data
        date: Date for simulation (YYYY_MM_DD)
        ignition_points: Ignition point coordinates
        simulation_hours: Simulation duration
        
    Returns:
        Simulation results
    """
    integration = MLCAIntegration(model_path, data_directory)
    return integration.run_integrated_simulation(
        date, ignition_points, simulation_hours
    )

def get_web_simulation_data(model_path: str,
                           data_directory: str,
                           date: str,
                           ignition_points: List[Tuple[int, int]],
                           simulation_hours: int = 6) -> Dict:
    """
    Get simulation data formatted for web consumption
    
    Args:
        model_path: Path to trained ML model
        data_directory: Directory containing stacked data
        date: Date for simulation (YYYY_MM_DD)
        ignition_points: Ignition point coordinates
        simulation_hours: Simulation duration
        
    Returns:
        Web-optimized simulation data
    """
    integration = MLCAIntegration(model_path, data_directory)
    results = integration.run_integrated_simulation(
        date, ignition_points, simulation_hours
    )
    
    return integration.create_web_api_data(date, results)


if __name__ == "__main__":
    print("🔗 ML-CA Integration Module")
    print("   Use quick_integrated_simulation() for simple runs")
    print("   Or create MLCAIntegration() for advanced control")
