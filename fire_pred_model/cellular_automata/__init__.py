# ====================
# cellular_automata/__init__.py
# Forest Fire Cellular Automata Module
# ====================

"""
Forest Fire Spread Simulation using Cellular Automata

This module provides:
- Core CA engine for fire spread simulation
- Integration with ML fire probability predictions  
- Rules engine with simplified and TensorFlow-based implementations
- Web API for interactive simulations
- Utility functions for data processing

Main Components:
- CellularAutomataEngine: Core simulation engine
- MLCAIntegration: Integration layer with ML predictions
- FireSpreadRules: Advanced TensorFlow-based spread rules
- SimplifiedFireRules: Fast numpy-based spread rules

Quick Start:
    from cellular_automata import run_fire_simulation
    
    results = run_fire_simulation(
        probability_map_path="path/to/probability.tif",
        ignition_points=[(50, 50)],
        simulation_hours=6
    )
"""

__version__ = "1.0.0"
__author__ = "Forest Fire Prediction Team"

# Import main functions for easy access
from .core import CellularAutomataEngine, run_fire_simulation
from .integration import MLCAIntegration, quick_integrated_simulation, get_web_simulation_data
from .config import CAConfig, SimulationParams, SIMULATION_SCENARIOS
from .rules import FireSpreadRules, SimplifiedFireRules

# Import utility functions
from .utils import (
    load_raster_data,
    create_ignition_map,
    create_fire_animation_data,
    normalize_array
)

__all__ = [
    # Core classes
    'CellularAutomataEngine',
    'MLCAIntegration', 
    'FireSpreadRules',
    'SimplifiedFireRules',
    
    # Configuration
    'CAConfig',
    'SimulationParams',
    'SIMULATION_SCENARIOS',
    
    # Convenience functions
    'run_fire_simulation',
    'quick_integrated_simulation',
    'get_web_simulation_data',
    
    # Utilities
    'load_raster_data',
    'create_ignition_map', 
    'create_fire_animation_data',
    'normalize_array'
]

# Module metadata
DESCRIPTION = "Cellular Automata engine for forest fire spread simulation"
REQUIREMENTS = [
    "numpy", "tensorflow", "rasterio", "scipy", 
    "flask", "flask-cors"
]
