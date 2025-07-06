# ====================
# cellular_automata/config.py
# Cellular Automata Configuration for Forest Fire Spread Simulation
# ====================

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class CAConfig:
    """Configuration class for Cellular Automata simulation"""
    
    # Spatial parameters
    resolution: float = 30.0  # meters per pixel
    
    # Temporal parameters
    time_step: float = 1.0  # hours per simulation step
    max_simulation_hours: int = 12  # maximum simulation duration
    
    # Fire spread parameters
    base_spread_rate: float = 0.1  # base probability of spread per hour
    wind_influence: float = 0.3  # wind influence factor (0-1)
    slope_influence: float = 0.2  # slope influence factor (0-1)
    fuel_influence: float = 0.4  # fuel/vegetation influence factor (0-1)
    
    # Weather constants (simplified - daily values)
    default_wind_speed: float = 5.0  # km/h
    default_wind_direction: float = 45.0  # degrees (0=North, 90=East)
    default_humidity: float = 40.0  # percentage
    default_temperature: float = 25.0  # celsius
    
    # Barrier effects
    urban_barrier_strength: float = 0.9  # how much urban areas block fire (0-1)
    water_barrier_strength: float = 1.0  # how much water blocks fire (0-1)
    road_barrier_strength: float = 0.3  # how much roads block fire (0-1)
    
    # Fire states
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    
    # Neighborhood type
    neighborhood_type: str = "moore"  # "moore" (8-cell) or "neumann" (4-cell)
    
    # Performance settings
    use_gpu: bool = True
    batch_processing: bool = True
    chunk_size: int = 1024  # for processing large areas in chunks

@dataclass
class SimulationParams:
    """Parameters for a specific simulation run"""
    
    # Input data paths
    probability_map_path: str
    dem_path: str = None
    lulc_path: str = None
    ghsl_path: str = None
    weather_data: Dict = None
    
    # Ignition parameters
    ignition_points: List[Tuple[int, int]] = None  # [(x, y), ...]
    ignition_intensities: List[float] = None  # fire intensity at each point
    ignition_times: List[float] = None  # ignition time in hours for each point
    
    # Simulation control
    simulation_hours: int = 6
    output_frequency: int = 1  # save output every N hours
    
    # Output settings
    output_dir: str = "outputs/simulations"
    save_intermediate_states: bool = True
    generate_animation: bool = True
    
    def __post_init__(self):
        """Set default values for optional parameters"""
        if self.ignition_points is None:
            self.ignition_points = []
        if self.ignition_intensities is None:
            self.ignition_intensities = [1.0] * len(self.ignition_points)
        if self.ignition_times is None:
            self.ignition_times = [0.0] * len(self.ignition_points)
        if self.weather_data is None:
            self.weather_data = {
                'wind_speed': CAConfig.default_wind_speed,
                'wind_direction': CAConfig.default_wind_direction,
                'humidity': CAConfig.default_humidity,
                'temperature': CAConfig.default_temperature
            }

# Predefined simulation scenarios
SIMULATION_SCENARIOS = {
    "quick_demo": {
        "simulation_hours": 3,
        "output_frequency": 1,
        "description": "Quick 3-hour simulation for demo"
    },
    "short_term": {
        "simulation_hours": 6,
        "output_frequency": 1,
        "description": "Short-term 6-hour prediction"
    },
    "extended": {
        "simulation_hours": 12,
        "output_frequency": 2,
        "description": "Extended 12-hour simulation"
    },
    "detailed": {
        "simulation_hours": 24,
        "output_frequency": 4,
        "description": "Detailed 24-hour simulation"
    }
}

# Wind direction mapping
WIND_DIRECTIONS = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

# Land use classification for fire behavior
LULC_FIRE_BEHAVIOR = {
    # High fire risk
    10: {"flammability": 0.9, "spread_rate": 1.2, "name": "Grassland"},
    20: {"flammability": 0.8, "spread_rate": 1.0, "name": "Shrubland"},
    30: {"flammability": 0.7, "spread_rate": 0.9, "name": "Deciduous Forest"},
    40: {"flammability": 0.6, "spread_rate": 0.8, "name": "Evergreen Forest"},
    
    # Medium fire risk
    50: {"flammability": 0.4, "spread_rate": 0.6, "name": "Mixed Forest"},
    60: {"flammability": 0.3, "spread_rate": 0.5, "name": "Agricultural"},
    
    # Low fire risk
    70: {"flammability": 0.1, "spread_rate": 0.2, "name": "Urban/Built"},
    80: {"flammability": 0.0, "spread_rate": 0.0, "name": "Water"},
    90: {"flammability": 0.0, "spread_rate": 0.0, "name": "Barren/Rock"},
    
    # Default for unknown classes
    0: {"flammability": 0.5, "spread_rate": 0.7, "name": "Unknown"}
}

# Slope effect on fire spread
def get_slope_factor(slope_degrees: float) -> float:
    """
    Calculate slope factor for fire spread
    Fire spreads faster uphill, slower downhill
    """
    if slope_degrees <= 0:
        return 0.8  # slightly slower on flat/downhill
    elif slope_degrees <= 15:
        return 1.0 + (slope_degrees / 15) * 0.5  # gradual increase
    elif slope_degrees <= 30:
        return 1.5 + (slope_degrees - 15) / 15 * 1.0  # steeper increase
    else:
        return 2.5  # maximum multiplier for very steep slopes

# Default configuration instance
DEFAULT_CONFIG = CAConfig()
