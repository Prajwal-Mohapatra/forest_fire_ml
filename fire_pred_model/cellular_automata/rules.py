# ====================
# cellular_automata/rules.py
# Fire spread rules for Cellular Automata simulation
# ====================

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Optional
from .config import CAConfig, LULC_FIRE_BEHAVIOR, get_slope_factor

class FireSpreadRules:
    """
    Implements fire spread rules for cellular automata simulation
    """
    
    def __init__(self, config: CAConfig):
        self.config = config
        self.neighborhood_kernel = self._create_neighborhood_kernel()
        
    def _create_neighborhood_kernel(self) -> tf.Tensor:
        """Create TensorFlow kernel for neighborhood operations"""
        if self.config.neighborhood_type == "moore":
            kernel = np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ], dtype=np.float32)
        else:  # neumann
            kernel = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ], dtype=np.float32)
        
        # Reshape for TensorFlow convolution: [height, width, in_channels, out_channels]
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        return tf.constant(kernel)
    
    def apply_spread_rules(self, 
                          fire_state: tf.Tensor,
                          probability_map: tf.Tensor,
                          wind_speed: float,
                          wind_direction: float,
                          dem: Optional[tf.Tensor] = None,
                          lulc: Optional[tf.Tensor] = None,
                          barriers: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply fire spread rules for one time step
        
        Args:
            fire_state: Current fire state [H, W]
            probability_map: Fire probability map [H, W]
            wind_speed: Wind speed in km/h
            wind_direction: Wind direction in degrees
            dem: Digital elevation model [H, W]
            lulc: Land use/land cover map [H, W]
            barriers: Barrier map [H, W] (0=no barrier, 1=complete barrier)
            
        Returns:
            Updated fire state
        """
        # Ensure proper tensor shapes
        if len(fire_state.shape) == 2:
            fire_state = fire_state[tf.newaxis, :, :, tf.newaxis]
        if len(probability_map.shape) == 2:
            probability_map = probability_map[tf.newaxis, :, :, tf.newaxis]
            
        # Calculate neighboring fire influence
        neighbor_influence = self._calculate_neighbor_influence(fire_state, wind_speed, wind_direction)
        
        # Calculate environmental factors
        environmental_factor = self._calculate_environmental_factor(
            probability_map, dem, lulc, barriers
        )
        
        # Combine factors to get spread probability
        spread_probability = self._combine_spread_factors(
            neighbor_influence, environmental_factor, probability_map
        )
        
        # Apply stochastic spread
        new_fire_state = self._apply_stochastic_spread(fire_state, spread_probability)
        
        return tf.squeeze(new_fire_state)
    
    def _calculate_neighbor_influence(self, 
                                    fire_state: tf.Tensor,
                                    wind_speed: float,
                                    wind_direction: float) -> tf.Tensor:
        """Calculate influence from neighboring burning cells"""
        
        # Create wind-biased kernel
        wind_kernel = self._create_wind_biased_kernel(wind_direction, wind_speed)
        
        # Apply convolution to count burning neighbors
        neighbor_count = tf.nn.conv2d(
            fire_state, 
            wind_kernel, 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        )
        
        # Normalize by maximum possible neighbors
        max_neighbors = tf.reduce_sum(wind_kernel)
        neighbor_influence = neighbor_count / max_neighbors
        
        return neighbor_influence
    
    def _create_wind_biased_kernel(self, wind_direction: float, wind_speed: float) -> tf.Tensor:
        """Create wind-biased convolution kernel"""
        base_kernel = self.neighborhood_kernel.numpy().squeeze()
        
        # Wind influence factor based on speed
        wind_factor = min(wind_speed / 20.0, 1.0) * self.config.wind_influence
        
        # Apply directional bias
        wind_rad = np.radians(wind_direction)
        
        # Define relative positions for 3x3 kernel
        positions = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
        
        biased_kernel = base_kernel.copy()
        
        for i, (dy, dx) in enumerate(positions):
            if i == 4:  # Skip center
                continue
                
            row, col = i // 3, i % 3
            
            # Calculate alignment with wind
            pos_angle = np.arctan2(-dy, dx)
            angle_diff = np.abs(np.arctan2(np.sin(pos_angle - wind_rad),
                                         np.cos(pos_angle - wind_rad)))
            alignment = np.cos(angle_diff)
            
            # Apply wind bias
            bias = 1.0 + wind_factor * alignment
            biased_kernel[row, col] *= bias
        
        # Reshape for TensorFlow
        return tf.constant(biased_kernel[:, :, np.newaxis, np.newaxis])
    
    def _calculate_environmental_factor(self,
                                      probability_map: tf.Tensor,
                                      dem: Optional[tf.Tensor] = None,
                                      lulc: Optional[tf.Tensor] = None,
                                      barriers: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Calculate environmental influence on fire spread"""
        
        # Start with base probability
        env_factor = probability_map
        
        # Apply slope effects if DEM available
        if dem is not None:
            slope_factor = self._calculate_slope_factor(dem)
            env_factor = env_factor * slope_factor
        
        # Apply land use effects if LULC available
        if lulc is not None:
            lulc_factor = self._calculate_lulc_factor(lulc)
            env_factor = env_factor * lulc_factor
        
        # Apply barrier effects
        if barriers is not None:
            if len(barriers.shape) == 2:
                barriers = barriers[tf.newaxis, :, :, tf.newaxis]
            barrier_factor = 1.0 - barriers * self.config.urban_barrier_strength
            env_factor = env_factor * barrier_factor
        
        return env_factor
    
    def _calculate_slope_factor(self, dem: tf.Tensor) -> tf.Tensor:
        """Calculate slope influence on fire spread"""
        # Add batch and channel dimensions if needed
        if len(dem.shape) == 2:
            dem = dem[tf.newaxis, :, :, tf.newaxis]
        
        # Calculate gradients
        grad_y, grad_x = tf.image.image_gradients(dem)
        
        # Calculate slope in degrees
        slope_rad = tf.atan(tf.sqrt(grad_x**2 + grad_y**2) / self.config.resolution)
        slope_deg = slope_rad * 180.0 / np.pi
        
        # Apply slope factor function
        # Fire spreads faster uphill, slower downhill
        slope_factor = tf.where(
            slope_deg <= 15.0,
            1.0 + slope_deg / 15.0 * 0.5,  # Gradual increase
            tf.where(
                slope_deg <= 30.0,
                1.5 + (slope_deg - 15.0) / 15.0 * 1.0,  # Steeper increase
                2.5  # Maximum for very steep slopes
            )
        )
        
        return slope_factor * self.config.slope_influence
    
    def _calculate_lulc_factor(self, lulc: tf.Tensor) -> tf.Tensor:
        """Calculate land use influence on fire spread"""
        if len(lulc.shape) == 2:
            lulc = lulc[tf.newaxis, :, :, tf.newaxis]
        
        # Create lookup tensor for LULC values
        lulc_values = tf.constant(list(LULC_FIRE_BEHAVIOR.keys()), dtype=tf.float32)
        flammability_values = tf.constant([
            LULC_FIRE_BEHAVIOR[k]["flammability"] for k in LULC_FIRE_BEHAVIOR.keys()
        ], dtype=tf.float32)
        
        # Find closest LULC class for each pixel
        lulc_flat = tf.reshape(lulc, [-1])
        distances = tf.abs(lulc_flat[:, tf.newaxis] - lulc_values[tf.newaxis, :])
        closest_indices = tf.argmin(distances, axis=1)
        
        # Get flammability values
        flammability_flat = tf.gather(flammability_values, closest_indices)
        flammability = tf.reshape(flammability_flat, tf.shape(lulc))
        
        return flammability * self.config.fuel_influence
    
    def _combine_spread_factors(self,
                              neighbor_influence: tf.Tensor,
                              environmental_factor: tf.Tensor,
                              base_probability: tf.Tensor) -> tf.Tensor:
        """Combine all factors to get final spread probability"""
        
        # Base spread rate
        base_spread = tf.constant(self.config.base_spread_rate)
        
        # Combine factors multiplicatively
        combined_probability = (
            base_spread * 
            (1.0 + neighbor_influence) * 
            environmental_factor * 
            base_probability
        )
        
        # Ensure probability stays in [0, 1] range
        return tf.clip_by_value(combined_probability, 0.0, 1.0)
    
    def _apply_stochastic_spread(self,
                               current_state: tf.Tensor,
                               spread_probability: tf.Tensor) -> tf.Tensor:
        """Apply stochastic fire spread based on probabilities"""
        
        # Generate random values for stochastic spread
        random_values = tf.random.uniform(tf.shape(spread_probability))
        
        # Determine new ignitions
        new_ignitions = tf.cast(random_values < spread_probability, tf.float32)
        
        # Current burning areas remain burning (simplified - no burnout)
        current_burning = tf.cast(current_state > 0.1, tf.float32)
        
        # Combine current and new fire
        new_state = tf.maximum(current_burning, new_ignitions)
        
        return new_state

class SimplifiedFireRules:
    """
    Simplified fire spread rules for rapid prototyping
    """
    
    def __init__(self, config: CAConfig):
        self.config = config
    
    def simple_spread(self,
                     fire_state: np.ndarray,
                     probability_map: np.ndarray,
                     wind_direction: float = 0.0) -> np.ndarray:
        """
        Simple fire spread using numpy operations
        
        Args:
            fire_state: Current fire state
            probability_map: Fire probability map
            wind_direction: Wind direction in degrees
            
        Returns:
            Updated fire state
        """
        from scipy import ndimage
        
        # Create simple kernel
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        # Apply wind bias (simplified)
        if wind_direction >= 0 and wind_direction < 90:  # NE quadrant
            kernel[0, 2] *= 1.5  # Boost NE
            kernel[2, 0] *= 0.5  # Reduce SW
        # Add other quadrants as needed...
        
        # Calculate neighbor influence
        neighbor_influence = ndimage.convolve(fire_state, kernel, mode='constant')
        
        # Combine with probability map
        spread_probability = (
            self.config.base_spread_rate * 
            probability_map * 
            (1.0 + neighbor_influence)
        )
        
        # Apply stochastic spread
        random_values = np.random.random(fire_state.shape)
        new_ignitions = (random_values < spread_probability).astype(np.float32)
        
        # Keep existing fire and add new ignitions
        new_state = np.maximum(fire_state, new_ignitions)
        
        return new_state
