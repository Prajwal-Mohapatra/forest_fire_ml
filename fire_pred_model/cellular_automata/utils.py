# ====================
# cellular_automata/utils.py
# Utility functions for Cellular Automata fire spread simulation
# ====================

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os
from typing import Tuple, List, Dict, Optional
import tensorflow as tf

def load_raster_data(file_path: str, target_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, dict]:
    """
    Load raster data from TIF file with optional resizing
    
    Args:
        file_path: Path to the raster file
        target_shape: Optional (height, width) to resize to
        
    Returns:
        Tuple of (data_array, metadata)
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'shape': data.shape,
                'nodata': src.nodata
            }
            
            # Handle nodata values
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            
            # Resize if target shape specified
            if target_shape and data.shape != target_shape:
                data = resize_array(data, target_shape)
                metadata['shape'] = target_shape
                
        return data, metadata
        
    except Exception as e:
        print(f"Error loading raster {file_path}: {str(e)}")
        raise e

def resize_array(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize array using TensorFlow for GPU acceleration
    
    Args:
        array: Input array to resize
        target_shape: Target (height, width)
        
    Returns:
        Resized array
    """
    if array.shape == target_shape:
        return array
        
    # Use TensorFlow for resizing
    tf_array = tf.constant(array[np.newaxis, :, :, np.newaxis])
    resized = tf.image.resize(tf_array, target_shape, method='bilinear')
    return resized[0, :, :, 0].numpy()

def create_neighborhood_kernel(neighborhood_type: str = "moore") -> np.ndarray:
    """
    Create convolution kernel for neighborhood analysis
    
    Args:
        neighborhood_type: "moore" (8-connected) or "neumann" (4-connected)
        
    Returns:
        Kernel for convolution operations
    """
    if neighborhood_type.lower() == "moore":
        # 8-connected neighborhood (Moore)
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.float32)
    elif neighborhood_type.lower() == "neumann":
        # 4-connected neighborhood (von Neumann)
        kernel = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")
    
    return kernel

def apply_wind_directional_bias(base_kernel: np.ndarray, wind_direction: float, wind_strength: float = 0.3) -> np.ndarray:
    """
    Apply wind directional bias to the neighborhood kernel
    
    Args:
        base_kernel: Base neighborhood kernel
        wind_direction: Wind direction in degrees (0=North, 90=East)
        wind_strength: How much wind affects spread (0-1)
        
    Returns:
        Modified kernel with wind bias
    """
    kernel = base_kernel.copy()
    
    # Convert wind direction to radians
    wind_rad = np.radians(wind_direction)
    
    # Define relative positions for 3x3 kernel
    positions = [
        (-1, -1), (-1, 0), (-1, 1),  # Top row
        ( 0, -1), ( 0, 0), ( 0, 1),  # Middle row
        ( 1, -1), ( 1, 0), ( 1, 1)   # Bottom row
    ]
    
    # Apply wind bias to each position
    for i, (dy, dx) in enumerate(positions):
        if i == 4:  # Skip center cell
            continue
            
        row, col = i // 3, i % 3
        
        # Calculate angle from center to this position
        pos_angle = np.arctan2(-dy, dx)  # Negative dy because y increases downward
        
        # Calculate alignment with wind direction
        angle_diff = np.abs(np.arctan2(np.sin(pos_angle - wind_rad), 
                                      np.cos(pos_angle - wind_rad)))
        
        # Apply bias (stronger for aligned directions)
        alignment = np.cos(angle_diff)  # 1 for aligned, -1 for opposite
        bias_factor = 1.0 + wind_strength * alignment
        
        kernel[row, col] *= bias_factor
    
    return kernel

def calculate_slope_and_aspect(dem: np.ndarray, resolution: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate slope and aspect from DEM using TensorFlow
    
    Args:
        dem: Digital elevation model
        resolution: Pixel resolution in meters
        
    Returns:
        Tuple of (slope_degrees, aspect_degrees)
    """
    # Convert to TensorFlow tensor
    dem_tf = tf.constant(dem, dtype=tf.float32)
    dem_tf = dem_tf[tf.newaxis, :, :, tf.newaxis]  # Add batch and channel dims
    
    # Calculate gradients
    grad_y, grad_x = tf.image.image_gradients(dem_tf)
    
    # Remove extra dimensions
    grad_x = grad_x[0, :, :, 0] / resolution
    grad_y = grad_y[0, :, :, 0] / resolution
    
    # Calculate slope in radians then convert to degrees
    slope_rad = tf.atan(tf.sqrt(grad_x**2 + grad_y**2))
    slope_deg = slope_rad * 180.0 / np.pi
    
    # Calculate aspect (direction of steepest descent)
    aspect_rad = tf.atan2(-grad_y, -grad_x)  # Negative for geographic convention
    aspect_deg = aspect_rad * 180.0 / np.pi
    
    # Convert aspect to 0-360 range
    aspect_deg = tf.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
    
    return slope_deg.numpy(), aspect_deg.numpy()

def apply_barriers(fire_state: np.ndarray, barrier_map: np.ndarray, barrier_strength: float = 0.9) -> np.ndarray:
    """
    Apply barrier effects to fire spread
    
    Args:
        fire_state: Current fire state array
        barrier_map: Barrier strength map (0-1, where 1 = complete barrier)
        barrier_strength: Global barrier strength multiplier
        
    Returns:
        Modified fire state with barrier effects
    """
    # Reduce fire probability where barriers exist
    barrier_effect = 1.0 - (barrier_map * barrier_strength)
    return fire_state * barrier_effect

def create_ignition_map(shape: Tuple[int, int], ignition_points: List[Tuple[int, int]], 
                       intensities: List[float] = None) -> np.ndarray:
    """
    Create ignition map from point locations
    
    Args:
        shape: (height, width) of the map
        ignition_points: List of (x, y) coordinates
        intensities: Fire intensity at each point (default: 1.0)
        
    Returns:
        Ignition map array
    """
    ignition_map = np.zeros(shape, dtype=np.float32)
    
    if intensities is None:
        intensities = [1.0] * len(ignition_points)
    
    for (x, y), intensity in zip(ignition_points, intensities):
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            ignition_map[y, x] = intensity
            
            # Optional: Add small radius around ignition point
            radius = 2
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= radius:
                            ignition_map[ny, nx] = max(ignition_map[ny, nx], 
                                                     intensity * (1.0 - dist / radius))
    
    return ignition_map

def save_simulation_frame(fire_state: np.ndarray, probability_map: np.ndarray, 
                         output_path: str, metadata: dict, time_step: int = 0):
    """
    Save simulation frame as GeoTIFF
    
    Args:
        fire_state: Current fire state
        probability_map: Original probability map for reference
        output_path: Output file path
        metadata: Geospatial metadata
        time_step: Current time step
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data for saving (combine fire state and probability)
    output_data = np.stack([fire_state, probability_map], axis=0)
    
    # Update metadata for multi-band output
    profile = {
        'driver': 'GTiff',
        'height': fire_state.shape[0],
        'width': fire_state.shape[1],
        'count': 2,  # fire_state + probability
        'dtype': rasterio.float32,
        'crs': metadata.get('crs'),
        'transform': metadata.get('transform'),
        'compress': 'lzw'
    }
    
    # Add time step to filename
    base_name = output_path.replace('.tif', f'_t{time_step:03d}.tif')
    
    with rasterio.open(base_name, 'w', **profile) as dst:
        dst.write(output_data)
        
        # Add band descriptions
        dst.set_band_description(1, f'Fire_State_Hour_{time_step}')
        dst.set_band_description(2, 'Fire_Probability')

def normalize_array(array: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Normalize array to specified range
    
    Args:
        array: Input array
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        
    Returns:
        Normalized array
    """
    arr_min, arr_max = np.nanmin(array), np.nanmax(array)
    if arr_max > arr_min:
        normalized = (array - arr_min) / (arr_max - arr_min)
        return normalized * (max_val - min_val) + min_val
    else:
        return np.full_like(array, min_val)

def create_fire_animation_data(simulation_frames: List[np.ndarray], 
                              metadata: dict) -> Dict:
    """
    Prepare data for web animation
    
    Args:
        simulation_frames: List of fire state arrays
        metadata: Geospatial metadata
        
    Returns:
        Dictionary with animation data
    """
    animation_data = {
        'frames': [],
        'bounds': metadata.get('bounds'),
        'shape': simulation_frames[0].shape if simulation_frames else None,
        'frame_count': len(simulation_frames)
    }
    
    for i, frame in enumerate(simulation_frames):
        # Convert to format suitable for web display
        frame_data = {
            'time_step': i,
            'fire_pixels': np.where(frame > 0.1),  # Locations with fire
            'fire_intensity': frame[frame > 0.1].tolist(),  # Intensity values
            'max_intensity': float(np.max(frame)),
            'total_burned_area': float(np.sum(frame > 0.1))  # Number of burning pixels
        }
        animation_data['frames'].append(frame_data)
    
    return animation_data
