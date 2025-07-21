# 📊 Data Pipeline Documentation

## Overview

The data pipeline processes multi-source environmental data to create ML-ready datasets and supports the complete workflow from raw satellite imagery to fire spread simulation outputs. This pipeline handles temporal alignment, spatial resampling, and format standardization across diverse geospatial datasets.

## Data Sources and Characteristics

### Primary Environmental Datasets

#### 1. Digital Elevation Model (DEM)
- **Source**: SRTM (Shuttle Radar Topography Mission)
- **Resolution**: 30m spatial resolution
- **Coverage**: Global coverage including Uttarakhand
- **Format**: GeoTIFF, single band
- **Data Type**: Int16 (elevation in meters)
- **Temporal**: Static (one-time acquisition)
- **Processing**: Direct use with derived slope/aspect calculations

```python
# DEM processing example
def process_dem_data(dem_path, output_dir):
    """Process DEM to create elevation, slope, and aspect layers"""
    
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    
    # Calculate slope (in degrees)
    slope = calculate_slope(elevation, transform)
    
    # Calculate aspect (in degrees, 0-360)
    aspect = calculate_aspect(elevation, transform)
    
    # Save processed layers
    save_geotiff(elevation, os.path.join(output_dir, 'elevation.tif'), transform, crs)
    save_geotiff(slope, os.path.join(output_dir, 'slope.tif'), transform, crs)
    save_geotiff(aspect, os.path.join(output_dir, 'aspect.tif'), transform, crs)
    
    return elevation, slope, aspect
```

#### 2. Weather Data (ERA5 Daily)
- **Source**: Copernicus ERA5 Reanalysis
- **Resolution**: 0.25° (~25km) spatial, daily temporal
- **Variables**: Temperature, humidity, wind speed, precipitation
- **Format**: NetCDF, multi-variable
- **Temporal Coverage**: 1979-present (using 2016 for this project)
- **Processing**: Spatial downscaling to 30m, temporal interpolation

```python
# ERA5 data processing
def process_era5_daily(era5_path, date, target_bounds, target_shape):
    """Process ERA5 daily data for specific date and region"""
    
    with xarray.open_dataset(era5_path) as ds:
        # Select date and variables
        daily_data = ds.sel(time=date)
        
        variables = {
            'temperature_2m_max': daily_data.mx2t,      # Maximum temperature
            'relative_humidity_min': daily_data.mn2r,   # Minimum relative humidity  
            'wind_speed_max': daily_data.mx10fg,        # Maximum wind speed
            'precipitation_total': daily_data.tp         # Total precipitation
        }
        
        processed_vars = {}
        for var_name, var_data in variables.items():
            # Crop to region of interest
            cropped = var_data.sel(
                latitude=slice(target_bounds[3], target_bounds[1]),
                longitude=slice(target_bounds[0], target_bounds[2])
            )
            
            # Resample to target resolution (30m)
            resampled = resample_to_target_grid(cropped, target_shape)
            processed_vars[var_name] = resampled
        
        return processed_vars
```

#### 3. Land Use/Land Cover (LULC 2020)
- **Source**: ESA WorldCover or similar
- **Resolution**: 10m native, resampled to 30m
- **Classes**: Forest, agriculture, urban, water, etc.
- **Format**: GeoTIFF, single band with class codes
- **Temporal**: Annual (using 2020 as proxy for 2016)
- **Processing**: Resampling and reclassification for fire fuel types

```python
# LULC processing and fuel type mapping
def process_lulc_data(lulc_path, output_path):
    """Process LULC data and map to fire fuel types"""
    
    # LULC class to fuel type mapping
    fuel_type_mapping = {
        10: 4,    # Tree cover -> High fuel load
        20: 3,    # Shrubland -> Medium fuel load  
        30: 2,    # Grassland -> Low fuel load
        40: 1,    # Cropland -> Very low fuel load
        50: 0,    # Built-up -> No fuel
        60: 0,    # Bare/sparse -> No fuel
        70: 0,    # Snow/ice -> No fuel
        80: 0,    # Water -> No fuel
        90: 0,    # Wetlands -> No fuel
        95: 1     # Mangroves -> Low fuel load
    }
    
    with rasterio.open(lulc_path) as src:
        lulc_data = src.read(1)
        profile = src.profile
    
    # Map LULC classes to fuel types
    fuel_types = np.zeros_like(lulc_data, dtype=np.uint8)
    for lulc_class, fuel_type in fuel_type_mapping.items():
        fuel_types[lulc_data == lulc_class] = fuel_type
    
    # Save fuel type map
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(fuel_types, 1)
    
    return fuel_types
```

#### 4. Human Settlement (GHSL 2015)
- **Source**: Global Human Settlement Layer
- **Resolution**: 30m
- **Data**: Built-up density (0-100%)
- **Format**: GeoTIFF, single band
- **Temporal**: Multi-temporal, using 2015 epoch
- **Processing**: Direct use as suppression factor

#### 5. Fire History (VIIRS Active Fire)
- **Source**: NASA VIIRS fire detection
- **Resolution**: 375m native, processed to 30m
- **Temporal**: Daily observations
- **Format**: Point data, converted to raster
- **Processing**: Temporal aggregation, spatial interpolation

```python
# VIIRS fire data processing
def process_viirs_fire_data(viirs_csv_path, date, bounds, target_shape):
    """Process VIIRS active fire detections for specific date"""
    
    # Load VIIRS data
    viirs_df = pd.read_csv(viirs_csv_path)
    viirs_df['acq_date'] = pd.to_datetime(viirs_df['acq_date'])
    
    # Filter by date and location
    daily_fires = viirs_df[
        (viirs_df['acq_date'] == pd.to_datetime(date)) &
        (viirs_df['latitude'] >= bounds[1]) & (viirs_df['latitude'] <= bounds[3]) &
        (viirs_df['longitude'] >= bounds[0]) & (viirs_df['longitude'] <= bounds[2])
    ]
    
    # Create fire mask raster
    fire_mask = np.zeros(target_shape, dtype=np.uint8)
    
    for _, fire in daily_fires.iterrows():
        # Convert lat/lon to pixel coordinates
        pixel_x, pixel_y = latlon_to_pixel(
            fire['latitude'], fire['longitude'], bounds, target_shape
        )
        
        if 0 <= pixel_x < target_shape[1] and 0 <= pixel_y < target_shape[0]:
            fire_mask[pixel_y, pixel_x] = 1
    
    # Apply Gaussian smoothing to create probability-like surface
    fire_probability = gaussian_filter(fire_mask.astype(np.float32), sigma=2)
    fire_probability = np.clip(fire_probability, 0, 1)
    
    return fire_probability
```

## Data Stacking and Alignment

### Spatial Alignment Process

```python
def align_all_datasets(datasets_config, target_date, output_path):
    """
    Align all environmental datasets to common grid
    
    Target Grid Specifications:
    - Spatial Resolution: 30m
    - Projection: WGS84 (EPSG:4326)
    - Bounds: Uttarakhand state boundaries
    - Extent: ~400x500 km approximately
    """
    
    # Define target grid parameters
    target_bounds = (77.8, 28.6, 81.1, 31.1)  # (min_lon, min_lat, max_lon, max_lat)
    target_crs = 'EPSG:4326'
    target_resolution = 0.000277778  # ~30m in degrees
    
    # Calculate target shape
    width = int((target_bounds[2] - target_bounds[0]) / target_resolution)
    height = int((target_bounds[3] - target_bounds[1]) / target_resolution)
    target_shape = (height, width)
    
    # Create target transform
    target_transform = rasterio.transform.from_bounds(
        *target_bounds, width, height
    )
    
    # Process each dataset
    aligned_bands = []
    band_names = []
    
    # 1. DEM (elevation, slope, aspect)
    dem_data = process_dem_data(datasets_config['dem_path'], target_bounds, target_shape)
    aligned_bands.extend([dem_data['elevation'], dem_data['slope'], dem_data['aspect']])
    band_names.extend(['elevation', 'slope', 'aspect'])
    
    # 2. Weather data (temperature, humidity, wind, precipitation)
    weather_data = process_era5_daily(
        datasets_config['era5_path'], target_date, target_bounds, target_shape
    )
    aligned_bands.extend(list(weather_data.values()))
    band_names.extend(list(weather_data.keys()))
    
    # 3. Land cover (fuel types)
    lulc_data = process_lulc_data(datasets_config['lulc_path'], target_bounds, target_shape)
    aligned_bands.append(lulc_data)
    band_names.append('fuel_type')
    
    # 4. Human settlement
    ghsl_data = process_ghsl_data(datasets_config['ghsl_path'], target_bounds, target_shape)
    aligned_bands.append(ghsl_data)
    band_names.append('settlement_density')
    
    # Stack all bands into single array
    stacked_array = np.stack(aligned_bands, axis=0)
    
    # Save stacked dataset
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': len(aligned_bands),
        'crs': target_crs,
        'transform': target_transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i, band in enumerate(aligned_bands):
            dst.write(band.astype(np.float32), i + 1)
        
        # Add band descriptions
        for i, name in enumerate(band_names):
            dst.set_band_description(i + 1, name)
    
    print(f"✅ Stacked dataset saved: {output_path}")
    print(f"   Shape: {stacked_array.shape}")
    print(f"   Bands: {', '.join(band_names)}")
    
    return stacked_array, band_names
```

### Temporal Processing Workflow

```python
def create_temporal_dataset(start_date, end_date, datasets_config, output_dir):
    """
    Create time series of stacked datasets for training period
    
    Processes daily datasets from April 1 to May 29, 2016
    """
    
    date_range = pd.date_range(start_date, end_date, freq='D')
    successful_dates = []
    
    for date in date_range:
        date_str = date.strftime('%Y_%m_%d')
        output_path = os.path.join(output_dir, f'stack_{date_str}.tif')
        
        try:
            # Check if VIIRS fire data available for this date
            fire_data_available = check_viirs_availability(date, datasets_config)
            
            if fire_data_available:
                stacked_data, band_names = align_all_datasets(
                    datasets_config, date, output_path
                )
                successful_dates.append(date_str)
                print(f"✅ Processed: {date_str}")
            else:
                print(f"⚠️ Skipped: {date_str} (no fire data)")
                
        except Exception as e:
            print(f"❌ Failed: {date_str} - {str(e)}")
    
    print(f"\n📊 Dataset Creation Summary:")
    print(f"   Total dates processed: {len(successful_dates)}")
    print(f"   Date range: {successful_dates[0]} to {successful_dates[-1]}")
    print(f"   Output directory: {output_dir}")
    
    # Create metadata file
    metadata = {
        'dataset_info': {
            'date_range': [start_date.isoformat(), end_date.isoformat()],
            'successful_dates': successful_dates,
            'total_files': len(successful_dates),
            'band_names': band_names
        },
        'spatial_info': {
            'bounds': (77.8, 28.6, 81.1, 31.1),
            'crs': 'EPSG:4326',
            'resolution_meters': 30,
            'grid_shape': [height, width]
        },
        'data_sources': datasets_config
    }
    
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return successful_dates
```

## Data Quality Control

### Validation and Quality Checks

```python
def validate_stacked_dataset(dataset_path):
    """
    Comprehensive quality validation for stacked datasets
    """
    
    validation_results = {
        'file_readable': False,
        'expected_bands': False,
        'spatial_consistency': False,
        'value_ranges_valid': False,
        'missing_data_acceptable': False,
        'overall_valid': False
    }
    
    try:
        with rasterio.open(dataset_path) as src:
            # Check basic file properties
            validation_results['file_readable'] = True
            
            # Check band count
            expected_bands = 9  # DEM(3) + Weather(4) + LULC(1) + GHSL(1)
            validation_results['expected_bands'] = (src.count == expected_bands)
            
            # Check spatial properties
            expected_crs = 'EPSG:4326'
            expected_resolution = abs(src.transform[0])  # Pixel size
            
            validation_results['spatial_consistency'] = (
                str(src.crs) == expected_crs and
                0.0002 < expected_resolution < 0.0003  # ~30m in degrees
            )
            
            # Validate data ranges for each band
            band_ranges = {
                1: (0, 8848),      # Elevation (m)
                2: (0, 90),        # Slope (degrees)
                3: (0, 360),       # Aspect (degrees)
                4: (-40, 50),      # Temperature (°C)
                5: (0, 100),       # Humidity (%)
                6: (0, 50),        # Wind speed (m/s)
                7: (0, 100),       # Precipitation (mm)
                8: (0, 4),         # Fuel type (categorical)
                9: (0, 100)        # Settlement density (%)
            }
            
            ranges_valid = True
            missing_data_stats = []
            
            for band_num, (min_val, max_val) in band_ranges.items():
                band_data = src.read(band_num)
                
                # Check value ranges
                actual_min, actual_max = np.nanmin(band_data), np.nanmax(band_data)
                if not (min_val <= actual_min and actual_max <= max_val):
                    ranges_valid = False
                    print(f"⚠️ Band {band_num}: values [{actual_min:.2f}, {actual_max:.2f}] outside expected [{min_val}, {max_val}]")
                
                # Check missing data percentage
                missing_pct = np.isnan(band_data).sum() / band_data.size * 100
                missing_data_stats.append(missing_pct)
                
                if missing_pct > 10:  # More than 10% missing
                    print(f"⚠️ Band {band_num}: {missing_pct:.1f}% missing data")
            
            validation_results['value_ranges_valid'] = ranges_valid
            validation_results['missing_data_acceptable'] = all(pct < 10 for pct in missing_data_stats)
            
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return validation_results
    
    # Overall validation
    validation_results['overall_valid'] = all([
        validation_results['file_readable'],
        validation_results['expected_bands'],
        validation_results['spatial_consistency'],
        validation_results['value_ranges_valid'],
        validation_results['missing_data_acceptable']
    ])
    
    return validation_results

def batch_validate_datasets(dataset_dir):
    """Validate all stacked datasets in directory"""
    
    dataset_files = [f for f in os.listdir(dataset_dir) if f.startswith('stack_') and f.endswith('.tif')]
    
    validation_summary = {
        'total_files': len(dataset_files),
        'valid_files': 0,
        'invalid_files': [],
        'validation_details': {}
    }
    
    for dataset_file in dataset_files:
        dataset_path = os.path.join(dataset_dir, dataset_file)
        validation_result = validate_stacked_dataset(dataset_path)
        
        validation_summary['validation_details'][dataset_file] = validation_result
        
        if validation_result['overall_valid']:
            validation_summary['valid_files'] += 1
            print(f"✅ {dataset_file}: Valid")
        else:
            validation_summary['invalid_files'].append(dataset_file)
            print(f"❌ {dataset_file}: Invalid")
    
    print(f"\n📊 Validation Summary:")
    print(f"   Total files: {validation_summary['total_files']}")
    print(f"   Valid files: {validation_summary['valid_files']}")
    print(f"   Invalid files: {len(validation_summary['invalid_files'])}")
    
    return validation_summary
```

## Data Access and Serving

### Dataset Management Class

```python
class FireDatasetManager:
    """
    Centralized management of fire prediction datasets
    """
    
    def __init__(self, data_root_dir):
        self.data_root_dir = data_root_dir
        self.stacked_data_dir = os.path.join(data_root_dir, 'stacked_datasets')
        self.metadata_path = os.path.join(data_root_dir, 'dataset_metadata.json')
        
        self.load_metadata()
    
    def load_metadata(self):
        """Load dataset metadata"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def get_available_dates(self):
        """Get list of available dates"""
        return self.metadata.get('dataset_info', {}).get('successful_dates', [])
    
    def get_dataset_path(self, date_str):
        """Get path to stacked dataset for specific date"""
        filename = f'stack_{date_str}.tif'
        dataset_path = os.path.join(self.stacked_data_dir, filename)
        
        if os.path.exists(dataset_path):
            return dataset_path
        else:
            raise FileNotFoundError(f"Dataset not found for date: {date_str}")
    
    def load_dataset(self, date_str, bands=None):
        """
        Load dataset for specific date
        
        Args:
            date_str: Date in YYYY_MM_DD format
            bands: List of band indices to load (1-based), or None for all
            
        Returns:
            tuple: (data_array, metadata_dict)
        """
        dataset_path = self.get_dataset_path(date_str)
        
        with rasterio.open(dataset_path) as src:
            if bands is None:
                data = src.read().astype(np.float32)
            else:
                data = src.read(bands).astype(np.float32)
            
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'shape': (src.height, src.width),
                'band_count': src.count,
                'band_descriptions': [src.get_band_description(i) for i in range(1, src.count + 1)]
            }
        
        return data, metadata
    
    def get_dataset_statistics(self, date_str):
        """Calculate statistics for dataset"""
        data, metadata = self.load_dataset(date_str)
        
        stats = {}
        band_names = metadata['band_descriptions']
        
        for i, band_name in enumerate(band_names):
            band_data = data[i]
            stats[band_name] = {
                'min': float(np.nanmin(band_data)),
                'max': float(np.nanmax(band_data)),
                'mean': float(np.nanmean(band_data)),
                'std': float(np.nanstd(band_data)),
                'missing_percentage': float(np.isnan(band_data).sum() / band_data.size * 100)
            }
        
        return stats
    
    def create_data_subset(self, date_str, bounds, output_path):
        """Create spatial subset of dataset"""
        data, metadata = self.load_dataset(date_str)
        
        # Calculate subset window
        window = rasterio.windows.from_bounds(*bounds, transform=metadata['transform'])
        
        with rasterio.open(self.get_dataset_path(date_str)) as src:
            subset_data = src.read(window=window)
            
            # Update metadata for subset
            subset_transform = src.window_transform(window)
            subset_profile = src.profile.copy()
            subset_profile.update({
                'height': window.height,
                'width': window.width,
                'transform': subset_transform
            })
            
            # Save subset
            with rasterio.open(output_path, 'w', **subset_profile) as dst:
                dst.write(subset_data)
        
        return output_path
```

## Integration with ML Pipeline

### Training Data Preparation

```python
def prepare_ml_training_data(dataset_manager, train_dates, validation_dates, 
                           patch_size=256, overlap=64):
    """
    Prepare training patches for ML model
    """
    
    training_patches = []
    validation_patches = []
    
    # Process training dates
    for date_str in train_dates:
        try:
            data, metadata = dataset_manager.load_dataset(date_str)
            
            # Extract features (first 9 bands) and target (fire occurrence)
            features = data[:9]  # Environmental bands
            target = create_fire_target_from_viirs(date_str, metadata)
            
            # Extract patches
            patches = extract_patches_with_overlap(
                features, target, patch_size, overlap
            )
            
            training_patches.extend(patches)
            print(f"✅ Training: {date_str} - {len(patches)} patches")
            
        except Exception as e:
            print(f"❌ Training: {date_str} - {str(e)}")
    
    # Process validation dates  
    for date_str in validation_dates:
        try:
            data, metadata = dataset_manager.load_dataset(date_str)
            features = data[:9]
            target = create_fire_target_from_viirs(date_str, metadata)
            
            patches = extract_patches_with_overlap(
                features, target, patch_size, overlap
            )
            
            validation_patches.extend(patches)
            print(f"✅ Validation: {date_str} - {len(patches)} patches")
            
        except Exception as e:
            print(f"❌ Validation: {date_str} - {str(e)}")
    
    return training_patches, validation_patches
```

## Export and Distribution

### Dataset Export Functions

```python
def export_dataset_for_kaggle(dataset_manager, output_dir, compress=True):
    """
    Export datasets in Kaggle competition format
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    available_dates = dataset_manager.get_available_dates()
    
    for date_str in available_dates:
        src_path = dataset_manager.get_dataset_path(date_str)
        dst_filename = f'stack_{date_str}.tif'
        dst_path = os.path.join(output_dir, dst_filename)
        
        if compress:
            # Compress using GDAL translate
            subprocess.run([
                'gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES',
                src_path, dst_path
            ], check=True)
        else:
            shutil.copy2(src_path, dst_path)
    
    # Copy metadata
    shutil.copy2(dataset_manager.metadata_path, output_dir)
    
    print(f"✅ Exported {len(available_dates)} datasets to {output_dir}")

def create_data_documentation(dataset_manager, output_path):
    """Create comprehensive data documentation"""
    
    doc_content = f"""
# Forest Fire Prediction Dataset Documentation

## Dataset Overview
- **Spatial Coverage**: Uttarakhand state, India
- **Temporal Coverage**: April-May 2016 fire season
- **Spatial Resolution**: 30 meters
- **Total Files**: {len(dataset_manager.get_available_dates())}

## Data Bands
1. **Elevation** (m): SRTM DEM elevation data
2. **Slope** (degrees): Terrain slope calculated from DEM
3. **Aspect** (degrees): Terrain aspect (0-360°)
4. **Temperature** (°C): ERA5 daily maximum temperature
5. **Humidity** (%) ERA5 daily minimum relative humidity
6. **Wind Speed** (m/s): ERA5 daily maximum wind speed
7. **Precipitation** (mm): ERA5 daily total precipitation
8. **Fuel Type** (0-4): Land cover derived fuel classification
9. **Settlement** (%): GHSL built-up density

## Data Sources
- **DEM**: NASA SRTM 30m
- **Weather**: Copernicus ERA5 Reanalysis
- **Land Cover**: ESA WorldCover 2020
- **Settlement**: Global Human Settlement Layer 2015
- **Fire Ground Truth**: NASA VIIRS active fire detections

## Usage Instructions
Load stacked datasets using rasterio:
```python
import rasterio
import numpy as np

with rasterio.open('stack_2016_05_15.tif') as src:
    data = src.read().astype(np.float32)  # Shape: (9, height, width)
    transform = src.transform
    crs = src.crs
```

## Quality Metrics
- **Spatial Alignment**: All bands aligned to common 30m grid
- **Missing Data**: <5% per band on average
- **Value Ranges**: Validated against expected physical ranges
- **Temporal Consistency**: Daily time series with no gaps
"""
    
    with open(output_path, 'w') as f:
        f.write(doc_content)
```

---

**Key Processes:**
- Multi-source data integration and alignment
- Spatial resampling and temporal interpolation
- Quality validation and consistency checking
- Training data preparation for ML pipeline
- Export formatting for various use cases

**Integration Points:**
- ML Model: Training data preparation and validation
- CA Engine: Environmental layer provision
- Web Interface: Data serving and visualization
- Export System: Results and metadata packaging
