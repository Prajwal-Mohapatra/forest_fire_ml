#!/usr/bin/env python3
"""
Dataset Validation Script
=========================
Comprehensive validation script to check for data stacking issues, 
date mismatches, and dataset integrity as mentioned in the Grok analysis.
Enhanced with Uttarakhand shapefile-based geographic boundary validation.
"""

import os
import glob
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from utils.preprocess import create_uttarakhand_mask_from_shapefile

def create_uttarakhand_mask(src):
    """
    Create a mask for Uttarakhand state boundaries using shapefile for precise boundary validation.
    This replaces coordinate-based masking with accurate polygon-based masking.
    """
    try:
        # Use the shapefile-based function from preprocess
        uttarakhand_mask = create_uttarakhand_mask_from_shapefile(src)
        
        # Return mask and dummy bounds for compatibility
        uttarakhand_bounds = {
            'method': 'shapefile',
            'shapefile': '/home/swayam/projects/forest_fire_spread/forest_fire_ml/utils/UK_BOUNDS/Uttarakhand_Boundary.shp'
        }
        
        return uttarakhand_mask, uttarakhand_bounds
        
    except Exception as e:
        print(f"⚠️ Could not create Uttarakhand mask from shapefile: {e}")
        print("   Falling back to full image (no masking)")
        # Return an all-True mask as fallback
        return np.ones((src.height, src.width), dtype=bool), None

def apply_geographic_masking(fire_mask, uttarakhand_mask):
    """
    Apply geographic masking to focus only on Uttarakhand region using shapefile boundaries
    """
    if uttarakhand_mask is not None:
        # Mask out fire pixels outside Uttarakhand using precise shapefile boundaries
        masked_fire = fire_mask * uttarakhand_mask
        return masked_fire
    else:
        return fire_mask

def validate_dataset_integrity(base_dir, verbose=True):
    """
    Validate the integrity of the stacked dataset
    
    Args:
        base_dir: Base directory containing stacked TIFF files
        verbose: Print detailed information
    
    Returns:
        dict: Validation results and recommendations
    """
    
    print("🔍 DATASET VALIDATION STARTING...")
    print("=" * 60)
    
    # Find all stacked files and masks
    stack_files = sorted(glob.glob(os.path.join('/home/swayam/projects/forest_fire_spread/datasets/dataset_stacked', 'stack_2016_*.tif')))
    mask_files = sorted(glob.glob(os.path.join('/home/swayam/projects/forest_fire_spread/datasets/dataset_unstacked/fire_mask', 'fire_mask_2016_*.tif')))
    
    print(f"📁 Found {len(stack_files)} stack files")
    print(f"🔥 Found {len(mask_files)} mask files")
    
    validation_results = {
        'total_stacks': len(stack_files),
        'total_masks': len(mask_files),
        'date_mismatches': [],
        'missing_stacks': [],
        'corrupted_files': [],
        'fire_statistics': {},
        'recommendations': []
    }
    
    # Check for date mismatches
    stack_dates = set()
    mask_dates = set()
    
    # Extract dates from filenames
    for stack_file in stack_files:
        try:
            filename = os.path.basename(stack_file)
            # Extract date from filename like 'stack_2016_04_15.tif'
            parts = filename.split('_')
            if len(parts) >= 4:
                date_str = f"{parts[1]}_{parts[2]}_{parts[3].split('.')[0]}"
                stack_dates.add(date_str)
        except Exception as e:
            print(f"⚠️ Error parsing stack filename {stack_file}: {e}")
    
    for mask_file in mask_files:
        try:
            filename = os.path.basename(mask_file)
            # Extract date from filename like 'fire_mask_2016_04_15.tif'
            parts = filename.split('_')
            if len(parts) >= 4:
                date_str = f"{parts[2]}_{parts[3]}_{parts[4].split('.')[0]}"
                mask_dates.add(date_str)
        except Exception as e:
            print(f"⚠️ Error parsing mask filename {mask_file}: {e}")
    
    # Find missing correspondences
    missing_stacks = mask_dates - stack_dates
    extra_stacks = stack_dates - mask_dates
    
    if missing_stacks:
        validation_results['missing_stacks'] = list(missing_stacks)
        print(f"⚠️ Found {len(missing_stacks)} dates with masks but no stacks:")
        for date in missing_stacks:
            print(f"   - {date}")
    
    if extra_stacks:
        print(f"ℹ️ Found {len(extra_stacks)} dates with stacks but no masks:")
        for date in extra_stacks:
            print(f"   - {date}")
    
    # Validate file integrity and collect statistics
    fire_stats = []
    corrupted_files = []
    geographic_stats = {'total_pixels': 0, 'uttarakhand_pixels': 0, 'fire_pixels_raw': 0, 'fire_pixels_masked': 0}
    
    print(f"\n📊 ANALYZING FILE INTEGRITY AND FIRE STATISTICS...")
    print("🗺️ Applying Uttarakhand shapefile-based geographic masking...")
    
    for i, stack_file in enumerate(stack_files):
        try:
            with rasterio.open(stack_file) as src:
                # Check basic properties
                if src.count != 10:
                    print(f"⚠️ Warning: {stack_file} has {src.count} bands (expected 10)")
                
                # Create Uttarakhand boundary mask
                uttarakhand_mask, bounds = create_uttarakhand_mask(src)
                
                # Read fire mask (band 10)
                fire_mask_raw = src.read(10)
                
                # Apply geographic masking
                fire_mask_masked = apply_geographic_masking(fire_mask_raw, uttarakhand_mask)
                
                # Calculate statistics
                total_pixels = fire_mask_raw.size
                uttarakhand_pixels = np.sum(uttarakhand_mask) if uttarakhand_mask is not None else total_pixels
                fire_pixels_raw = np.sum(fire_mask_raw > 0)
                fire_pixels_masked = np.sum(fire_mask_masked > 0)
                
                fire_percentage_raw = (fire_pixels_raw / total_pixels) * 100 if total_pixels > 0 else 0
                fire_percentage_masked = (fire_pixels_masked / uttarakhand_pixels) * 100 if uttarakhand_pixels > 0 else 0
                
                # Update cumulative geographic stats
                geographic_stats['total_pixels'] += total_pixels
                geographic_stats['uttarakhand_pixels'] += uttarakhand_pixels
                geographic_stats['fire_pixels_raw'] += fire_pixels_raw
                geographic_stats['fire_pixels_masked'] += fire_pixels_masked
                
                # Get date from filename
                filename = os.path.basename(stack_file)
                date_parts = filename.split('_')[1:4]
                date_str = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2].split('.')[0]}"
                
                fire_stats.append({
                    'file': filename,
                    'date': date_str,
                    'total_pixels': total_pixels,
                    'uttarakhand_pixels': uttarakhand_pixels,
                    'fire_pixels_raw': fire_pixels_raw,
                    'fire_pixels_masked': fire_pixels_masked,
                    'fire_percentage_raw': fire_percentage_raw,
                    'fire_percentage_masked': fire_percentage_masked,
                    'has_fire_raw': fire_pixels_raw > 0,
                    'has_fire_masked': fire_pixels_masked > 0
                })
                
                if verbose and i % 10 == 0:
                    print(f"   Processed {i+1}/{len(stack_files)} files...")
                    
        except Exception as e:
            corrupted_files.append(stack_file)
            print(f"❌ Error reading {stack_file}: {e}")
    
    validation_results['corrupted_files'] = corrupted_files
    validation_results['fire_statistics'] = fire_stats
    
    # Analyze fire statistics
    fire_df = pd.DataFrame(fire_stats)
    fire_df['date'] = pd.to_datetime(fire_df['date'])
    fire_df = fire_df.sort_values('date')
    
    print(f"\n🔥 FIRE STATISTICS SUMMARY:")
    print(f"   Total days analyzed: {len(fire_df)}")
    print(f"   Days with fire (raw): {fire_df['has_fire_raw'].sum()}")
    print(f"   Days with fire (Uttarakhand-masked): {fire_df['has_fire_masked'].sum()}")
    print(f"   Days without fire (raw): {(~fire_df['has_fire_raw']).sum()}")
    print(f"   Days without fire (Uttarakhand-masked): {(~fire_df['has_fire_masked']).sum()}")
    print(f"   Average fire percentage (raw): {fire_df['fire_percentage_raw'].mean():.6f}%")
    print(f"   Average fire percentage (Uttarakhand-masked): {fire_df['fire_percentage_masked'].mean():.6f}%")
    print(f"   Max fire percentage (raw): {fire_df['fire_percentage_raw'].max():.6f}%")
    print(f"   Max fire percentage (Uttarakhand-masked): {fire_df['fire_percentage_masked'].max():.6f}%")
    
    # Geographic coverage stats
    total_coverage = geographic_stats['uttarakhand_pixels'] / geographic_stats['total_pixels'] * 100
    fire_ratio_improvement = (geographic_stats['fire_pixels_masked'] / geographic_stats['uttarakhand_pixels']) / (geographic_stats['fire_pixels_raw'] / geographic_stats['total_pixels']) if geographic_stats['fire_pixels_raw'] > 0 else 0
    
    print(f"\n🗺️ SHAPEFILE-BASED GEOGRAPHIC MASKING RESULTS:")
    print(f"   Uttarakhand coverage: {total_coverage:.2f}% of total area")
    print(f"   Raw fire pixels: {geographic_stats['fire_pixels_raw']:,}")
    print(f"   Shapefile-masked fire pixels: {geographic_stats['fire_pixels_masked']:,}")
    print(f"   Fire density improvement: {fire_ratio_improvement:.2f}x")
    
    # Find peak fire days
    top_fire_days = fire_df.nlargest(5, 'fire_percentage_masked')
    print(f"\n📈 TOP 5 FIRE ACTIVITY DAYS (Uttarakhand-masked):")
    for _, row in top_fire_days.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: {row['fire_percentage_masked']:.6f}% ({row['fire_pixels_masked']} pixels)")
    
    # Find zero fire days
    zero_fire_days = fire_df[fire_df['fire_pixels_masked'] == 0]
    if len(zero_fire_days) > 0:
        print(f"\n⚪ DAYS WITH NO FIRE DETECTED (Uttarakhand-masked, {len(zero_fire_days)} days):")
        for _, row in zero_fire_days.head(10).iterrows():
            print(f"   {row['date'].strftime('%Y-%m-%d')}")
        if len(zero_fire_days) > 10:
            print(f"   ... and {len(zero_fire_days) - 10} more")
    
    # Generate recommendations
    recommendations = []
    
    if len(missing_stacks) > 0:
        recommendations.append(f"Re-run stacking script for {len(missing_stacks)} missing dates")
    
    if len(corrupted_files) > 0:
        recommendations.append(f"Fix or regenerate {len(corrupted_files)} corrupted files")
    
    if fire_df['fire_percentage_masked'].max() < 0.01:
        recommendations.append("Extremely low fire activity (even after masking) - consider different fire detection thresholds")
    
    if len(zero_fire_days) > len(fire_df) * 0.8:
        recommendations.append("Most days have no fire (after Uttarakhand masking) - check fire mask generation process")
    
    # Add shapefile-based geographic masking recommendation
    if fire_ratio_improvement > 1.5:
        recommendations.append(f"Shapefile-based geographic masking improved fire density by {fire_ratio_improvement:.1f}x - integration successful")
    elif fire_ratio_improvement < 1.1:
        recommendations.append("Shapefile-based geographic masking shows minimal improvement - verify Uttarakhand shapefile boundaries")
    
    # Check temporal coverage
    date_range = fire_df['date'].max() - fire_df['date'].min()
    expected_days = date_range.days + 1
    actual_days = len(fire_df)
    
    if actual_days < expected_days:
        missing_days = expected_days - actual_days
        recommendations.append(f"Missing {missing_days} days in temporal coverage")
    
    validation_results['recommendations'] = recommendations
    
    print(f"\n💡 RECOMMENDATIONS:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✅ Dataset appears to be in good condition!")
    
    return validation_results, fire_df

def plot_fire_activity_timeline(fire_df, output_dir='outputs'):
    """
    Plot fire activity over time with both raw and Uttarakhand-masked data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Fire percentage over time - Raw
    ax1.plot(fire_df['date'], fire_df['fire_percentage_raw'], 'r-', linewidth=1, alpha=0.7, label='Raw')
    ax1.scatter(fire_df['date'], fire_df['fire_percentage_raw'], c='red', s=15, alpha=0.6)
    ax1.set_title('Fire Activity Percentage Over Time (Raw)', fontsize=14)
    ax1.set_ylabel('Fire Percentage (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Fire percentage over time - Shapefile Masked
    ax2.plot(fire_df['date'], fire_df['fire_percentage_masked'], 'orange', linewidth=1, alpha=0.7, label='Shapefile Masked')
    ax2.scatter(fire_df['date'], fire_df['fire_percentage_masked'], c='orange', s=15, alpha=0.6)
    ax2.set_title('Fire Activity Percentage Over Time (Shapefile Masked)', fontsize=14)
    ax2.set_ylabel('Fire Percentage (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Fire pixels over time - Raw
    ax3.plot(fire_df['date'], fire_df['fire_pixels_raw'], 'darkred', linewidth=1, alpha=0.7)
    ax3.scatter(fire_df['date'], fire_df['fire_pixels_raw'], c='darkred', s=15, alpha=0.6)
    ax3.set_title('Fire Pixels Over Time (Raw)', fontsize=14)
    ax3.set_ylabel('Fire Pixels', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Fire pixels over time - Shapefile Masked
    ax4.plot(fire_df['date'], fire_df['fire_pixels_masked'], 'darkorange', linewidth=1, alpha=0.7)
    ax4.scatter(fire_df['date'], fire_df['fire_pixels_masked'], c='darkorange', s=15, alpha=0.6)
    ax4.set_title('Fire Pixels Over Time (Shapefile Masked)', fontsize=14)
    ax4.set_ylabel('Fire Pixels', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fire_activity_timeline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Fire activity timeline saved to {output_path}")
    plt.show()

def check_band_consistency(base_dir, sample_size=10):
    """
    Check consistency of bands across files
    """
    print(f"\n🔬 CHECKING BAND CONSISTENCY (sampling {sample_size} files)...")
    
    stack_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_*.tif')))
    
    if len(stack_files) == 0:
        print("❌ No stack files found!")
        return
    
    # Sample files for analysis
    sample_files = stack_files[::max(1, len(stack_files)//sample_size)][:sample_size]
    
    band_stats = defaultdict(list)
    
    for file_path in sample_files:
        try:
            with rasterio.open(file_path) as src:
                for band_idx in range(1, src.count + 1):
                    band_data = src.read(band_idx)
                    
                    stats = {
                        'min': float(np.nanmin(band_data)),
                        'max': float(np.nanmax(band_data)),
                        'mean': float(np.nanmean(band_data)),
                        'std': float(np.nanstd(band_data)),
                        'nan_count': int(np.sum(np.isnan(band_data))),
                        'inf_count': int(np.sum(np.isinf(band_data)))
                    }
                    
                    band_stats[band_idx].append(stats)
                    
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
    
    # Analyze band statistics
    print(f"📈 BAND STATISTICS SUMMARY:")
    for band_idx in range(1, 11):  # Assuming 10 bands
        if band_idx in band_stats:
            stats_list = band_stats[band_idx]
            
            avg_min = np.mean([s['min'] for s in stats_list])
            avg_max = np.mean([s['max'] for s in stats_list])
            avg_mean = np.mean([s['mean'] for s in stats_list])
            total_nans = sum([s['nan_count'] for s in stats_list])
            total_infs = sum([s['inf_count'] for s in stats_list])
            
            band_name = "Fire Mask" if band_idx == 10 else f"Feature {band_idx}"
            
            print(f"   Band {band_idx} ({band_name}):")
            print(f"     Range: {avg_min:.3f} to {avg_max:.3f}")
            print(f"     Average: {avg_mean:.3f}")
            
            if total_nans > 0:
                print(f"     ⚠️ NaN values: {total_nans}")
            if total_infs > 0:
                print(f"     ⚠️ Inf values: {total_infs}")

def main():
    """
    Main validation function
    """
    # Configuration
    base_dir = "/home/swayam/projects/forest_fire_spread/datasets/dataset_stacked/"
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Directory {base_dir} does not exist!")
        print("Please update the base_dir path to point to your dataset directory.")
        return
    
    # Run validation
    try:
        validation_results, fire_df = validate_dataset_integrity(base_dir)
        
        # Plot timeline
        if len(fire_df) > 0:
            plot_fire_activity_timeline(fire_df)
        
        # Check band consistency
        check_band_consistency(base_dir)
        
        print(f"\n✅ VALIDATION COMPLETE!")
        print(f"📋 Summary:")
        print(f"   - Total stacks: {validation_results['total_stacks']}")
        print(f"   - Total masks: {validation_results['total_masks']}")
        print(f"   - Corrupted files: {len(validation_results['corrupted_files'])}")
        print(f"   - Missing stacks: {len(validation_results['missing_stacks'])}")
        print(f"   - Recommendations: {len(validation_results['recommendations'])}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
