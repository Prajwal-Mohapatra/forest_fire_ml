#!/usr/bin/env python3
"""
Dataset Validation Script
=========================
Comprehensive validation script to check for data stacking issues, 
date mismatches, and dataset integrity as mentioned in the Grok analysis.
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
    stack_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_*.tif')))
    mask_files = sorted(glob.glob(os.path.join(base_dir, 'fire_mask_2016_*.tif')))
    
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
    
    print(f"\n📊 ANALYZING FILE INTEGRITY AND FIRE STATISTICS...")
    
    for i, stack_file in enumerate(stack_files):
        try:
            with rasterio.open(stack_file) as src:
                # Check basic properties
                if src.count != 10:
                    print(f"⚠️ Warning: {stack_file} has {src.count} bands (expected 10)")
                
                # Read fire mask (band 10)
                fire_mask = src.read(10)
                
                # Calculate fire statistics
                total_pixels = fire_mask.size
                fire_pixels = np.sum(fire_mask > 0)
                fire_percentage = (fire_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                
                # Get date from filename
                filename = os.path.basename(stack_file)
                date_parts = filename.split('_')[1:4]
                date_str = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2].split('.')[0]}"
                
                fire_stats.append({
                    'file': filename,
                    'date': date_str,
                    'total_pixels': total_pixels,
                    'fire_pixels': fire_pixels,
                    'fire_percentage': fire_percentage,
                    'has_fire': fire_pixels > 0
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
    print(f"   Days with fire: {fire_df['has_fire'].sum()}")
    print(f"   Days without fire: {(~fire_df['has_fire']).sum()}")
    print(f"   Average fire percentage: {fire_df['fire_percentage'].mean():.6f}%")
    print(f"   Max fire percentage: {fire_df['fire_percentage'].max():.6f}%")
    print(f"   Min fire percentage: {fire_df['fire_percentage'].min():.6f}%")
    
    # Find peak fire days
    top_fire_days = fire_df.nlargest(5, 'fire_percentage')
    print(f"\n📈 TOP 5 FIRE ACTIVITY DAYS:")
    for _, row in top_fire_days.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: {row['fire_percentage']:.6f}% ({row['fire_pixels']} pixels)")
    
    # Find zero fire days
    zero_fire_days = fire_df[fire_df['fire_pixels'] == 0]
    if len(zero_fire_days) > 0:
        print(f"\n⚪ DAYS WITH NO FIRE DETECTED ({len(zero_fire_days)} days):")
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
    
    if fire_df['fire_percentage'].max() < 0.01:
        recommendations.append("Extremely low fire activity - consider different fire detection thresholds")
    
    if len(zero_fire_days) > len(fire_df) * 0.8:
        recommendations.append("Most days have no fire - check fire mask generation process")
    
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
    Plot fire activity over time
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Fire percentage over time
    ax1.plot(fire_df['date'], fire_df['fire_percentage'], 'r-', linewidth=1, alpha=0.7)
    ax1.scatter(fire_df['date'], fire_df['fire_percentage'], c='red', s=20, alpha=0.6)
    ax1.set_title('Fire Activity Percentage Over Time', fontsize=14)
    ax1.set_ylabel('Fire Percentage (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Fire pixels over time
    ax2.plot(fire_df['date'], fire_df['fire_pixels'], 'orange', linewidth=1, alpha=0.7)
    ax2.scatter(fire_df['date'], fire_df['fire_pixels'], c='orange', s=20, alpha=0.6)
    ax2.set_title('Fire Pixels Over Time', fontsize=14)
    ax2.set_ylabel('Fire Pixels', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
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
    base_dir = "/home/swayam/projects/forest_fire_spread/dataset"
    
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
