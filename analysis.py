#!/usr/bin/env python3
"""
Bakery Employee Analysis - Heatmap Focus
=======================================

This script creates an employee activity heatmap and exports the data to CSV.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Import your existing utility functions
from src.utils.data_utils import load_sensor_data, classify_departments
from src.utils.file_utils import ensure_dir_exists
from src.utils.time_utils import format_seconds_to_hms
from src.visualization.activity_vis import plot_activity_distribution_by_employee
from src.visualization.base import set_visualization_style, get_activity_order

def print_step_header(step_num, title):
    """Print a nicely formatted step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")

def create_employee_heatmap_and_csv(data, output_dir):
    """Create the employee activity heatmap and export data to CSV"""
    print_step_header(1, "EMPLOYEE ACTIVITY HEATMAP & CSV EXPORT")
    
    print("🎨 Generating employee activity heatmap...")
    print("   • Rows: Employees")
    print("   • Columns: Activities (Walk, Stand, Handle center, Handle down, Handle up)")
    print("   • Values: Percentage of time + Duration (hh:mm:ss)")
    print("   • Style: Clean with activity-specific color intensities")
    
    # Create the heatmap
    fig = plot_activity_distribution_by_employee(
        data,
        save_path=output_dir / 'employee_activity_heatmap.png',
        figsize=(12, 6),
        language='en'
    )
    
    print(f"✅ Heatmap saved to: {output_dir / 'employee_activity_heatmap.png'}")
    
    # Now create the CSV export
    print("\n📊 Exporting data to CSV...")
    csv_data = create_activity_csv_data(data)
    csv_path = output_dir / 'employee_activity_data.csv'
    csv_data.to_csv(csv_path, index=False)
    
    print(f"✅ CSV data saved to: {csv_path}")
    print(f"   • Format: Percentage (hh:mm:ss) in each cell")
    print(f"   • Headers: employee, walk, stand, handle_center, handle_down, handle_up")
    
    return fig, csv_data

def create_activity_csv_data(data):
    """Create CSV data with percentage and duration in each cell"""
    
    # Get standard activity order
    activity_order = get_activity_order()
    
    # Calculate total duration by employee
    emp_totals = data.groupby('id')['duration'].sum().reset_index()
    
    # Calculate activity durations by employee
    emp_activity = data.groupby(['id', 'activity'])['duration'].sum().reset_index()
    
    # Merge to get employee totals
    emp_activity = pd.merge(emp_activity, emp_totals, on='id', suffixes=('', '_total'))
    
    # Calculate percentage and format time
    emp_activity['percentage'] = (emp_activity['duration'] / emp_activity['duration_total'] * 100).round(1)
    emp_activity['formatted_time'] = emp_activity['duration'].apply(format_seconds_to_hms)
    emp_activity['combined'] = emp_activity['percentage'].astype(str) + '% (' + emp_activity['formatted_time'] + ')'
    
    # Create pivot table with combined percentage and time
    pivot_combined = pd.pivot_table(
        emp_activity,
        values='combined',
        index='id',
        columns='activity',
        fill_value='0.0% (00:00:00)',
        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else '0.0% (00:00:00)'
    )
    
    # Reorder columns based on standard activity order and filter out unwanted columns
    ordered_cols = [col for col in activity_order if col in pivot_combined.columns]
    pivot_combined = pivot_combined[ordered_cols]
    
    # Reset index to make employee ID a column
    pivot_combined = pivot_combined.reset_index()
    
    # Rename columns to match desired headers
    column_mapping = {
        'id': 'employee',
        'Walk': 'walk',
        'Stand': 'stand', 
        'Handle center': 'handle_center',
        'Handle down': 'handle_down',
        'Handle up': 'handle_up'
    }
    
    pivot_combined = pivot_combined.rename(columns=column_mapping)
    
    return pivot_combined

def main():
    """Main analysis function"""
    print("🏭 BAKERY EMPLOYEE ACTIVITY ANALYSIS")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = 'data/raw/processed_sensor_data.csv'
    OUTPUT_DIR = Path('analysis_output')
    
    # Create output directory
    ensure_dir_exists(OUTPUT_DIR)
    
    try:
        # Load data
        print("📁 Loading data...")
        data = load_sensor_data(DATA_PATH)
        
        # Add department classification
        data = classify_departments(data)
        
        print(f"📊 Dataset loaded: {len(data):,} records from {data['id'].nunique()} employees")
        
        # Step 1: Create Heatmap and CSV
        heatmap_fig, csv_data = create_employee_heatmap_and_csv(data, OUTPUT_DIR)
        
        # Print a preview of the CSV data
        print("\n📋 CSV Data Preview:")
        print(csv_data.to_string(index=False))
        
        # Summary
        print(f"\n{'='*60}")
        print("✅ ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"📊 Created employee activity heatmap")
        print(f"📁 Exported data to CSV format")
        print(f"💾 All files saved to: {OUTPUT_DIR}")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {DATA_PATH}")
        print("   Please make sure the file exists and the path is correct.")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)