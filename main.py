#!/usr/bin/env python3
"""
Bakery Employee Analysis System

This script orchestrates the analysis of bakery employee tracking data,
with enhanced visualizations and time series analysis.
"""

import pandas as pd
import time
from pathlib import Path
import argparse
import sys

from src.visualization.employee_region_vis import create_employee_region_heatmap
from src.utils.file_utils import ensure_dir_exists, check_required_files
from src.utils.data_utils import (
    load_sensor_data, load_floor_plan_data, 
    classify_departments, aggregate_by_shift, 
    aggregate_by_activity, create_activity_profile_by_employee
)
from src.utils.time_utils import format_seconds_to_hms

from src.visualization.activity_vis import (
    plot_activity_distribution, 
    plot_activity_distribution_by_employee,
    plot_hourly_activity_patterns
)
from src.visualization.floor_plan_vis import (
    create_region_heatmap
)
from src.visualization.transition_vis import (
    create_employee_transition_visualization,
    plot_movement_transitions_chart
)

from src.analysis.transitions import (
    calculate_region_transitions,
    analyze_movement_patterns,
    analyze_department_transitions
)
from src.analysis.statistical import (
    analyze_employee_differences,
    analyze_ergonomic_patterns,
    compare_shifts
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze bakery employee tracking data')
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                      help='Path to the data file')
    parser.add_argument('--layout', type=str, default='data/assets/layout.png',
                      help='Path to the floor plan image')
    parser.add_argument('--metadata', type=str, default='data/assets/process_metadata.json',
                      help='Path to the process metadata file')
    parser.add_argument('--connections', type=str, default='data/assets/floor-plan-connections-2025-04-09.json',
                      help='Path to the floor plan connections file')
    parser.add_argument('--output', type=str, default='output',
                      help='Directory to save output files')
    
    # Add step arguments
    parser.add_argument('--step1', action='store_true', help='Run shift analysis')
    parser.add_argument('--step2', action='store_true', help='Run activity analysis')
    parser.add_argument('--step3', action='store_true', help='Run region analysis')
    parser.add_argument('--step4', action='store_true', help='Run employee region heatmaps')
    parser.add_argument('--step5', action='store_true', help='Run statistical analysis')
    parser.add_argument('--employee', type=str, help='Specific employee ID to analyze (e.g., 32-A)')
    
    return parser.parse_args()

def run_employee_heatmaps(data, floor_plan_data, output_dir, employee_id=None):
    """
    Run employee heatmap generation for one or all employees
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    floor_plan_data : dict
        Dictionary containing floor plan data
    output_dir : Path
        Output directory
    employee_id : str, optional
        If provided, generate heatmap only for this employee
    """
    # Create directory for employee region heatmaps
    employee_heatmaps_dir = ensure_dir_exists(output_dir / 'employee_heatmaps')
    
    # Get employees to process
    if employee_id:
        if employee_id not in data['id'].unique():
            print(f"Error: Employee {employee_id} not found in the data")
            return
        employees = [employee_id]
    else:
        employees = sorted(data['id'].unique())
    
    print(f"\nGenerating heatmaps for {len(employees)} employees:")
    
    # Generate region heatmap for each employee
    for emp_id in employees:
        print(f"  Processing employee {emp_id}...")
        
        # Create directory for this employee
        emp_dir = ensure_dir_exists(employee_heatmaps_dir / emp_id)
        
        # Generate region heatmap
        create_employee_region_heatmap(
            data,
            floor_plan_data,
            emp_id,
            save_path=emp_dir / f"{emp_id}_region_heatmap.png",
            top_n=10,  # Display top 10 regions
            min_transitions=2  # Include transitions with at least 2 occurrences
        )
    
    print(f"  Saved employee region heatmaps to {output_dir}/employee_heatmaps/")

def main():
    """Main function to orchestrate the analysis"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if any step is specified, otherwise run all steps
    run_all = not (args.step1 or args.step2 or args.step3 or args.step4 or args.step5)
    
    # Check required files
    required_files = {
        'Data file': args.data,
        'Floor plan image': args.layout,
        'Metadata file': args.metadata,
        'Connections file': args.connections
    }
    
    files_exist, missing_files = check_required_files(required_files)
    if not files_exist:
        print("The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return 1
    
    print("=" * 80)
    print("=== Bakery Employee Movement Analysis ===")
    print("=" * 80)
    
    # Create output directory structure
    output_path = ensure_dir_exists(args.output)
    data_dir = ensure_dir_exists(output_path / 'data')
    vis_dir = ensure_dir_exists(output_path / 'visualizations')
    stats_dir = ensure_dir_exists(output_path / 'statistics')
    floor_plan_dir = ensure_dir_exists(vis_dir / 'floor_plan')
    transitions_dir = ensure_dir_exists(output_path / 'region_transitions')
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = load_sensor_data(args.data)
    
    # Add department classification
    data = classify_departments(data)
    
    # Load floor plan data
    floor_plan_data = load_floor_plan_data(args.layout, args.metadata, args.connections)
    
    # Display basic information
    print(f"\nDataset Information:")
    print(f"  Records: {len(data):,}")
    print(f"  Employees: {data['id'].nunique()} - {sorted(data['id'].unique())}")
    print(f"  Shifts: {data['shift'].nunique()} - {sorted(data['shift'].unique())}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"  Regions: {data['region'].nunique()}")
    print(f"  Activities: {data['activity'].nunique()} - {sorted(data['activity'].unique())}")
    
    # Total duration
    total_seconds = data['duration'].sum()
    total_formatted = format_seconds_to_hms(total_seconds)
    total_hours = total_seconds / 3600
    print(f"  Total duration: {total_formatted} ({total_hours:.2f} hours)")
    
    # 1. Shift Analysis
    if run_all or args.step1:
        print("\n" + "=" * 40)
        print("=== 1. Shift Analysis ===")
        
        shift_summary = aggregate_by_shift(data)
        shift_summary.to_csv(data_dir / 'shift_summary.csv', index=False)
        print("  Saved shift summary to data/shift_summary.csv")
    
    # 2. Activity Analysis
    if run_all or args.step2:
        print("\n" + "=" * 40)
        print("=== 2. Activity Analysis ===")
        
        activity_summary = aggregate_by_activity(data)
        activity_summary.to_csv(data_dir / 'activity_summary.csv', index=False)
        print("  Saved activity summary to data/activity_summary.csv")
        
        plot_activity_distribution(activity_summary, save_path=vis_dir / 'activity_distribution.png')
        
        # Activity profile by employee - use the modified version that shows formatted times
        plot_activity_distribution_by_employee(data, save_path=vis_dir / 'activity_distribution_by_employee.png')
        print("  Saved activity distribution by employee to visualizations/activity_distribution_by_employee.png")
    
    # 3. Region Analysis
    if run_all or args.step3:
        print("\n" + "=" * 40)
        print("=== 3. Region Analysis ===")
        
        # Aggregate by region
        region_summary = data.groupby('region')['duration'].sum().reset_index()
        region_summary['percentage'] = (region_summary['duration'] / region_summary['duration'].sum() * 100).round(1)
        region_summary['formatted_time'] = region_summary['duration'].apply(format_seconds_to_hms)
        region_summary = region_summary.sort_values('duration', ascending=False)
        
        region_summary.to_csv(data_dir / 'region_summary.csv', index=False)
        print("  Saved region summary to data/region_summary.csv")
        
        # Create floor plan visualizations
        if floor_plan_data:
            create_region_heatmap(floor_plan_data, region_summary, save_path=floor_plan_dir / 'region_heatmap.png')
    
    # 4. Employee Region Heatmaps
    if run_all or args.step4:
        print("\n" + "=" * 40)
        print("=== 4. Employee Region Heatmaps ===")
        
        run_employee_heatmaps(data, floor_plan_data, vis_dir, args.employee)
    
    # 5. Statistical Analysis
    if run_all or args.step5:
        print("\n" + "=" * 40)
        print("=== 5. Statistical Analysis ===")
        
        employee_analysis = analyze_employee_differences(data)
        
        # Save walking patterns
        if 'walking_patterns' in employee_analysis:
            employee_analysis['walking_patterns'].to_csv(stats_dir / 'walking_patterns_by_employee.csv', index=False)
            print("  Saved walking patterns analysis to statistics/walking_patterns_by_employee.csv")
        
        # Save handling patterns
        if 'handling_patterns' in employee_analysis:
            employee_analysis['handling_patterns'].to_csv(stats_dir / 'handling_patterns_by_employee.csv', index=False)
            print("  Saved handling patterns analysis to statistics/handling_patterns_by_employee.csv")
        
        # Ergonomic Analysis
        ergonomic_analysis = analyze_ergonomic_patterns(data)
        
        if 'position_summary' in ergonomic_analysis:
            ergonomic_analysis['position_summary'].to_csv(stats_dir / 'handling_position_summary.csv', index=False)
            print("  Saved handling position summary to statistics/handling_position_summary.csv")
        
        if 'employee_position_summary' in ergonomic_analysis:
            ergonomic_analysis['employee_position_summary'].to_csv(stats_dir / 'employee_handling_positions.csv', index=False)
            print("  Saved employee handling positions to statistics/employee_handling_positions.csv")
        
        # Shift Comparison
        shift_comparison = compare_shifts(data)
        
        if 'shift_metrics' in shift_comparison:
            shift_comparison['shift_metrics'].to_csv(stats_dir / 'shift_metrics.csv', index=False)
            print("  Saved shift metrics to statistics/shift_metrics.csv")
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "=" * 40)
    print("=== Analysis Complete ===")
    print("=" * 40)
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"\nResults saved to: {output_path}")
    
    # Only show outputs for the steps that were run
    print("\nKey outputs to check:")
    if run_all or args.step1:
        print(f"1. {data_dir}/shift_summary.csv - Time spent during each shift")
    if run_all or args.step2:
        print(f"2. {vis_dir}/activity_distribution_by_employee.png - Activity profiles by employee")
    if run_all or args.step3:
        print(f"3. {floor_plan_dir}/region_heatmap.png - Region usage heatmap")
    if run_all or args.step4:
        print(f"4. {vis_dir}/employee_heatmaps/ - Employee region heatmaps")
    if run_all or args.step5:
        print(f"5. {stats_dir}/shift_metrics.csv - Shift comparison metrics")

if __name__ == "__main__":
    sys.exit(main())