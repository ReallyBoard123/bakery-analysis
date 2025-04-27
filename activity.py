#!/usr/bin/env python3
"""
Bakery Walking Pattern Analysis

This script analyzes walking patterns of bakery employees based on sensor data,
creating timeline-based density plots to show when employees walk during the day.
"""

import pandas as pd
import argparse
import time
from pathlib import Path
import sys

from src.analysis.activity.walking import analyze_walking_by_shift, analyze_walking_patterns
from src.utils.file_utils import ensure_dir_exists, check_required_files
from src.utils.data_utils import load_sensor_data, classify_departments

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze bakery employee walking patterns')
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                      help='Path to the data file')
    parser.add_argument('--output', type=str, default='output',
                      help='Directory to save output files')
    parser.add_argument('--german', action='store_true', help='Generate visualizations and reports in German')
    parser.add_argument('--walk', action='store_true', help='Analyze walking patterns')
    parser.add_argument('--employee', type=str, help='Specific employee ID to analyze (e.g., 32-A)')
    parser.add_argument('--time-slot', type=int, default=30, 
                        help='Time slot size in minutes (default: 30)')
    
    return parser.parse_args()

def main():
    """Main function to orchestrate the analysis"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine language based on --german flag
    language = 'de' if args.german else 'en'
    
    # Validate time slot input
    if args.time_slot <= 0 or args.time_slot > 120:
        print("Error: Time slot size must be between 1 and 120 minutes")
        return 1
    
    # Check if 24 hours is divisible by the time slot to avoid partial slots
    if 24 * 60 % args.time_slot != 0:
        print(f"Warning: 24 hours is not evenly divisible by {args.time_slot} minutes.")
        print(f"Consider using one of these values: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, or 120 minutes.")
    
    # Check required files
    required_files = {
        'Data file': args.data
    }
    
    files_exist, missing_files = check_required_files(required_files)
    if not files_exist:
        print("The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return 1
    
    print("=" * 80)
    print("=== Bakery Walking Pattern Analysis ===")
    print("=" * 80)
    
    # Create output directory structure
    output_path = ensure_dir_exists(args.output)
    walking_dir = ensure_dir_exists(output_path / 'walking_analysis')
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = load_sensor_data(args.data)
    
    # Add department classification if the function exists
    try:
        data = classify_departments(data)
        print("Added department classification")
    except Exception as e:
        print(f"Note: Could not classify departments: {e}")
    
    # Filter by employee if specified
    if args.employee:
        if args.employee not in data['id'].unique():
            print(f"Error: Employee {args.employee} not found in the data")
            return 1
        
        data = data[data['id'] == args.employee]
        print(f"Filtered data for employee {args.employee}")
    
    # Check if walking analysis is requested
    if args.walk:
        print("\n" + "=" * 40)
        print(f"=== Walking Pattern Analysis ({args.time_slot}-minute intervals) ===")
        
        print(f"Analyzing walking patterns throughout the day...")
        walking_results = analyze_walking_patterns(data, walking_dir, args.time_slot, language)
        
        print(f"Analyzing walking patterns by shift...")
        if 'shift' in data.columns:
            shift_results = analyze_walking_by_shift(data, walking_dir, args.time_slot, language)
            print(f"Created shift comparison visualizations for {len(shift_results)} employees")
        else:
            print("Shift information not found in dataset - skipping shift analysis")

        print(f"\nSaved walking analysis results to {walking_dir}/")
        print(f"Timeline visualizations saved to {walking_dir}/timeline/")
        print(f"Shift comparison visualizations saved to {walking_dir}/shifts/")
    else:
        print("\nNo analysis type specified. Use --walk to analyze walking patterns.")
        return 0
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "=" * 40)
    print("=== Analysis Complete ===")
    print("=" * 40)
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"\nResults saved to: {output_path}")
    
    print("\nKey visualizations to check:")
    print(f"1. {walking_dir}/timeline/all_employees_walking_patterns_{args.time_slot}min.png - Combined walking patterns")
    
    employees = data['id'].unique()
    if len(employees) <= 5:  # Only list individual files if there aren't too many
        for emp_id in employees:
            print(f"2. {walking_dir}/timeline/employee_{emp_id}_walking_pattern_{args.time_slot}min.png - Walking pattern for employee {emp_id}")
    else:
        print(f"2. {walking_dir}/timeline/ - Individual employee walking patterns")
    
    print(f"3. {walking_dir}/total_walking_time.png - Overall walking summary")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())