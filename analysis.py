#!/usr/bin/env python3
"""
Bakery Employee Analysis
======================

This script performs a comprehensive analysis of bakery employee activities,
including heatmaps, regional activity patterns, and cart usage analysis.

Usage:
    python analysis.py [--steps 1,2,3] [--help]

Options:
    --steps STEPS    Comma-separated list of steps to run (1-3)
    --help           Show this help message

Steps:
    1. Employee activity heatmap and CSV export
    2. Regional activity pattern analysis
    3. Cart usage and idle time analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import argparse
from pathlib import Path
import os
from typing import List, Set

# Define output directories
OUTPUT_DIR = Path('analysis_results')
STEP_DIRS = {
    'step1': OUTPUT_DIR / 'step1_heatmap_analysis',
    'step2': OUTPUT_DIR / 'step2_regional_activity',
    'step3': OUTPUT_DIR / 'step3_cart_analysis'
}

# Import your existing utility functions
from src.utils.data_utils import load_sensor_data, classify_departments
from src.utils.file_utils import ensure_dir_exists
from src.utils.time_utils import format_seconds_to_hms
from src.visualization.activity_vis import plot_activity_distribution_by_employee
from src.visualization.base import set_visualization_style, get_activity_order

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bakery Employee Analysis")
    parser.add_argument(
        '--steps',
        type=str,
        default='1,2,3',
        help='Comma-separated list of steps to run (1-3)'
    )
    return parser.parse_args()

def validate_steps(steps_str: str) -> Set[int]:
    """Validate and parse the steps argument"""
    try:
        steps = {int(step.strip()) for step in steps_str.split(',') if step.strip()}
        if not all(1 <= step <= 3 for step in steps):
            raise ValueError("Step numbers must be between 1 and 3")
        return steps
    except ValueError as e:
        print(f"❌ Invalid steps argument: {e}")
        print("Please provide a comma-separated list of numbers between 1 and 3")
        sys.exit(1)

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

def run_step2_analysis():
    """Run the Step 2 analysis which generates regional activity pattern graphs"""
    print_step_header(2, "REGIONAL ACTIVITY PATTERN GRAPHS")
    
    print("🚀 Starting Step 2 Analysis...")
    print("This will generate 35 graphs (7 employees × 5 activities)")
    print("Each graph shows regional activity patterns over 30-minute intervals")
    print()
    
    try:
        # Check if the analysis script exists
        script_path = Path("employee_activity_analysis.py")
        if not script_path.exists():
            print("❌ Error: employee_activity_analysis.py not found!")
            print("Please make sure the script is in the current directory.")
            return 1
        
        # Get the output directory for step 2
        step2_output_dir = STEP_DIRS['step2']
        step2_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the analysis with high-quality settings
        print(f"🔄 Running employee_activity_analysis.py with high-quality settings...")
        print(f"📂 Output will be saved to: {step2_output_dir.absolute()}")
        
        # Set the environment variable for the subprocess
        env = os.environ.copy()
        env['STEP2_OUTPUT_DIR'] = str(step2_output_dir.absolute())
        
        result = subprocess.run([
            sys.executable, 
            str(script_path),
            "--dpi", "600",
            "--width", "14",
            "--height", "10",
            "--font-size", "12"
        ], capture_output=True, text=True, env=env)
        
        # Print the output from the script
        if result.stdout:
            print("\n" + result.stdout)
        if result.stderr:
            print("\n" + result.stderr, file=sys.stderr)
            
        if result.returncode == 0:
            print("\n✅ Step 2 Analysis completed successfully!")
            print(f"📁 Results saved to: {step2_output_dir.absolute()}")
        else:
            print(f"\n❌ Analysis failed with return code: {result.returncode}")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

def run_step3_analysis():
    """Run the Step 3 analysis which analyzes cart usage and idle times"""
    print_step_header(3, "CART USAGE AND IDLE TIME ANALYSIS")
    print("This will analyze cart usage patterns and idle times")
    print("• Processes cart movement data")
    print("• Identifies idle periods")
    print("• Generates visualizations of cart usage")
    print()
    
    try:
        # Get the output directory for step 3 and create subdirectories
        step3_output_dir = STEP_DIRS['step3']
        cart_data_dir = step3_output_dir / 'cart_data'
        cart_analysis_dir = step3_output_dir / 'cart_analysis'
        
        # Create all necessary directories
        for directory in [step3_output_dir, cart_data_dir, cart_analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Step 3 output will be saved to: {step3_output_dir.absolute()}")
        
        # Set environment variables for all subprocesses
        env = os.environ.copy()
        env['STEP3_OUTPUT_DIR'] = str(step3_output_dir.absolute())
        env['CART_DATA_DIR'] = str(cart_data_dir.absolute())
        env['IDLE_ANALYSIS_DIR'] = str(cart_analysis_dir.absolute())
        
        # Step 3.1: Run cart.py to process cart data
        print("\n🚚 Step 3.1/3: Processing cart data...")
        cart_script = Path("cart.py")
        if not cart_script.exists():
            print("❌ Error: cart.py not found!")
            return 1
            
        cart_result = subprocess.run(
            [sys.executable, str(cart_script)],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Print the output from cart.py
        if cart_result.stdout:
            print("\n" + cart_result.stdout)
        if cart_result.stderr:
            print("\n" + cart_result.stderr, file=sys.stderr)
        if cart_result.returncode != 0:
            print(f"❌ cart.py failed with return code {cart_result.returncode}")
            return cart_result.returncode
            
        # Step 3.2: Run idle_cart.py to analyze idle times
        print("\n⏱️  Step 3.2/3: Analyzing cart idle times...")
        idle_script = Path("idle_cart.py")
        if not idle_script.exists():
            print("❌ Error: idle_cart.py not found!")
            return 1
            
        idle_result = subprocess.run(
            [sys.executable, str(idle_script)],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Print the output from idle_cart.py
        if idle_result.stdout:
            print("\n" + idle_result.stdout)
        if idle_result.stderr:
            print("\n" + idle_result.stderr, file=sys.stderr)
        if idle_result.returncode != 0:
            print(f"❌ idle_cart.py failed with return code {idle_result.returncode}")
            return idle_result.returncode
            
        # Step 3.3: Run cart_usage_analysis.py to generate visualizations
        print("\n📊 Step 3.3/3: Generating cart usage visualizations...")
        analysis_script = Path("cart_usage_analysis.py")
        if not analysis_script.exists():
            print("❌ Error: cart_usage_analysis.py not found!")
            return 1
            
        analysis_result = subprocess.run(
            [sys.executable, str(analysis_script)],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Print the output from cart.py
        if cart_result.stdout:
            print("\n" + cart_result.stdout)
        if cart_result.stderr:
            print("\n" + cart_result.stderr, file=sys.stderr)
        if cart_result.returncode != 0:
            print(f"❌ cart.py failed with return code {cart_result.returncode}")
            return cart_result.returncode
            
        # Step 3.2: Run idle_cart.py to analyze idle times
        print("\n⏱️  Step 3.2/3: Analyzing cart idle times...")
        idle_script = Path("idle_cart.py")
        if not idle_script.exists():
            print("❌ Error: idle_cart.py not found!")
            return 1
            
        idle_result = subprocess.run(
            [sys.executable, str(idle_script)],
            capture_output=True,
            text=True,
            env=env  # Use the same environment with STEP3_OUTPUT_DIR
        )
        
        # Print the output from idle_cart.py
        if idle_result.stdout:
            print("\n" + idle_result.stdout)
        if idle_result.stderr:
            print("\n" + idle_result.stderr, file=sys.stderr)
        if idle_result.returncode != 0:
            print(f"❌ idle_cart.py failed with return code {idle_result.returncode}")
            return idle_result.returncode
            
        # Step 3.3: Run cart_usage_analysis.py to generate visualizations
        print("\n📊 Step 3.3/3: Generating cart usage visualizations...")
        analysis_script = Path("cart_usage_analysis.py")
        if not analysis_script.exists():
            print("❌ Error: cart_usage_analysis.py not found!")
            return 1
            
        analysis_result = subprocess.run(
            [sys.executable, str(analysis_script)],
            capture_output=True,
            text=True,
            env=env  # Use the same environment with STEP3_OUTPUT_DIR
        )
        
        # Print the output from cart_usage_analysis.py
        if analysis_result.stdout:
            print("\n" + analysis_result.stdout)
        if analysis_result.stderr:
            print("\n" + analysis_result.stderr, file=sys.stderr)
            
        if analysis_result.returncode == 0:
            print("\n✅ Step 3 Analysis completed successfully!")
            print(f"📁 Results saved to: {step3_output_dir.absolute()}")
        else:
            print(f"\n❌ Analysis failed with return code: {analysis_result.returncode}")
            return analysis_result.returncode
            
        print("\n✅ Step 3 analysis completed successfully!")
        print(f"📊 Generated cart usage and idle time analysis")
        print(f"💾 Files saved to: {step3_output_dir.absolute()}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in subprocess: {str(e)}")
        if hasattr(e, 'stdout') and e.stdout:
            print("\nSTDOUT:" + e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("\nSTDERR:" + e.stderr, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error during cart analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def main():
    """Main analysis function"""
    print("🏭 BAKERY EMPLOYEE ACTIVITY ANALYSIS")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_arguments()
    steps_to_run = validate_steps(args.steps)
    
    # Ensure output directories exist
    for dir_path in STEP_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Output will be saved in: {OUTPUT_DIR.absolute()}")
    for step, dir_path in STEP_DIRS.items():
        print(f"   • {step}: {dir_path.name}")
    
    # Load the data if any step needs it
    if any(step in steps_to_run for step in [1, 2]):
        print("\n📁 Loading data...")
        data = load_sensor_data("data/raw/processed_sensor_data.csv")
        if data is None or data.empty:
            print("❌ Failed to load data. Exiting.")
            return 1
        
        # Classify departments
        data = classify_departments(data)
        print(f"📊 Dataset loaded: {len(data):,} records from {data['id'].nunique()} employees")
    else:
        data = None
    
    # Run requested steps
    if 1 in steps_to_run:
        create_employee_heatmap_and_csv(data, STEP_DIRS['step1'])
    
    if 2 in steps_to_run:
        run_step2_analysis()
    
    if 3 in steps_to_run:
        run_step3_analysis()
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Only show summary for steps that were run
    if steps_to_run == {1, 2, 3}:
        print("📊 Ran all analysis steps")
    else:
        print("📊 Ran steps:", ", ".join(str(step) for step in sorted(steps_to_run)))
    
    print("\n📊 ANALYSIS RESULTS SAVED TO:")
    for step, dir_path in STEP_DIRS.items():
        if int(step.replace('step', '')) in steps_to_run:
            print(f"   • {step.capitalize()}: {dir_path.absolute()}")
    
    print(f"\n✅ All analysis results are organized in: {OUTPUT_DIR.absolute()}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)