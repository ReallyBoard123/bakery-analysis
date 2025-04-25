"""
Activity Analysis Main Module

Main entry point for specialized activity analysis in the bakery.
This orchestrates the analysis of walking, standing, and different handling activities.
"""

import pandas as pd
import time
import argparse
from pathlib import Path
import sys

from src.utils.file_utils import ensure_dir_exists
from src.utils.data_utils import load_sensor_data, classify_departments
from src.utils.time_utils import format_seconds_to_hms

# Import walking analysis module
from src.analysis.activity.walking import analyze_walking_patterns

# Import other activity-specific analysis modules
from src.analysis.activity.stand import analyze_standing_patterns
from src.analysis.activity.up import analyze_handling_up
from src.analysis.activity.down import analyze_handling_down

def parse_arguments():
    """Parse command line arguments specific to activity analysis"""
    parser = argparse.ArgumentParser(description='Analyze bakery employee activities')
    
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                      help='Path to the data file')
    parser.add_argument('--output', type=str, default='output/activity_analysis',
                      help='Directory to save output files')
    parser.add_argument('--walking', action='store_true', help='Run walking analysis')
    parser.add_argument('--standing', action='store_true', help='Run standing analysis')
    parser.add_argument('--handle_up', action='store_true', help='Run handle-up analysis')
    parser.add_argument('--handle_down', action='store_true', help='Run handle-down analysis')
    parser.add_argument('--employee', type=str, help='Specific employee ID to analyze')
    parser.add_argument('--focus_employees', type=str, nargs='+', 
                      help='List of employee IDs to focus on (e.g., 32-D 32-G)')
    parser.add_argument('--department', type=str, choices=['Bread', 'Cake'], 
                      help='Department to analyze')
    parser.add_argument('--exploratory', action='store_true', 
                      help='Run exploratory analysis before detailed analysis')
    parser.add_argument('--german', action='store_true', 
                      help='Generate visualizations and reports in German')
    parser.add_argument('--detailed_walking', action='store_true',
                      help='Run enhanced walking analysis with detailed breakdowns')
    
    return parser.parse_args()

def run_exploratory_analysis(data, output_dir):
    """
    Run initial exploratory analysis on the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path
        Output directory
    """
    from src.analysis.activity.exploratory import (
        analyze_activity_durations,
        analyze_region_transitions,
        analyze_time_in_region,
        print_exploratory_summary
    )
    
    # Create directory for exploratory results
    exploratory_dir = ensure_dir_exists(output_dir / 'exploratory')
    
    print("\n=== Running Exploratory Analysis ===")
    
    # Print basic summary
    print_exploratory_summary(data)
    
    # Analyze durations by activity type
    duration_stats = analyze_activity_durations(data)
    
    # Save duration stats
    pd.DataFrame(duration_stats['overall']).T.to_csv(
        exploratory_dir / 'activity_duration_stats.csv'
    )
    
    # Analyze region transitions
    transition_stats = analyze_region_transitions(data)
    
    # Save transition data
    with open(exploratory_dir / 'region_transitions.txt', 'w') as f:
        for emp_id, stats in transition_stats.items():
            f.write(f"Employee {emp_id}:\n")
            f.write(f"  Total region transitions: {stats['total_transitions']}\n")
            f.write(f"  Walk transitions: {stats['walk_transitions']} ({stats['walk_percentage']:.2f}%)\n")
            
            if stats['common_paths']:
                f.write("  Most common paths:\n")
                for path, count in stats['common_paths']:
                    from_region, to_region = path
                    f.write(f"    {from_region} → {to_region}: {count} times\n")
            
            f.write("\n")
    
    # Analyze time in region
    region_time_stats = analyze_time_in_region(data)
    
    # Save region time stats
    regions_df = pd.DataFrame([
        {
            'region': region,
            'visits': stats['count'],
            'avg_duration': stats['mean'],
            'median_duration': stats['median'],
            'min_duration': stats['min'],
            'max_duration': stats['max']
        }
        for region, stats in region_time_stats['overall'].items()
    ])
    
    if not regions_df.empty:
        regions_df.to_csv(exploratory_dir / 'time_in_region.csv', index=False)
    
    print(f"Exploratory analysis complete. Results saved to {exploratory_dir}")

def main():
    """Main function to orchestrate the activity analysis"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine language
    language = 'de' if args.german else 'en'
    
    # Check if any analysis is specified, otherwise run all
    run_all = not (args.walking or args.standing or args.handle_up or args.handle_down or args.detailed_walking)
    
    # Create output directory
    output_path = ensure_dir_exists(args.output)
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_sensor_data(args.data)
    
    # Add department classification if not already present
    if 'department' not in data.columns:
        data = classify_departments(data)
    
    # Filter out Unknown activity
    data = data[data['activity'] != 'Unknown']
    
    # Display basic information
    print("\nDataset Information:")
    print(f"  Records: {len(data):,}")
    print(f"  Employees: {data['id'].nunique()} - {sorted(data['id'].unique())}")
    print(f"  Activities: {data['activity'].nunique()} - {sorted(data['activity'].unique())}")
    
    # Total duration
    total_seconds = data['duration'].sum()
    total_formatted = format_seconds_to_hms(total_seconds)
    total_hours = total_seconds / 3600
    print(f"  Total duration: {total_formatted} ({total_hours:.2f} hours)")
    
    # Filter by employee or department if specified
    if args.employee:
        data = data[data['id'] == args.employee]
        print(f"Filtered to employee {args.employee}: {len(data)} records")
    
    if args.department:
        data = data[data['department'] == args.department]
        print(f"Filtered to {args.department} department: {len(data)} records")
    
    # Set up focus employees list
    focus_employees = None
    if args.focus_employees:
        focus_employees = args.focus_employees
        print(f"Focus analysis on employees: {', '.join(focus_employees)}")
    elif args.employee:
        focus_employees = [args.employee]
    
    # Run exploratory analysis if requested
    if args.exploratory:
        run_exploratory_analysis(data, output_path)
    
    # Run walking analysis
    if run_all or args.walking:
        print("\n=== Running Standard Walking Analysis ===")
        walking_output = ensure_dir_exists(output_path / 'walking')
        analyze_walking_patterns(data, walking_output, language=language)
    
    # Run detailed walking analysis if requested
    if run_all or args.detailed_walking:
        print("\n=== Running Enhanced Walking Analysis ===")
        detailed_walking_output = ensure_dir_exists(output_path / 'enhanced_walking')
        
        # Import the enhanced walking analysis function
        from src.analysis.activity.enhanced_walking import analyze_walking_patterns as analyze_enhanced_walking
        
        analyze_enhanced_walking(data, detailed_walking_output, language=language, focus_employees=focus_employees)
        print(f"Enhanced walking analysis complete. Results saved to {detailed_walking_output}")
    
    # Run standing analysis
    if run_all or args.standing:
        print("\n=== Running Standing Analysis ===")
        standing_output = ensure_dir_exists(output_path / 'standing')
        analyze_standing_patterns(data, standing_output, language=language)
    
    # Run handle-up analysis
    if run_all or args.handle_up:
        print("\n=== Running Handle-Up Analysis ===")
        up_output = ensure_dir_exists(output_path / 'handle_up')
        analyze_handling_up(data, up_output, language=language)
    
    # Run handle-down analysis
    if run_all or args.handle_down:
        print("\n=== Running Handle-Down Analysis ===")
        down_output = ensure_dir_exists(output_path / 'handle_down')
        analyze_handling_down(data, down_output, language=language)
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n=== Activity Analysis Complete ===")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    
    # Print usage instructions for the enhanced walking analysis
    if not args.detailed_walking and not run_all:
        print("\nTo run enhanced walking analysis with detailed breakdowns:")
        print(f"  python activity.py --detailed_walking --focus_employees 32-D 32-G --output {args.output}")

if __name__ == "__main__":
    sys.exit(main())