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

from src.analysis.ergonomics import (
    calculate_ergonomic_score,
    generate_ergonomic_report,
    analyze_activity_durations,
    export_duration_factor_thresholds
)
from src.analysis.workflow import (
    analyze_walking_patterns,
    identify_workflow_patterns,
    analyze_productivity_rhythm,
    cluster_employee_work_patterns,
    identify_optimization_opportunities
)
from src.visualization.employee_region_vis import create_employee_region_heatmap
from src.visualization.region_activity_analysis import analyze_activities_by_region
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
    parser.add_argument('--analysisonly', action='store_true', help='Run advanced analysis without generating visualizations')
    parser.add_argument('--ergonomicsonly', action='store_true', help='Run only ergonomic analysis and reports')
    
    # Add step arguments
    parser.add_argument('--step1', action='store_true', help='Run shift analysis')
    parser.add_argument('--step2', action='store_true', help='Run activity analysis')
    parser.add_argument('--step3', action='store_true', help='Run region analysis')
    parser.add_argument('--step4', action='store_true', help='Run employee region heatmaps')
    parser.add_argument('--step5', action='store_true', help='Run statistical analysis')
    parser.add_argument('--step6', action='store_true', help='Run regional activity analysis')
    parser.add_argument('--step7', action='store_true', help='Run ergonomic analysis')
    parser.add_argument('--step8', action='store_true', help='Run workflow analysis')
    parser.add_argument('--employee', type=str, help='Specific employee ID to analyze (e.g., 32-A)')
    parser.add_argument('--department', type=str, choices=['Bread', 'Cake'], 
                      help='Department to analyze (Bread or Cake)')
    
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

def run_ergonomic_analysis(data, output_dir, employee_id=None):
    """
    Run ergonomic analysis and generate reports
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    output_dir : Path
        Output directory
    employee_id : str, optional
        If provided, analyze only this employee
    """
    # Create directory for ergonomic analysis results
    ergonomic_dir = ensure_dir_exists(output_dir / 'ergonomic_analysis')
    
    print("\n" + "=" * 40)
    print("=== Ergonomic Analysis ===")
    
    # Filter data for specific employee if provided
    if employee_id:
        if employee_id not in data['id'].unique():
            print(f"Error: Employee {employee_id} not found in the data")
            return
        filtered_data = data[data['id'] == employee_id]
        print(f"  Analyzing ergonomics for employee {employee_id}...")
    else:
        filtered_data = data
        print(f"  Analyzing ergonomics for all employees...")
    
    # Calculate ergonomic scores
    print("  Calculating ergonomic scores...")
    employee_scores = calculate_ergonomic_score(filtered_data)
    
    # Analyze duration statistics
    print("  Analyzing activity duration statistics...")
    activity_stats = analyze_activity_durations(filtered_data)
    
    # Export duration factor thresholds
    print("  Exporting duration factor thresholds...")
    export_duration_factor_thresholds(filtered_data, ergonomic_dir)
    
    # Convert activity statistics to DataFrame for saving
    activity_stats_data = []
    for activity, stats in activity_stats.items():
        activity_stats_data.append({
            'activity': activity,
            'count': stats['count'],
            'mean': stats['mean'],
            'median': stats['median'],
            'p25': stats['p25'],
            'p75': stats['p75'],
            'p90': stats['p90'],
            'p95': stats['p95'],
            'min': stats['min'],
            'max': stats['max']
        })
    
    # Save activity duration statistics
    activity_stats_df = pd.DataFrame(activity_stats_data)
    activity_stats_df.to_csv(ergonomic_dir / 'activity_duration_statistics.csv', index=False)
    print(f"  Saved activity duration statistics to ergonomic_analysis/activity_duration_statistics.csv")
    
    # Display key statistics for critical activities
    print("\nKey Duration Statistics (seconds):")
    print(f"{'Activity':<12} {'Count':>7} {'Median':>8} {'Mean':>8} {'P25':>6} {'P75':>6} {'P90':>6}")
    print("-" * 60)
    for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
        if activity in activity_stats:
            stats = activity_stats[activity]
            print(f"{activity:<12} {stats['count']:>7} {stats['median']:>8.1f} {stats['mean']:>8.1f} {stats['p25']:>6.1f} {stats['p75']:>6.1f} {stats['p90']:>6.1f}")
    
    # Display the threshold values being used
    if 'Handle up' in activity_stats:
        handle_up_median = activity_stats['Handle up']['median']
        print("\nThreshold values for Handle up:")
        print(f"  Short:  < {handle_up_median * 0.5:.1f}s (factor: 0.5)")
        print(f"  Medium: < {handle_up_median:.1f}s (factor: 1.0)")
        print(f"  Long:   < {handle_up_median * 2:.1f}s (factor: 1.5)")
        print(f"  Very long: >= {handle_up_median * 2:.1f}s (factor: 2.0)")
    
    # Create summary DataFrame
    summary_data = []
    for emp_score in employee_scores:
        summary_data.append({
            'employee_id': emp_score['employee_id'],
            'ergonomic_score': emp_score['final_score'],
            'total_hours': emp_score['total_duration'] / 3600,
            **{f"{act}_deduction": deduct for act, deduct in emp_score['components']['deductions'].items()}
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('ergonomic_score', ascending=False)
    summary_df.to_csv(ergonomic_dir / 'ergonomic_scores_summary.csv', index=False)
    print(f"  Saved ergonomic scores summary to ergonomic_analysis/ergonomic_scores_summary.csv")
    
    # Generate ergonomic reports
    print("  Generating ergonomic reports with visualizations...")
    generate_ergonomic_report(employee_scores, ergonomic_dir / 'reports')
    print(f"  Saved ergonomic reports to ergonomic_analysis/reports/")
    
    # Print scores overview
    print("\nErgonomic Scores Overview:")
    for _, row in summary_df.iterrows():
        score_color = "good" if row['ergonomic_score'] >= 75 else ("fair" if row['ergonomic_score'] >= 60 else "poor")
        print(f"  Employee {row['employee_id']}: {row['ergonomic_score']:.1f}/100 ({score_color})")
    
    print("\n=== Ergonomic Analysis Complete ===")

def run_advanced_analysis(data, output_dir):
    """
    Run advanced ergonomic and workflow analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    output_dir : Path
        Output directory
    """
    # Create directory for analysis results
    analysis_dir = ensure_dir_exists(output_dir / 'advanced_analysis')
    
    print("\n" + "=" * 40)
    print("=== Advanced Analysis ===")
    
    # 1. Ergonomic Analysis
    print("\n=== 1. Ergonomic Analysis ===")
    print("  Calculating ergonomic scores...")
    employee_scores = calculate_ergonomic_score(data)
    
    # Analyze duration statistics
    print("  Analyzing activity duration statistics...")
    activity_stats = analyze_activity_durations(data)
    
    # Export duration factor thresholds
    print("  Exporting duration factor thresholds...")
    export_duration_factor_thresholds(data, analysis_dir)
    
    # Convert activity statistics to DataFrame for saving
    activity_stats_data = []
    for activity, stats in activity_stats.items():
        activity_stats_data.append({
            'activity': activity,
            'count': stats['count'],
            'mean': stats['mean'],
            'median': stats['median'],
            'p25': stats['p25'],
            'p75': stats['p75'],
            'p90': stats['p90'],
            'p95': stats['p95'],
            'min': stats['min'],
            'max': stats['max']
        })
    
    # Save activity duration statistics
    activity_stats_df = pd.DataFrame(activity_stats_data)
    activity_stats_df.to_csv(analysis_dir / 'activity_duration_statistics.csv', index=False)
    print(f"  Saved activity duration statistics to advanced_analysis/activity_duration_statistics.csv")
    
    # Display key statistics for critical activities
    print("\nKey Duration Statistics (seconds):")
    print(f"{'Activity':<12} {'Count':>7} {'Median':>8} {'Mean':>8} {'P25':>6} {'P75':>6} {'P90':>6}")
    print("-" * 60)
    for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
        if activity in activity_stats:
            stats = activity_stats[activity]
            print(f"{activity:<12} {stats['count']:>7} {stats['median']:>8.1f} {stats['mean']:>8.1f} {stats['p25']:>6.1f} {stats['p75']:>6.1f} {stats['p90']:>6.1f}")
    
    # Convert ergonomic scores to DataFrame for saving
    employee_summary = []
    for emp_result in employee_scores:
        employee_summary.append({
            'employee_id': emp_result['employee_id'],
            'ergonomic_score': emp_result['final_score'],
            'total_hours': emp_result['total_duration'] / 3600,
            **{f"{act}_deduction": deduct for act, deduct in emp_result['components']['deductions'].items()}
        })
    
    summary_df = pd.DataFrame(employee_summary)
    summary_df.to_csv(analysis_dir / 'ergonomic_scores_summary.csv', index=False)
    print(f"  Saved ergonomic scores summary to advanced_analysis/ergonomic_scores_summary.csv")
    
    # Generate ergonomic reports
    print("  Generating ergonomic reports with visualizations...")
    generate_ergonomic_report(employee_scores, analysis_dir / 'ergonomic_reports')
    print(f"  Saved ergonomic reports to advanced_analysis/ergonomic_reports/")
    
    # 2. Workflow Analysis
    print("\n=== 2. Workflow Analysis ===")
    print("  Analyzing walking patterns...")
    walking_patterns = analyze_walking_patterns(data)
    
    # Convert walking patterns to DataFrame
    walking_data = []
    for emp_id, patterns in walking_patterns.items():
        walking_data.append({
            'employee_id': emp_id,
            'total_walking_time': patterns['total_walking_time'],
            'avg_walk_duration': patterns['avg_walk_duration'],
            'long_walks_count': patterns['long_walks_count'],
            'long_walks_total_time': patterns['long_walks_total_time'],
            'most_common_long_walk': str(patterns['common_long_walks'][0]) if patterns['common_long_walks'] else 'None'
        })
    
    walking_df = pd.DataFrame(walking_data)
    walking_df.to_csv(analysis_dir / 'walking_patterns.csv', index=False)
    print(f"  Saved walking pattern analysis to advanced_analysis/walking_patterns.csv")
    
    # Productivity rhythm analysis
    print("  Analyzing productivity rhythms...")
    productivity_data = analyze_productivity_rhythm(data)
    productivity_data.to_csv(analysis_dir / 'productivity_rhythm.csv', index=False)
    print(f"  Saved productivity rhythm analysis to advanced_analysis/productivity_rhythm.csv")
    
    # Employee work pattern clustering
    print("  Clustering employee work patterns...")
    clustering_results = cluster_employee_work_patterns(data)
    
    if 'error' not in clustering_results:
        cluster_df = clustering_results['employee_clusters']
        cluster_df.to_csv(analysis_dir / 'employee_clusters.csv', index=False)
        print(f"  Saved employee clustering results to advanced_analysis/employee_clusters.csv")
        
        # Save cluster profiles
        with open(analysis_dir / 'cluster_profiles.txt', 'w') as f:
            f.write("Employee Work Pattern Clusters\n")
            f.write("="*80 + "\n\n")
            
            for cluster_id, profile in clustering_results['cluster_profiles'].items():
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Employees: {', '.join(profile['employees'])}\n")
                f.write(f"  Size: {profile['size']} employees\n")
                f.write("  Characteristics:\n")
                
                for feature, value in profile['avg_features'].items():
                    f.write(f"    - {feature}: {value:.2f}\n")
                f.write("\n")
        
        print(f"  Saved cluster profiles to advanced_analysis/cluster_profiles.txt")
    else:
        print(f"  Error in clustering: {clustering_results['error']}")
    
    # Workflow pattern analysis
    print("  Identifying workflow patterns...")
    workflow_patterns = identify_workflow_patterns(data)
    
    if 'error' not in workflow_patterns:
        # Save detailed patterns for each employee
        for emp_id, patterns in workflow_patterns.items():
            if 'patterns' in patterns:
                patterns_df = patterns['patterns']
                if not patterns_df.empty:
                    patterns_df.to_csv(analysis_dir / f'workflow_patterns_{emp_id}.csv', index=False)
                    print(f"  Saved workflow patterns for {emp_id} to advanced_analysis/workflow_patterns_{emp_id}.csv")
        
        # Generate optimization recommendations
        print("  Generating optimization recommendations...")
        optimization_opportunities = identify_optimization_opportunities(
            walking_patterns, productivity_data, workflow_patterns)
        
        # Save optimization report
        with open(analysis_dir / 'optimization_recommendations.txt', 'w') as f:
            f.write("Bakery Workflow Optimization Recommendations\n")
            f.write("="*80 + "\n\n")
            
            f.write("Movement Optimization:\n")
            for rec in optimization_opportunities['movement_optimization']:
                f.write(f"  - [{rec['priority']}] {rec['employee']}: {rec['recommendation']}\n")
            f.write("\n")
            
            f.write("Workflow Optimization:\n")
            for rec in optimization_opportunities['workflow_optimization']:
                f.write(f"  - [{rec['priority']}] {rec['employee']}: {rec['recommendation']}\n")
            f.write("\n")
            
            f.write("Scheduling Recommendations:\n")
            for rec in optimization_opportunities['scheduling_recommendations']:
                f.write(f"  - [{rec['priority']}] {rec['target']}: {rec['recommendation']}\n")
        
        print(f"  Saved optimization recommendations to advanced_analysis/optimization_recommendations.txt")
    else:
        print(f"  Error in workflow pattern analysis: {workflow_patterns['error']}")
    
    print("\n=== Advanced Analysis Complete ===")

def main():
    """Main function to orchestrate the analysis"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if any step is specified, otherwise run all steps
    run_all = not (args.step1 or args.step2 or args.step3 or args.step4 or 
                  args.step5 or args.step6 or args.step7 or args.step8 or
                  args.analysisonly or args.ergonomicsonly)
    
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
    
    # If ergonomicsonly is specified, just run the ergonomic analysis and exit
    if args.ergonomicsonly:
        run_ergonomic_analysis(data, output_path, args.employee)
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 40)
        print("=== Analysis Complete ===")
        print("=" * 40)
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"\nResults saved to: {output_path}/ergonomic_analysis/")
        return 0
    
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
        
        # Activity profile by employee
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
    
    # 6. Regional Activity Analysis
    if run_all or args.step6:
        print("\n" + "=" * 40)
        print("=== 6. Regional Activity Analysis ===")
        
        regional_analysis_dir = ensure_dir_exists(vis_dir / 'regional_activity_analysis')
        
        # If department is specified, analyze that department
        if args.department:
            department = args.department
            print(f"  Analyzing activities by region for {department} department...")
            fig = analyze_activities_by_region(
                data, 
                department=department, 
                top_n_regions=5,
                save_path=regional_analysis_dir / f"{department.lower()}_regional_activity_analysis.png"
            )
            print(f"  Saved {department} department regional activity analysis to visualizations/regional_activity_analysis")
        # Otherwise, analyze both departments
        else:
            for department in ['Bread', 'Cake']:
                print(f"  Analyzing activities by region for {department} department...")
                fig = analyze_activities_by_region(
                    data, 
                    department=department, 
                    top_n_regions=5,
                    save_path=regional_analysis_dir / f"{department.lower()}_regional_activity_analysis.png"
                )
            print(f"  Saved regional activity analyses to visualizations/regional_activity_analysis")
    
    # 7. Ergonomic Analysis
    if run_all or args.step7:
        print("\n" + "=" * 40)
        print("=== 7. Ergonomic Analysis ===")
        
        run_ergonomic_analysis(data, output_path, args.employee)
    
    # 8. Workflow Analysis
    if run_all or args.step8:
        print("\n" + "=" * 40)
        print("=== 8. Workflow Analysis ===")
        
        workflow_dir = ensure_dir_exists(stats_dir / 'workflow')
        
        # Analyze walking patterns
        walking_patterns = analyze_walking_patterns(data)
        
        # Convert to DataFrame
        walking_data = []
        for emp_id, patterns in walking_patterns.items():
            walking_data.append({
                'employee_id': emp_id,
                'total_walking_time': patterns['total_walking_time'],
                'avg_walk_duration': patterns['avg_walk_duration'],
                'long_walks_count': patterns['long_walks_count'],
                'long_walks_total_time': patterns['long_walks_total_time'],
            })
        
        walking_df = pd.DataFrame(walking_data)
        walking_df.to_csv(workflow_dir / 'walking_patterns.csv', index=False)
        print(f"  Saved walking pattern analysis to statistics/workflow/walking_patterns.csv")
    
    # Run full advanced analysis if analysisonly flag is set
    if args.analysisonly:
        run_advanced_analysis(data, output_path)
    
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
    if run_all or args.step6:
        print(f"6. {vis_dir}/regional_activity_analysis/ - Activity breakdown by region per employee")
    if run_all or args.step7:
        print(f"7. {output_path}/ergonomic_analysis/reports/ - Ergonomic assessment reports")
    if args.ergonomicsonly:
        print(f"• {output_path}/ergonomic_analysis/reports/ - Ergonomic assessment reports")
        print(f"• {output_path}/ergonomic_analysis/ergonomic_scores_summary.csv - Scores summary")
        print(f"• {output_path}/ergonomic_analysis/activity_duration_statistics.csv - Activity duration data")
        print(f"• {output_path}/ergonomic_analysis/duration_factor_thresholds.txt - Threshold details")

if __name__ == "__main__":
    sys.exit(main())