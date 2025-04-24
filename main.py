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
    generate_all_employees_comparison,
    generate_all_regions_comparison,
    generate_employee_handling_comparison_by_region,
    generate_ergonomic_report,
    analyze_activity_durations,
    export_duration_factor_thresholds,
    analyze_region_ergonomics,
    generate_region_ergonomic_report,
    run_handling_time_analysis
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

from src.utils.translation_utils import get_translation, translate_format_string

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
    parser.add_argument('--handlinganalysis', action='store_true', help='Run handling time position analysis')
    parser.add_argument('--german', action='store_true', help='Generate visualizations and reports in German')
    
    # Add step arguments
    parser.add_argument('--step1', action='store_true', help='Run shift analysis')
    parser.add_argument('--step2', action='store_true', help='Run activity analysis')
    parser.add_argument('--step3', action='store_true', help='Run region analysis')
    parser.add_argument('--step4', action='store_true', help='Run employee region heatmaps')
    parser.add_argument('--step5', action='store_true', help='Run statistical analysis')
    parser.add_argument('--step6', action='store_true', help='Run regional activity analysis')
    parser.add_argument('--step7', action='store_true', help='Run ergonomic analysis')
    parser.add_argument('--step8', action='store_true', help='Run workflow analysis')
    parser.add_argument('--step9', action='store_true', help='Run handling time analysis')
    parser.add_argument('--employee', type=str, help='Specific employee ID to analyze (e.g., 32-A)')
    parser.add_argument('--department', type=str, choices=['Bread', 'Cake'], 
                      help='Department to analyze (Bread or Cake)')
    
    return parser.parse_args()

def run_employee_heatmaps(data, floor_plan_data, output_dir, employee_id=None, language='en'):
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
    language : str, optional
        Language code ('en' or 'de')
    """
    # Create directory for employee region heatmaps
    employee_heatmaps_dir = ensure_dir_exists(output_dir / 'employee_heatmaps')
    
    # Get employees to process
    if employee_id:
        if employee_id not in data['id'].unique():
            print(get_translation('Error: Employee {0} not found in the data', language).format(employee_id))
            return
        employees = [employee_id]
    else:
        employees = sorted(data['id'].unique())
    
    print(f"\n{get_translation('Generating heatmaps for {0} employees:', language).format(len(employees))}")
    
    # Generate region heatmap for each employee
    for emp_id in employees:
        print(f"  {get_translation('Processing employee {0}...', language).format(emp_id)}")
        
        # Create directory for this employee
        emp_dir = ensure_dir_exists(employee_heatmaps_dir / emp_id)
        
        # Generate region heatmap
        create_employee_region_heatmap(
            data,
            floor_plan_data,
            emp_id,
            save_path=emp_dir / f"{emp_id}_region_heatmap.png",
            top_n=10,  # Display top 10 regions
            min_transitions=2,  # Include transitions with at least 2 occurrences
            language=language
        )
    
    print(f"  {get_translation('Saved employee region heatmaps to {0}/employee_heatmaps/', language).format(output_dir)}")

def run_ergonomic_analysis(data, output_dir, employee_id=None, language='en'):
    """
    Run ergonomic analysis and generate employee and region reports
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    output_dir : Path
        Output directory
    employee_id : str, optional
        If provided, analyze only this employee
    language : str, optional
        Language code ('en' or 'de')
    """
    # Create directory for ergonomic analysis results
    ergonomic_dir = ensure_dir_exists(output_dir / 'ergonomic_analysis')
    employee_reports_dir = ensure_dir_exists(ergonomic_dir / 'employee_reports')
    region_reports_dir = ensure_dir_exists(ergonomic_dir / 'region_reports')
    
    print("\n" + "=" * 40)
    print(f"=== {get_translation('Ergonomic Analysis', language)} ===")
    
    # Filter data for specific employee if provided
    if employee_id:
        if employee_id not in data['id'].unique():
            print(get_translation('Error: Employee {0} not found in the data', language).format(employee_id))
            return
        filtered_data = data[data['id'] == employee_id]
        print(f"  {get_translation('Analyzing ergonomics for employee {0}...', language).format(employee_id)}")
    else:
        filtered_data = data
        print(f"  {get_translation('Analyzing ergonomics for all employees...', language)}")
    
    # Calculate ergonomic scores
    print(f"  {get_translation('Calculating ergonomic scores...', language)}")
    employee_scores = calculate_ergonomic_score(filtered_data)
    
    # Analyze duration statistics
    print(f"  {get_translation('Analyzing activity duration statistics...', language)}")
    activity_stats = analyze_activity_durations(filtered_data)
    
    # Export duration factor thresholds
    print(f"  {get_translation('Exporting duration factor thresholds...', language)}")
    export_duration_factor_thresholds(filtered_data, ergonomic_dir, language)
    
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
    print(f"  {get_translation('Saved activity duration statistics to ergonomic_analysis/activity_duration_statistics.csv', language)}")
    
    # Create summary DataFrame for employees
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
    print(f"  {get_translation('Saved ergonomic scores summary to ergonomic_analysis/ergonomic_scores_summary.csv', language)}")
    
    # Generate employee ergonomic reports
    print(f"  {get_translation('Generating employee ergonomic reports...', language)}")
    generate_ergonomic_report(employee_scores, employee_reports_dir, language)
    print(f"  {get_translation('Saved employee ergonomic reports to ergonomic_analysis/employee_reports/', language)}")
    
    # After generating individual employee reports
    if not employee_id:
        print(f"  {get_translation('Generating employee comparison visualizations...', language)}")
        generate_all_employees_comparison(employee_scores, ergonomic_dir, language)
    
    # Analyze region ergonomics (skip if analyzing just one employee)
    if not employee_id:
        print(f"  {get_translation('Analyzing region ergonomics...', language)}")
        region_analyses = analyze_region_ergonomics(data, min_percentage=5, min_duration=2)
        
        # Generate region ergonomic reports
        print(f"  {get_translation('Generating region ergonomic reports...', language)}")
        generate_region_ergonomic_report(region_analyses, region_reports_dir, language)
        print(f"  {get_translation('Saved region ergonomic reports to ergonomic_analysis/region_reports/', language)}")
        
        # Save region ergonomic summary
        region_summary = []
        for region, analysis in region_analyses.items():
            region_summary.append({
                'region': region,
                'ergonomic_score': analysis['ergonomic_score'],
                'total_hours': analysis['total_duration'] / 3600,
                'qualified_employees': analysis['qualified_employees'],
                'handle_up_pct': analysis['activity_distribution'].get('Handle up', {}).get('percentage', 0),
                'handle_down_pct': analysis['activity_distribution'].get('Handle down', {}).get('percentage', 0),
                'handle_center_pct': analysis['activity_distribution'].get('Handle center', {}).get('percentage', 0),
            })
        
        region_summary_df = pd.DataFrame(region_summary)
        region_summary_df = region_summary_df.sort_values('ergonomic_score', ascending=False)
        region_summary_df.to_csv(ergonomic_dir / 'region_ergonomic_summary.csv', index=False)
        print(f"  {get_translation('Saved region ergonomic summary to ergonomic_analysis/region_ergonomic_summary.csv', language)}")
        
        # Generate region comparison visualizations
        print(f"  {get_translation('Generating region comparison visualizations...', language)}")
        generate_all_regions_comparison(region_analyses, ergonomic_dir, language)
        
        # Generate employee handling comparison by region
        print(f"  {get_translation('Generating employee handling comparison by region...', language)}")
        generate_employee_handling_comparison_by_region(data, region_analyses, ergonomic_dir, language)
        print(f"  {get_translation('Saved employee handling comparison by region to ergonomic_analysis/region_employee_comparisons/', language)}")
    
    # Print scores overview
    print(f"\n{get_translation('Ergonomic Scores Overview:', language)}")
    for _, row in summary_df.iterrows():
        score_color = "good" if row['ergonomic_score'] >= 75 else ("fair" if row['ergonomic_score'] >= 60 else "poor")
        print(f"  {get_translation('Employee {0}: {1:.1f}/100 ({2})', language).format(row['employee_id'], row['ergonomic_score'], score_color)}")
    
    print(f"\n=== {get_translation('Ergonomic Analysis Complete', language)} ===")

def run_advanced_analysis(data, output_dir, language='en'):
    """
    Run advanced ergonomic and workflow analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    output_dir : Path
        Output directory
    language : str, optional
        Language code ('en' or 'de')
    """
    # Create directory for analysis results
    analysis_dir = ensure_dir_exists(output_dir / 'advanced_analysis')
    
    print("\n" + "=" * 40)
    print(f"=== {get_translation('Advanced Analysis', language)} ===")
    
    # 1. Ergonomic Analysis
    print(f"\n=== 1. {get_translation('Ergonomic Analysis', language)} ===")
    print(f"  {get_translation('Calculating ergonomic scores...', language)}")
    employee_scores = calculate_ergonomic_score(data)
    
    # Analyze duration statistics
    print(f"  {get_translation('Analyzing activity duration statistics...', language)}")
    activity_stats = analyze_activity_durations(data)
    
    # Export duration factor thresholds
    print(f"  {get_translation('Exporting duration factor thresholds...', language)}")
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
    print(f"  {get_translation('Saved activity duration statistics to advanced_analysis/activity_duration_statistics.csv', language)}")
    
    # Display key statistics for critical activities
    print(f"\n{get_translation('Key Duration Statistics (seconds):', language)}")
    print(f"{'Activity':<12} {'Count':>7} {'Median':>8} {'Mean':>8} {'P25':>6} {'P75':>6} {'P90':>6}")
    print("-" * 60)
    for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
        if activity in activity_stats:
            stats = activity_stats[activity]
            print(f"{get_translation(activity, language):<12} {stats['count']:>7} {stats['median']:>8.1f} {stats['mean']:>8.1f} {stats['p25']:>6.1f} {stats['p75']:>6.1f} {stats['p90']:>6.1f}")
    
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
    print(f"  {get_translation('Saved ergonomic scores summary to advanced_analysis/ergonomic_scores_summary.csv', language)}")
    
    # Generate ergonomic reports
    print(f"  {get_translation('Generating ergonomic reports with visualizations...', language)}")
    generate_ergonomic_report(employee_scores, analysis_dir / 'ergonomic_reports')
    print(f"  {get_translation('Saved ergonomic reports to advanced_analysis/ergonomic_reports/', language)}")
    
    # 2. Workflow Analysis
    print(f"\n=== 2. {get_translation('Workflow Analysis', language)} ===")
    print(f"  {get_translation('Analyzing walking patterns...', language)}")
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
    print(f"  {get_translation('Saved walking pattern analysis to advanced_analysis/walking_patterns.csv', language)}")
    
    # Productivity rhythm analysis
    print(f"  {get_translation('Analyzing productivity rhythms...', language)}")
    productivity_data = analyze_productivity_rhythm(data)
    productivity_data.to_csv(analysis_dir / 'productivity_rhythm.csv', index=False)
    print(f"  {get_translation('Saved productivity rhythm analysis to advanced_analysis/productivity_rhythm.csv', language)}")
    
    # Employee work pattern clustering
    print(f"  {get_translation('Clustering employee work patterns...', language)}")
    clustering_results = cluster_employee_work_patterns(data)
    
    if 'error' not in clustering_results:
        cluster_df = clustering_results['employee_clusters']
        cluster_df.to_csv(analysis_dir / 'employee_clusters.csv', index=False)
        print(f"  {get_translation('Saved employee clustering results to advanced_analysis/employee_clusters.csv', language)}")
        
        # Save cluster profiles
        with open(analysis_dir / 'cluster_profiles.txt', 'w') as f:
            f.write(f"{get_translation('Employee Work Pattern Clusters', language)}\n")
            f.write("="*80 + "\n\n")
            
            for cluster_id, profile in clustering_results['cluster_profiles'].items():
                f.write(f"{get_translation('Cluster', language)} {cluster_id}:\n")
                f.write(f"  {get_translation('Employees', language)}: {', '.join(profile['employees'])}\n")
                f.write(f"  {get_translation('Size', language)}: {profile['size']} {get_translation('employees', language)}\n")
                f.write(f"  {get_translation('Characteristics', language)}:\n")
                
                for feature, value in profile['avg_features'].items():
                    f.write(f"    - {feature}: {value:.2f}\n")
                f.write("\n")
        
        print(f"  {get_translation('Saved cluster profiles to advanced_analysis/cluster_profiles.txt', language)}")
    else:
        print(f"  {get_translation('Error in clustering', language)}: {clustering_results['error']}")
    
    # Workflow pattern analysis
    print(f"  {get_translation('Identifying workflow patterns...', language)}")
    workflow_patterns = identify_workflow_patterns(data)
    
    if 'error' not in workflow_patterns:
        # Save detailed patterns for each employee
        for emp_id, patterns in workflow_patterns.items():
            if 'patterns' in patterns:
                patterns_df = patterns['patterns']
                if not patterns_df.empty:
                    patterns_df.to_csv(analysis_dir / f'workflow_patterns_{emp_id}.csv', index=False)
                    print(f"  {get_translation('Saved workflow patterns for {0} to advanced_analysis/workflow_patterns_{0}.csv', language).format(emp_id)}")
        
        # Generate optimization recommendations
        print(f"  {get_translation('Generating optimization recommendations...', language)}")
        optimization_opportunities = identify_optimization_opportunities(
            walking_patterns, productivity_data, workflow_patterns)
        
        # Save optimization report
        with open(analysis_dir / 'optimization_recommendations.txt', 'w') as f:
            f.write(f"{get_translation('Bakery Workflow Optimization Recommendations', language)}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{get_translation('Movement Optimization', language)}:\n")
            for rec in optimization_opportunities['movement_optimization']:
                f.write(f"  - [{rec['priority']}] {rec['employee']}: {rec['recommendation']}\n")
            f.write("\n")
            
            f.write(f"{get_translation('Workflow Optimization', language)}:\n")
            for rec in optimization_opportunities['workflow_optimization']:
                f.write(f"  - [{rec['priority']}] {rec['employee']}: {rec['recommendation']}\n")
            f.write("\n")
            
            f.write(f"{get_translation('Scheduling Recommendations', language)}:\n")
            for rec in optimization_opportunities['scheduling_recommendations']:
                f.write(f"  - [{rec['priority']}] {rec['target']}: {rec['recommendation']}\n")
        
        print(f"  {get_translation('Saved optimization recommendations to advanced_analysis/optimization_recommendations.txt', language)}")
    else:
        print(f"  {get_translation('Error in workflow pattern analysis', language)}: {workflow_patterns['error']}")
    
    print(f"\n=== {get_translation('Advanced Analysis Complete', language)} ===")

def main():
    """Main function to orchestrate the analysis"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine language based on --german flag
    language = 'de' if args.german else 'en'
    
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
        print(get_translation("The following required files are missing:", language))
        for file in missing_files:
            print(f"  - {file}")
        return 1
    
    print("=" * 80)
    print(f"=== {get_translation('Bakery Employee Movement Analysis', language)} ===")
    print("=" * 80)
    
    # Create output directory structure
    output_path = ensure_dir_exists(args.output)
    data_dir = ensure_dir_exists(output_path / 'data')
    vis_dir = ensure_dir_exists(output_path / 'visualizations')
    stats_dir = ensure_dir_exists(output_path / 'statistics')
    floor_plan_dir = ensure_dir_exists(vis_dir / 'floor_plan')
    transitions_dir = ensure_dir_exists(output_path / 'region_transitions')
    
    # Load data
    print(f"\n{get_translation('Loading data from {0}...', language).format(args.data)}")
    data = load_sensor_data(args.data)
    
    # Add department classification
    data = classify_departments(data)
    
    # Load floor plan data
    floor_plan_data = load_floor_plan_data(args.layout, args.metadata, args.connections)
    
    # Display basic information
    print(f"\n{get_translation('Dataset Information', language)}:")
    print(f"  {get_translation('Records', language)}: {len(data):,}")
    print(f"  {get_translation('Employees', language)}: {data['id'].nunique()} - {sorted(data['id'].unique())}")
    print(f"  {get_translation('Shifts', language)}: {data['shift'].nunique()} - {sorted(data['shift'].unique())}")
    print(f"  {get_translation('Date range', language)}: {data['date'].min()} to {data['date'].max()}")
    print(f"  {get_translation('Regions', language)}: {data['region'].nunique()}")
    print(f"  {get_translation('Activities', language)}: {data['activity'].nunique()} - {sorted(data['activity'].unique())}")
    
    # Total duration
    total_seconds = data['duration'].sum()
    total_formatted = format_seconds_to_hms(total_seconds)
    total_hours = total_seconds / 3600
    print(f"  {get_translation('Total duration', language)}: {total_formatted} ({total_hours:.2f} {get_translation('hours', language)})")
    
    # If ergonomicsonly is specified, just run the ergonomic analysis and exit
    if args.ergonomicsonly:
        run_ergonomic_analysis(data, output_path, args.employee, language)
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 40)
        print(f"=== {get_translation('Analysis Complete', language)} ===")
        print("=" * 40)
        print(get_translation('Execution time: {0:.2f} seconds', language).format(execution_time))
        print(f"\n{get_translation('Results saved to: {0}', language).format(f'{output_path}/ergonomic_analysis/')}")
        return 0
    
    if args.handlinganalysis:
        run_handling_time_analysis(data, output_path, language)
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 40)
        print(f"=== {get_translation('Analysis Complete', language)} ===")
        print("=" * 40)
        print(get_translation('Execution time: {0:.2f} seconds', language).format(execution_time))
        print(f"\n{get_translation('Results saved to: {0}', language).format(f'{output_path}/handling_analysis/')}")
        return 0
    
    # 1. Shift Analysis
    if run_all or args.step1:
        print("\n" + "=" * 40)
        print(f"=== 1. {get_translation('Shift Analysis', language)} ===")
        
        shift_summary = aggregate_by_shift(data)
        shift_summary.to_csv(data_dir / 'shift_summary.csv', index=False)
        print(f"  {get_translation('Saved shift summary to data/shift_summary.csv', language)}")
    
    # 2. Activity Analysis
    if run_all or args.step2:
        print("\n" + "=" * 40)
        print(f"=== 2. {get_translation('Activity Analysis', language)} ===")
        
        activity_summary = aggregate_by_activity(data)
        activity_summary.to_csv(data_dir / 'activity_summary.csv', index=False)
        print(f"  {get_translation('Saved activity summary to data/activity_summary.csv', language)}")
        
        plot_activity_distribution(activity_summary, save_path=vis_dir / 'activity_distribution.png', language=language)
        
        # Activity profile by employee
        plot_activity_distribution_by_employee(data, save_path=vis_dir / 'activity_distribution_by_employee.png', language=language)
        print(f"  {get_translation('Saved activity distribution by employee to visualizations/activity_distribution_by_employee.png', language)}")
    
    # 3. Region Analysis
    if run_all or args.step3:
        print("\n" + "=" * 40)
        print(f"=== 3. {get_translation('Region Analysis', language)} ===")
        
        # Aggregate by region
        region_summary = data.groupby('region')['duration'].sum().reset_index()
        region_summary['percentage'] = (region_summary['duration'] / region_summary['duration'].sum() * 100).round(1)
        region_summary['formatted_time'] = region_summary['duration'].apply(format_seconds_to_hms)
        region_summary = region_summary.sort_values('duration', ascending=False)
        
        region_summary.to_csv(data_dir / 'region_summary.csv', index=False)
        print(f"  {get_translation('Saved region summary to data/region_summary.csv', language)}")
        
        # Create floor plan visualizations
        if floor_plan_data:
            create_region_heatmap(floor_plan_data, region_summary, save_path=floor_plan_dir / 'region_heatmap.png', language=language)
    
    # 4. Employee Region Heatmaps
    if run_all or args.step4:
        print("\n" + "=" * 40)
        print(f"=== 4. {get_translation('Employee Region Heatmaps', language)} ===")
        
        run_employee_heatmaps(data, floor_plan_data, vis_dir, args.employee, language)
    
    # 5. Statistical Analysis
    if run_all or args.step5:
        print("\n" + "=" * 40)
        print(f"=== 5. {get_translation('Statistical Analysis', language)} ===")
        
        employee_analysis = analyze_employee_differences(data)
        
        # Save walking patterns
        if 'walking_patterns' in employee_analysis:
            employee_analysis['walking_patterns'].to_csv(stats_dir / 'walking_patterns_by_employee.csv', index=False)
            print(f"  {get_translation('Saved walking patterns analysis to statistics/walking_patterns_by_employee.csv', language)}")
        
        # Save handling patterns
        if 'handling_patterns' in employee_analysis:
            employee_analysis['handling_patterns'].to_csv(stats_dir / 'handling_patterns_by_employee.csv', index=False)
            print(f"  {get_translation('Saved handling patterns analysis to statistics/handling_patterns_by_employee.csv', language)}")
        
        # Ergonomic Analysis
        ergonomic_analysis = analyze_ergonomic_patterns(data)
        
        if 'position_summary' in ergonomic_analysis:
            ergonomic_analysis['position_summary'].to_csv(stats_dir / 'handling_position_summary.csv', index=False)
            print(f"  {get_translation('Saved handling position summary to statistics/handling_position_summary.csv', language)}")
        
        if 'employee_position_summary' in ergonomic_analysis:
            ergonomic_analysis['employee_position_summary'].to_csv(stats_dir / 'employee_handling_positions.csv', index=False)
            print(f"  {get_translation('Saved employee handling positions to statistics/employee_handling_positions.csv', language)}")
        
        # Shift Comparison
        shift_comparison = compare_shifts(data)
        
        if 'shift_metrics' in shift_comparison:
            shift_comparison['shift_metrics'].to_csv(stats_dir / 'shift_metrics.csv', index=False)
            print(f"  {get_translation('Saved shift metrics to statistics/shift_metrics.csv', language)}")
    
    # 6. Regional Activity Analysis
    if run_all or args.step6:
        print("\n" + "=" * 40)
        print(f"=== 6. {get_translation('Regional Activity Analysis', language)} ===")
        
        regional_analysis_dir = ensure_dir_exists(vis_dir / 'regional_activity_analysis')
        
        # If department is specified, analyze that department
        if args.department:
            department = args.department
            print(f"  {get_translation('Analyzing activities by region for {0} department...', language).format(get_translation(department, language))}")
            fig = analyze_activities_by_region(
                data, 
                department=department, 
                top_n_regions=5,
                save_path=regional_analysis_dir / f"{department.lower()}_regional_activity_analysis.png",
                language=language
            )
            print(f"  {get_translation('Saved {0} department regional activity analysis to visualizations/regional_activity_analysis', language).format(get_translation(department, language))}")
        # Otherwise, analyze both departments
        else:
            for department in ['Bread', 'Cake']:
                print(f"  {get_translation('Analyzing activities by region for {0} department...', language).format(get_translation(department, language))}")
                fig = analyze_activities_by_region(
                    data, 
                    department=department, 
                    top_n_regions=5,
                    save_path=regional_analysis_dir / f"{department.lower()}_regional_activity_analysis.png",
                    language=language
                )
            print(f"  {get_translation('Saved regional activity analyses to visualizations/regional_activity_analysis', language)}")
    
    # 7. Ergonomic Analysis
    if run_all or args.step7:
        print("\n" + "=" * 40)
        print(f"=== 7. {get_translation('Ergonomic Analysis', language)} ===")
        
        run_ergonomic_analysis(data, output_path, args.employee, language)
    
    # 8. Workflow Analysis
    if run_all or args.step8:
        print("\n" + "=" * 40)
        print(f"=== 8. {get_translation('Workflow Analysis', language)} ===")
        
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
        print(f"  {get_translation('Saved walking pattern analysis to statistics/workflow/walking_patterns.csv', language)}")
    
    # 9. Handling Time Analysis
    if run_all or args.step9:
        print("\n" + "=" * 40)
        print(f"=== 9. {get_translation('Handling Time Analysis', language)} ===")
        
        run_handling_time_analysis(data, output_path, language)
    
    # Run full advanced analysis if analysisonly flag is set
    if args.analysisonly:
        run_advanced_analysis(data, output_path, language)
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "=" * 40)
    print(f"=== {get_translation('Analysis Complete', language)} ===")
    print("=" * 40)
    print(get_translation('Execution time: {0:.2f} seconds', language).format(execution_time))
    print(f"\n{get_translation('Results saved to: {0}', language).format(output_path)}")
    
    # Only show outputs for the steps that were run
    print(f"\n{get_translation('Key outputs to check:', language)}")
    if run_all or args.step1:
        print(f"1. {data_dir}/shift_summary.csv - {get_translation('Time spent during each shift', language)}")
    if run_all or args.step2:
        print(f"2. {vis_dir}/activity_distribution_by_employee.png - {get_translation('Activity profiles by employee', language)}")
    if run_all or args.step3:
        print(f"3. {floor_plan_dir}/region_heatmap.png - {get_translation('Region usage heatmap', language)}")
    if run_all or args.step4:
        print(f"4. {vis_dir}/employee_heatmaps/ - {get_translation('Employee region heatmaps', language)}")
    if run_all or args.step5:
        print(f"5. {stats_dir}/shift_metrics.csv - {get_translation('Shift comparison metrics', language)}")
    if run_all or args.step6:
        print(f"6. {vis_dir}/regional_activity_analysis/ - {get_translation('Activity breakdown by region per employee', language)}")
    if run_all or args.step7:
        print(f"7. {output_path}/ergonomic_analysis/reports/ - {get_translation('Ergonomic assessment reports', language)}")
    if run_all or args.step8:
        print(f"8. {stats_dir}/workflow/walking_patterns.csv - {get_translation('Workflow analysis results', language)}")
    if run_all or args.step9:
        print(f"9. {output_path}/handling_analysis/ - {get_translation('Handling time position analysis', language)}")
        print(f"   {output_path}/visualizations/handling_analysis/ - {get_translation('Handling position visualizations', language)}")
    if args.ergonomicsonly:
        print(f"• {output_path}/ergonomic_analysis/reports/ - {get_translation('Ergonomic assessment reports', language)}")
        print(f"• {output_path}/ergonomic_analysis/ergonomic_scores_summary.csv - {get_translation('Scores summary', language)}")
        print(f"• {output_path}/ergonomic_analysis/activity_duration_statistics.csv - {get_translation('Activity duration data', language)}")
        print(f"• {output_path}/ergonomic_analysis/duration_factor_thresholds.txt - {get_translation('Threshold details', language)}")
    if args.handlinganalysis:
        print(f"• {output_path}/handling_analysis/handling_time_by_region.csv - {get_translation('Handling time by region', language)}")
        print(f"• {output_path}/handling_analysis/workstation_category_analysis.csv - {get_translation('Workstation category analysis', language)}")
        print(f"• {output_path}/handling_analysis/bread_vs_cake_comparison.csv - {get_translation('Bread vs cake workstation comparison', language)}")
        print(f"• {output_path}/visualizations/handling_analysis/ - {get_translation('Handling position visualizations', language)}")

if __name__ == "__main__":
    sys.exit(main())