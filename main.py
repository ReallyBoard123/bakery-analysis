#!/usr/bin/env python3
"""
Bakery Employee Analysis System

This script orchestrates the analysis of bakery employee tracking data.
"""

import pandas as pd
import time
from pathlib import Path
import argparse
import sys

# Import from modular structure
from src.analysis.ergonomics import (
    # Core analysis functions
    calculate_ergonomic_score,
    analyze_activity_durations,
    analyze_region_ergonomics,
    
    # Report generation
    generate_ergonomic_report,
    generate_region_ergonomic_report,
    generate_all_employees_comparison,
    
    # Main entry point for handling analysis
    run_handling_time_analysis
)

from src.utils.file_utils import ensure_dir_exists, check_required_files
from src.utils.data_utils import (
    load_sensor_data, 
    load_floor_plan_data, 
    classify_departments, 
    aggregate_by_shift, 
    aggregate_by_activity
)
from src.utils.time_utils import format_seconds_to_hms
from src.utils.translation_utils import get_translation

from src.visualization.activity_vis import (
    plot_activity_distribution, 
    plot_activity_distribution_by_employee,
    analyze_activities_by_region
)
from src.visualization.floor_plan_vis import create_region_heatmap
from src.visualization.employee_region_vis import create_employee_region_heatmap
from visualization.common_vis import plot_activity_breakdown_pie, plot_activity_by_day, plot_department_comparison, plot_employee_activities_by_date, plot_employee_summary, plot_region_activity_heatmap, plot_shift_comparison

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
    parser.add_argument('--step8', action='store_true', help='Run common visualizations')
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
            top_n=10,
            min_transitions=2,
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
    
    # Print scores overview
    print(f"\n{get_translation('Ergonomic Scores Overview:', language)}")
    for _, row in summary_df.iterrows():
        score_color = "good" if row['ergonomic_score'] >= 75 else ("fair" if row['ergonomic_score'] >= 60 else "poor")
        print(f"  {get_translation('Employee {0}: {1:.1f}/100 ({2})', language).format(row['employee_id'], row['ergonomic_score'], score_color)}")
    
    print(f"\n=== {get_translation('Ergonomic Analysis Complete', language)} ===")

def main():
    """Main function to orchestrate the analysis"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine language based on --german flag
    language = 'de' if args.german else 'en'
    
    # Check if any step is specified, otherwise run all steps
    run_all = not (args.step1 or args.step2 or args.step3 or args.step4 or 
                  args.step5 or args.step6 or args.step7 or args.step8 or args.step9 or
                  args.analysisonly or args.ergonomicsonly or args.handlinganalysis)
    
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
    floor_plan_dir = ensure_dir_exists(vis_dir / 'floor_plan')
    
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
    
    # If handlinganalysis flag is set, run just the handling time analysis
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
        
        run_employee_heatmaps(data, floor_plan_data, output_path, args.employee, language)
    
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
        
    # 8. Common Visualizations
    if run_all or args.step8:
        print("\n" + "=" * 40)
    print(f"=== 8. {get_translation('Common Visualizations', language)} ===")
    
    # Create directory for common visualizations
    common_vis_dir = ensure_dir_exists(vis_dir / 'common')
    
    # Activity by day
    print(f"  {get_translation('Generating activity by day visualization...', language)}")
    plot_activity_by_day(data, save_path=common_vis_dir / 'activity_by_day.png', language=language)
    
    # Employee summary by department
    print(f"  {get_translation('Generating employee summary visualization...', language)}")
    plot_employee_summary(data, save_path=common_vis_dir / 'employee_summary.png', language=language)
    
    # Department comparison
    print(f"  {get_translation('Generating department comparison visualization...', language)}")
    plot_department_comparison(data, save_path=common_vis_dir / 'department_comparison.png', language=language)
    
    # Shift comparison
    print(f"  {get_translation('Generating shift comparison visualization...', language)}")
    plot_shift_comparison(data, save_path=common_vis_dir / 'shift_comparison.png', language=language)
    
    # Activity breakdown pie chart
    print(f"  {get_translation('Generating overall activity breakdown...', language)}")
    plot_activity_breakdown_pie(data, save_path=common_vis_dir / 'activity_breakdown.png', language=language)
    
    # If employee is specified, create employee-specific visualizations
    if args.employee:
        employee_id = args.employee
        print(f"  {get_translation('Generating visualizations for employee {0}...', language).format(employee_id)}")
        
        # Employee activities by date
        plot_employee_activities_by_date(
            data, 
            employee_id=employee_id, 
            save_path=common_vis_dir / f"{employee_id}_activities_by_date.png",
            language=language
        )
        
        # Employee activity breakdown
        plot_activity_breakdown_pie(
            data, 
            filter_by='id', 
            filter_value=employee_id,
            save_path=common_vis_dir / f"{employee_id}_activity_breakdown.png",
            language=language
        )
    
    # If department is specified, create department-specific visualizations
    if args.department:
        department = args.department
        print(f"  {get_translation('Generating visualizations for {0} department...', language).format(get_translation(department, language))}")
        
        # Department activity breakdown
        plot_activity_breakdown_pie(
            data, 
            filter_by='department', 
            filter_value=department,
            save_path=common_vis_dir / f"{department.lower()}_activity_breakdown.png",
            language=language
        )
    
    # Region activity heatmap
    print(f"  {get_translation('Generating region activity heatmap...', language)}")
    plot_region_activity_heatmap(data, save_path=common_vis_dir / 'region_activity_heatmap.png', language=language)
    
    print(f"  {get_translation('Saved common visualizations to visualizations/common/', language)}")
    
    # 9. Handling Time Analysis
    if run_all or args.step9:
        print("\n" + "=" * 40)
        print(f"=== 9. {get_translation('Handling Time Analysis', language)} ===")
        
        run_handling_time_analysis(data, output_path, language)
    
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
        print(f"4. {output_path}/employee_heatmaps/ - {get_translation('Employee region heatmaps', language)}")
    if run_all or args.step6:
        print(f"6. {vis_dir}/regional_activity_analysis/ - {get_translation('Activity breakdown by region per employee', language)}")
    if run_all or args.step7:
        print(f"7. {output_path}/ergonomic_analysis/ - {get_translation('Ergonomic assessment reports', language)}")
    # Add to output summary
    if run_all or args.step8:
        print(f"8. {vis_dir}/common/ - {get_translation('Common visualizations', language)}")
    if run_all or args.step9:
        print(f"9. {output_path}/handling_analysis/ - {get_translation('Handling time position analysis', language)}")
    
    if args.ergonomicsonly:
        print(f"• {output_path}/ergonomic_analysis/employee_reports/ - {get_translation('Ergonomic assessment reports', language)}")
        print(f"• {output_path}/ergonomic_analysis/ergonomic_scores_summary.csv - {get_translation('Scores summary', language)}")
    
    if args.handlinganalysis:
        print(f"• {output_path}/handling_analysis/handling_time_by_region.csv - {get_translation('Handling time by region', language)}")
        print(f"• {output_path}/handling_analysis/workstation_category_analysis.csv - {get_translation('Workstation category analysis', language)}")
        print(f"• {output_path}/handling_analysis/bread_vs_cake_comparison.csv - {get_translation('Bread vs cake workstation comparison', language)}")
        print(f"• {output_path}/visualizations/handling_analysis/ - {get_translation('Handling position visualizations', language)}")

if __name__ == "__main__":
    sys.exit(main())