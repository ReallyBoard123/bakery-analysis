"""
Department Analysis Module

Contains functions for analyzing and comparing employee handling behaviors within departments.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.time_utils import format_seconds_to_hms
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import (
    get_activity_colors, get_text
)

from .utilities import create_region_categories

def compare_employees_within_department(data, output_dir=None, language='en'):
    """
    Compare how employees differ from their peers in their specialized areas
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
        
    Returns:
    --------
    dict
        Dictionary containing department-specific analysis results
    """
    print(f"\n{get_text('Analyzing Individual Differences Within Departments', language)}")
    print("=" * 80)
    
    # Define department memberships
    bread_employees = ['32-A', '32-C', '32-D', '32-E']
    cake_employees = ['32-F', '32-G', '32-H']
    
    # Define specialized workstation regions
    region_categories = create_region_categories()
    bread_regions = region_categories['Bread Workstation']
    cake_regions = region_categories['Cake Workstation']
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    results = {}
    
    # Process bread department
    bread_results = analyze_department_employees(
        data, 'Bread', bread_employees, bread_regions, 
        handling_categories, output_dir, language
    )
    results['Bread'] = bread_results
    
    # Process cake department
    cake_results = analyze_department_employees(
        data, 'Cake', cake_employees, cake_regions, 
        handling_categories, output_dir, language
    )
    results['Cake'] = cake_results
    
    return results

def analyze_department_employees(data, department, employees, regions, 
                               activities, output_dir=None, language='en'):
    """
    Analyze a specific department's employees in their specialized regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    department : str
        Department name ('Bread' or 'Cake')
    employees : list
        List of employee IDs in this department
    regions : list
        List of specialized region names for this department
    activities : list
        List of handling activities to analyze
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Filter data for these employees in these regions doing these activities
    dept_data = data[
        data['id'].isin(employees) & 
        data['region'].isin(regions) & 
        data['activity'].isin(activities)
    ]
    
    if dept_data.empty:
        print(f"  {get_text('No data found for {0} department in specialized regions', language).format(department)}")
        return {}
    
    # Calculate department-wide averages
    dept_total = dept_data['duration'].sum()
    
    dept_averages = {}
    for activity in activities:
        act_data = dept_data[dept_data['activity'] == activity]
        act_duration = act_data['duration'].sum()
        act_percentage = (act_duration / dept_total * 100) if dept_total > 0 else 0
        dept_averages[activity] = act_percentage
    
    print(f"\n{get_text('{0} Department Average Handling Patterns:', language).format(department)}")
    for activity, percentage in dept_averages.items():
        print(f"  {get_text(activity, language)}: {percentage:.1f}%")
    
    # Analyze individual employees
    employee_results = {}
    
    for emp_id in employees:
        emp_data = dept_data[dept_data['id'] == emp_id]
        
        if emp_data.empty:
            employee_results[emp_id] = {
                'has_data': False,
                'message': f"No data for employee {emp_id} in {department} specialized regions"
            }
            continue
        
        emp_total = emp_data['duration'].sum()
        
        # Activity breakdown
        activity_breakdown = {}
        for activity in activities:
            act_data = emp_data[emp_data['activity'] == activity]
            act_duration = act_data['duration'].sum()
            act_percentage = (act_duration / emp_total * 100) if emp_total > 0 else 0
            
            # Calculate deviation from department average
            dept_avg = dept_averages.get(activity, 0)
            deviation = act_percentage - dept_avg
            
            activity_breakdown[activity] = {
                'duration': act_duration,
                'percentage': act_percentage,
                'dept_average': dept_avg,
                'deviation': deviation
            }
        
        # Determine areas where this employee differs significantly
        significant_differences = []
        for activity, data in activity_breakdown.items():
            if abs(data['deviation']) >= 5:  # 5% or more is considered significant
                direction = 'higher' if data['deviation'] > 0 else 'lower'
                significant_differences.append({
                    'activity': activity,
                    'deviation': data['deviation'],
                    'direction': direction
                })
        
        employee_results[emp_id] = {
            'has_data': True,
            'total_duration': emp_total,
            'activity_breakdown': activity_breakdown,
            'significant_differences': significant_differences
        }
    
    # Create visualization
    if output_dir:
        create_department_comparison_visualization(
            department, employee_results, dept_averages, 
            activities, output_dir, language
        )
    
    # Print summary of differences
    print(f"\n{get_text('Individual Differences in {0} Department:', language).format(department)}")
    for emp_id, results in employee_results.items():
        if results['has_data']:
            if results['significant_differences']:
                print(f"  {emp_id}:")
                for diff in results['significant_differences']:
                    print(f"    - {diff['activity']}: {diff['deviation']:.1f}% {diff['direction']} than department average")
            else:
                print(f"  {emp_id}: {get_text('Works similarly to department average', language)}")
    
    # Save results to CSV if output_dir is provided
    if output_dir:
        output_path = ensure_dir_exists(output_dir / 'handling_analysis')
        
        # Prepare data for CSV
        emp_comparison_data = []
        for emp_id, results in employee_results.items():
            if results['has_data']:
                row = {'Employee': emp_id}
                for activity in activities:
                    if activity in results['activity_breakdown']:
                        data = results['activity_breakdown'][activity]
                        row[f"{activity} %"] = data['percentage']
                        row[f"{activity} deviation"] = data['deviation']
                emp_comparison_data.append(row)
        
        if emp_comparison_data:
            emp_comparison_df = pd.DataFrame(emp_comparison_data)
            emp_comparison_df.to_csv(output_path / f"{department.lower()}_employee_comparison.csv", index=False)
            print(f"  {get_text('Saved {0} employee comparison to handling_analysis/{0}_employee_comparison.csv', language).format(department.lower())}")
    
    return {
        'department_averages': dept_averages,
        'employee_results': employee_results
    }

def create_department_comparison_visualization(department, employee_results, 
                                             dept_averages, activities, 
                                             output_dir, language='en'):
    """
    Create visualization comparing employees within a department
    
    Parameters:
    -----------
    department : str
        Department name ('Bread' or 'Cake')
    employee_results : dict
        Dictionary of employee analysis results
    dept_averages : dict
        Dictionary of department average percentages by activity
    activities : list
        List of handling activities
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"{get_text('Matplotlib or Seaborn not available, skipping visualizations', language)}")
        return
    
    # Ensure visualization directory exists
    vis_dir = ensure_dir_exists(output_dir / 'visualizations' / 'handling_analysis')
    
    # Get standard activity colors
    activity_colors = get_activity_colors()
    
    # Filter to employees with data
    valid_employees = {emp_id: results for emp_id, results in employee_results.items() 
                     if results['has_data']}
    
    if not valid_employees:
        return
    
    # Create figure for radar chart - one per activity
    fig, axes = plt.subplots(1, len(activities), figsize=(15, 5))
    
    # Ensure axes is always a list
    if len(activities) == 1:
        axes = [axes]
    
    for i, activity in enumerate(activities):
        ax = axes[i]
        
        # Gather data for this activity
        emp_values = []
        emp_ids = []
        
        for emp_id, results in valid_employees.items():
            breakdown = results['activity_breakdown']
            if activity in breakdown:
                emp_values.append(breakdown[activity]['percentage'])
                emp_ids.append(emp_id)
        
        # Create bar chart
        color = activity_colors.get(activity, '#CCCCCC')
        bars = ax.bar(emp_ids, emp_values, color=color, alpha=0.6)
        
        # Add department average line
        dept_avg = dept_averages.get(activity, 0)
        ax.axhline(y=dept_avg, linestyle='--', color='red', label=get_text('Department Average', language))
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                      xy=(bar.get_x() + bar.get_width()/2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        # Add title and labels
        ax.set_title(f"{get_text(activity, language)}")
        ax.set_ylabel(get_text('Percentage of Time (%)', language))
        ax.set_ylim(bottom=0)
        
        # Only show legend on first plot
        if i == 0:
            ax.legend()
    
    # Add overall title
    plt.suptitle(f"{get_text('Individual Differences in {0} Department', language).format(department)}", 
               fontsize=16, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(vis_dir / f"{department.lower()}_employee_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  {get_text('Saved {0} department comparison visualization to visualizations/handling_analysis/{0}_employee_comparison.png', language).format(department.lower())}")

def compare_departments_in_specializations(data, output_dir=None, language='en'):
    """
    Compare how bread and cake workers' patterns differ in their specialized environments
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
        
    Returns:
    --------
    dict
        Dictionary containing comparison results
    """
    print(f"\n{get_text('Comparing Departments in Specialized Environments', language)}")
    print("=" * 80)
    
    # Define department memberships
    bread_employees = ['32-A', '32-C', '32-D', '32-E']
    cake_employees = ['32-F', '32-G', '32-H']
    
    # Define specialized workstation regions
    region_categories = create_region_categories()
    bread_regions = region_categories['Bread Workstation']
    cake_regions = region_categories['Cake Workstation']
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # Filter data for bread employees in bread regions
    bread_specialized_data = data[
        data['id'].isin(bread_employees) & 
        data['region'].isin(bread_regions) & 
        data['activity'].isin(handling_categories)
    ]
    
    # Filter data for cake employees in cake regions
    cake_specialized_data = data[
        data['id'].isin(cake_employees) & 
        data['region'].isin(cake_regions) & 
        data['activity'].isin(handling_categories)
    ]
    
    # Check if we have enough data to proceed
    if bread_specialized_data.empty or cake_specialized_data.empty:
        print(f"{get_text('Insufficient data for comparison', language)}")
        return {}
    
    # Calculate total times and percentages for each department
    bread_total = bread_specialized_data['duration'].sum()
    cake_total = cake_specialized_data['duration'].sum()
    
    bread_activities = {}
    cake_activities = {}
    
    for activity in handling_categories:
        # Bread department
        bread_act_data = bread_specialized_data[bread_specialized_data['activity'] == activity]
        bread_act_time = bread_act_data['duration'].sum()
        bread_act_pct = (bread_act_time / bread_total * 100) if bread_total > 0 else 0
        
        bread_activities[activity] = {
            'duration': bread_act_time,
            'percentage': bread_act_pct,
            'formatted_time': format_seconds_to_hms(bread_act_time)
        }
        
        # Cake department
        cake_act_data = cake_specialized_data[cake_specialized_data['activity'] == activity]
        cake_act_time = cake_act_data['duration'].sum()
        cake_act_pct = (cake_act_time / cake_total * 100) if cake_total > 0 else 0
        
        cake_activities[activity] = {
            'duration': cake_act_time,
            'percentage': cake_act_pct,
            'formatted_time': format_seconds_to_hms(cake_act_time)
        }
    
    # Create comparison results
    comparison_results = {
        'bread': {
            'total_duration': bread_total,
            'formatted_total': format_seconds_to_hms(bread_total),
            'activities': bread_activities
        },
        'cake': {
            'total_duration': cake_total,
            'formatted_total': format_seconds_to_hms(cake_total),
            'activities': cake_activities
        }
    }
    
    # Create visualization
    if output_dir:
        create_department_specialization_visualization(
            comparison_results, handling_categories, output_dir, language
        )
    
    # Print summary
    print(f"\n{get_text('Department Handling Patterns in Specialized Environments:', language)}")
    print(f"\n{get_text('Bread Department:', language)}")
    print(f"  {get_text('Total Time:', language)} {format_seconds_to_hms(bread_total)}")
    for activity, data in bread_activities.items():
        print(f"  {get_text(activity, language)}: {data['percentage']:.1f}% ({data['formatted_time']})")
    
    print(f"\n{get_text('Cake Department:', language)}")
    print(f"  {get_text('Total Time:', language)} {format_seconds_to_hms(cake_total)}")
    for activity, data in cake_activities.items():
        print(f"  {get_text(activity, language)}: {data['percentage']:.1f}% ({data['formatted_time']})")
    
    # Save results to CSV if output_dir is provided
    if output_dir:
        output_path = ensure_dir_exists(output_dir / 'handling_analysis')
        
        # Prepare data for CSV
        comparison_data = []
        for activity in handling_categories:
            row = {
                'Activity': activity,
                'Bread %': bread_activities[activity]['percentage'],
                'Bread Time': bread_activities[activity]['formatted_time'],
                'Cake %': cake_activities[activity]['percentage'],
                'Cake Time': cake_activities[activity]['formatted_time'],
                'Difference (Cake-Bread)': cake_activities[activity]['percentage'] - bread_activities[activity]['percentage']
            }
            comparison_data.append(row)
        
        # Add totals row
        comparison_data.append({
            'Activity': 'Total',
            'Bread %': 100,
            'Bread Time': format_seconds_to_hms(bread_total),
            'Cake %': 100,
            'Cake Time': format_seconds_to_hms(cake_total),
            'Difference (Cake-Bread)': 0
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_path / 'department_specialization_comparison.csv', index=False)
        print(f"  {get_text('Saved department specialization comparison to handling_analysis/department_specialization_comparison.csv', language)}")
    
    return comparison_results

def create_department_specialization_visualization(comparison_results, activities, 
                                                output_dir, language='en'):
    """
    Create visualization comparing departments in their specialized environments
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary with department comparison data
    activities : list
        List of handling activities
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"{get_text('Matplotlib or Seaborn not available, skipping visualizations', language)}")
        return
    
    # Ensure visualization directory exists
    vis_dir = ensure_dir_exists(output_dir / 'visualizations' / 'handling_analysis')
    
    # Get standard activity colors
    activity_colors = get_activity_colors()
    
    # Extract data for each department
    bread_data = comparison_results['bread']
    cake_data = comparison_results['cake']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Side-by-side percentage comparison
    bread_pcts = [bread_data['activities'][act]['percentage'] for act in activities]
    cake_pcts = [cake_data['activities'][act]['percentage'] for act in activities]
    
    x = range(len(activities))
    width = 0.35
    
    b1 = ax1.bar([i - width/2 for i in x], bread_pcts, width, label=get_text('Bread Department', language), color='#8c6464')
    b2 = ax1.bar([i + width/2 for i in x], cake_pcts, width, label=get_text('Cake Department', language), color='#ffd700')
    
    # Add percentage labels
    for bars, values in [(b1, bread_pcts), (b2, cake_pcts)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    ax1.set_ylabel(get_text('Percentage of Time (%)', language))
    ax1.set_title(get_text('Activity Percentages by Department', language))
    ax1.set_xticks(x)
    ax1.set_xticklabels([get_text(act, language) for act in activities])
    ax1.legend()
    
    # 2. Activity breakdown pie charts
    bread_data_pie = [bread_data['activities'][act]['percentage'] for act in activities]
    cake_data_pie = [cake_data['activities'][act]['percentage'] for act in activities]
    
    # Colors for pie chart
    colors = [activity_colors.get(act, '#CCCCCC') for act in activities]
    
    # Create a figure with two pie charts side by side
    wedges1, texts1, autotexts1 = ax2.pie(
        bread_data_pie, 
        labels=[get_text(act, language) for act in activities],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'alpha': 0.7},
        textprops={'fontsize': 10}
    )
    
    # Add center circle to make a donut chart
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax2.add_patch(centre_circle)
    
    # Add text in the center
    ax2.text(0, 0, get_text('Bread\n{0}', language).format(format_seconds_to_hms(bread_data['total_duration'])), 
           ha='center', va='center', fontsize=10)
    
    # Add title and adjust layout
    plt.suptitle(get_text('Department Handling Patterns in Specialized Environments', language), 
               fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(vis_dir / 'department_specialization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a second figure for the cake department pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    
    wedges2, texts2, autotexts2 = ax.pie(
        cake_data_pie, 
        labels=[get_text(act, language) for act in activities],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'alpha': 0.7},
        textprops={'fontsize': 10}
    )
    
    # Add center circle to make a donut chart
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax.add_patch(centre_circle)
    
    # Add text in the center
    ax.text(0, 0, get_text('Cake\n{0}', language).format(format_seconds_to_hms(cake_data['total_duration'])), 
          ha='center', va='center', fontsize=10)
    
    # Add title
    plt.title(get_text('Cake Department Handling Pattern', language), fontsize=14)
    
    plt.tight_layout()
    
    # Save second figure
    plt.savefig(vis_dir / 'cake_department_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  {get_text('Saved department specialization visualizations to visualizations/handling_analysis/', language)}")