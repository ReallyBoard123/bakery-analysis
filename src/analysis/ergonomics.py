"""
Ergonomics Analysis Module

Analyzes ergonomic patterns and generates employee and region ergonomic scores with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Import visualization utilities from base
from src.visualization.base import (
    save_figure, set_visualization_style, 
    get_activity_colors, get_employee_colors,
    get_activity_order, add_duration_percentage_label
)
from src.utils.time_utils import format_seconds_to_hms

def create_region_clusters():
    """Group similar regions into functional clusters"""
    return {
        'Bread_Stations': ['1_Brotstation', '2_Brotstation', '3_Brotstation', '4_Brotstation'],
        'Pastry_Stations': ['1_Konditorei_station', '2_Konditorei_station', '3_Konditorei_station', 
                           'konditorei_deco', 'konditorei_deco_1', 'konditorei_deco_2'],
        'Ovens': ['Etagenofen', '1_Stickenofen', '2_Stickenofen', 'Gärraum', 'Fettbacken', 
                 'g\u00e4rraum_handle', 'etagenofen'],
        'Storage': ['Lager', '1_lager', '2_lager', '3_lager', 'Silo', 'Lager_eingang', '2_lager_eingang'],
        'Hallways': ['flur_staircase_up', 'flur_eingang_umkleidung_WC', '1_corridor', '2_flur_aufzug', 
                    '1_flur_aufzug', '1_corridor_a', '1_corridor_b', 'flur_staircase']
    }

def analyze_activity_durations(data):
    """
    Analyze the distribution of activity durations to establish data-driven thresholds
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    
    Returns:
    --------
    dict
        Dictionary with activity duration statistics
    """
    # Analyze durations by activity type
    activity_stats = {}
    
    for activity in data['activity'].unique():
        act_data = data[data['activity'] == activity]['duration']
        
        # Calculate percentiles
        percentiles = np.percentile(act_data, [25, 50, 75, 90, 95])
        
        activity_stats[activity] = {
            'count': len(act_data),
            'mean': act_data.mean(),
            'median': act_data.median(),
            'p25': percentiles[0],
            'p50': percentiles[1],
            'p75': percentiles[2],
            'p90': percentiles[3],
            'p95': percentiles[4],
            'min': act_data.min(),
            'max': act_data.max()
        }
    
    return activity_stats

def calculate_ergonomic_score(data, min_duration=5):
    """
    Calculate ergonomic scores out of 100 for each employee with revised methodology
    that more heavily penalizes problematic postures
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    min_duration : int, optional
        Minimum duration in seconds to filter out potential false readings
    
    Returns:
    --------
    dict
        Dictionary containing ergonomic scores and detailed factor breakdown
    """
    # Filter out very short activities that might be false readings
    filtered_data = data[data['duration'] >= min_duration]
    
    # Starting score is 100, deductions will be made for ergonomic issues
    base_score = 100
    
    # Define direct percentage-based penalties by posture
    # These represent deduction points per percentage point of time spent in the posture
    posture_penalties = {
        'Handle up': 0.4,     # 40% penalty per % of time (high risk)
        'Handle down': 0.5,   # 50% penalty per % of time (highest risk)
        'Handle center': 0.05, # 5% penalty per % of time (low risk)
        'Stand': 0.15,        # 15% penalty per % of time (medium risk)
        'Walk': 0.05          # 5% penalty per % of time (low risk)
    }
    
    # Add region cluster column
    region_mapping = create_region_clusters()
    filtered_data['region_cluster'] = filtered_data['region'].apply(
        lambda r: next((cluster for cluster, regions in region_mapping.items() 
                      if r in regions), 'Other'))
    
    # Calculate employee-level ergonomic scores
    employee_scores = []
    
    for emp_id, emp_data in filtered_data.groupby('id'):
        total_duration = emp_data['duration'].sum()
        
        # Initialize score components
        score_components = {
            'base_score': base_score,
            'deductions': {},
            'final_score': 0,
            'activity_breakdown': {},
            'region_breakdown': {},
        }
        
        # Calculate direct percentage-based deductions by activity
        total_deduction = 0
        
        for activity in posture_penalties.keys():
            act_data = emp_data[emp_data['activity'] == activity]
            if act_data.empty:
                continue
                
            activity_duration = act_data['duration'].sum()
            activity_percentage = (activity_duration / total_duration) * 100
            
            # Calculate direct percentage-based deduction
            direct_deduction = activity_percentage * posture_penalties[activity]
            total_deduction += direct_deduction
            score_components['deductions'][activity] = direct_deduction
            
            # Store activity breakdown
            score_components['activity_breakdown'][activity] = {
                'percentage': activity_percentage,
                'duration': activity_duration,
                'deduction': direct_deduction
            }
        
        # Calculate region-specific deductions
        region_deductions = {}
        for region, region_data in emp_data.groupby('region'):
            region_duration = region_data['duration'].sum()
            region_percentage = (region_duration / total_duration) * 100
            
            # Calculate weighted deduction for this region
            region_deduction = 0
            for activity, act_data in region_data.groupby('activity'):
                if activity not in posture_penalties:
                    continue
                    
                activity_duration = act_data['duration'].sum()
                activity_percentage = (activity_duration / region_duration) * 100
                region_deduction += activity_percentage * posture_penalties[activity] / 100
            
            # Store region breakdown
            region_deductions[region] = region_deduction * region_percentage / 100
            score_components['region_breakdown'][region] = {
                'percentage': region_percentage,
                'duration': region_duration,
                'deduction': region_deductions[region]
            }
        
        # Calculate final score (capped at minimum of 0)
        final_score = max(base_score - total_deduction, 0)
        score_components['final_score'] = final_score
        
        employee_scores.append({
            'employee_id': emp_id,
            'final_score': final_score,
            'total_duration': total_duration,
            'components': score_components
        })
    
    return employee_scores

def export_duration_factor_thresholds(data, output_dir):
    """
    Export the duration factor thresholds used in ergonomic scoring
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path
        Directory to save the threshold information
    
    Returns:
    --------
    None
    """
    activity_stats = analyze_activity_durations(data)
    output_path = Path(output_dir)
    
    with open(output_path / 'duration_factor_thresholds.txt', 'w') as f:
        f.write("Ergonomic Scoring System\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Deduction points per percentage of time spent in each posture:\n")
        f.write("  Handle up: 0.4 points deducted per percentage point of time spent\n")
        f.write("  Handle down: 0.5 points deducted per percentage point of time spent\n")
        f.write("  Handle center: 0.05 points deducted per percentage point of time spent\n")
        f.write("  Stand: 0.15 points deducted per percentage point of time spent\n")
        f.write("  Walk: 0.05 points deducted per percentage point of time spent\n\n")
        
        f.write("Example: An employee spending 50% of time in Handle down position\n")
        f.write("  would receive a deduction of 50 × 0.5 = 25 points\n\n")
        
        f.write("Score categories:\n")
        f.write("  Good: 75-100\n")
        f.write("  Fair: 60-75\n")
        f.write("  Poor: 0-60\n\n")
        
        f.write("Duration statistics by activity:\n")
        for activity, stats in activity_stats.items():
            f.write(f"{activity}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean: {stats['mean']:.1f} seconds\n")
            f.write(f"  Median: {stats['median']:.1f} seconds\n")
            f.write(f"  25th percentile: {stats['p25']:.1f} seconds\n")
            f.write(f"  75th percentile: {stats['p75']:.1f} seconds\n")
            f.write(f"  90th percentile: {stats['p90']:.1f} seconds\n\n")

def generate_ergonomic_report(employee_scores, output_dir):
    """
    Generate detailed ergonomic report visualizations for each employee
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries from calculate_ergonomic_score
    output_dir : Path
        Directory to save the reports
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard activity colors and order
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    employee_colors = get_employee_colors()
    
    # Calculate benchmark values for comparison
    avg_score = np.mean([emp['final_score'] for emp in employee_scores])
    max_score = np.max([emp['final_score'] for emp in employee_scores])
    min_score = np.min([emp['final_score'] for emp in employee_scores])
    
    # Generate report for each employee
    for emp_data in employee_scores:
        emp_id = emp_data['employee_id']
        score = emp_data['final_score']
        components = emp_data['components']
        
        # Create figure with 2 panels
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 2])
        
        # 1. Main score display
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.axis('off')
        
        # Get employee-specific color
        emp_color = employee_colors.get(emp_id, '#333333')
        
        # Display the score prominently
        score_color = 'green' if score >= 75 else ('orange' if score >= 60 else 'red')
        ax_main.text(0.5, 0.7, f"Ergonomic Score: {score:.1f}/100", 
                    fontsize=28, ha='center', color=score_color, fontweight='bold')
        
        # Add comparison to other employees
        comparison_text = f"Average: {avg_score:.1f} | Min: {min_score:.1f} | Max: {max_score:.1f}"
        ax_main.text(0.5, 0.5, comparison_text, fontsize=14, ha='center')
        
        # Add employee ID and total duration
        hours = emp_data['total_duration'] / 3600
        ax_main.text(0.5, 0.3, f"Employee: {emp_id} | Hours Analyzed: {hours:.1f}", 
                    fontsize=16, ha='center', fontweight='bold', color=emp_color)
        
        # 2. Deductions breakdown
        ax_deduct = fig.add_subplot(gs[1, 0])
        
        # Extract deductions
        deductions = components['deductions']
        activities = list(deductions.keys())
        values = list(deductions.values())
        
        # Sort by deduction value
        sorted_idx = np.argsort(values)
        activities = [activities[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Use activity-specific colors from base visualization
        colors = [activity_colors.get(act, '#CCCCCC') for act in activities]
        
        # Create horizontal bar chart
        bars = ax_deduct.barh(activities, values, color=colors)
        ax_deduct.set_xlabel('Deduction Points')
        ax_deduct.set_title('Ergonomic Deductions by Activity', fontsize=14)
        
        # Add labels
        for bar in bars:
            width = bar.get_width()
            ax_deduct.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                          f"{width:.1f}", va='center')
        
        # 3. Activity time distribution (bar chart)
        ax_activity = fig.add_subplot(gs[1, 1])
        
        # Extract activity durations
        act_breakdown = components['activity_breakdown']
        
        # Order activities according to standard order
        ordered_activities = []
        ordered_durations = []
        ordered_colors = []
        
        # First add activities in standard order if they exist
        for activity in activity_order:
            if activity in act_breakdown:
                ordered_activities.append(activity)
                ordered_durations.append(act_breakdown[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Then add any remaining activities
        for activity in act_breakdown.keys():
            if activity not in ordered_activities and activity in activity_colors:
                ordered_activities.append(activity)
                ordered_durations.append(act_breakdown[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Create bar chart
        bars = ax_activity.bar(ordered_activities, ordered_durations, color=ordered_colors)
        
        # Add percentage and duration labels
        total_seconds = sum(act_breakdown[act]['duration'] for act in act_breakdown)
        for i, activity in enumerate(ordered_activities):
            act_duration = act_breakdown[activity]['duration']
            act_percentage = act_breakdown[activity]['percentage']
            
            # Format duration
            formatted_duration = format_seconds_to_hms(act_duration)
            
            # Position label at top of bar
            bar = bars[i]
            height = bar.get_height()
            ax_activity.text(bar.get_x() + bar.get_width()/2., height,
                           f"{act_percentage:.1f}%\n{formatted_duration}", 
                           ha='center', va='bottom', fontsize=10)
        
        ax_activity.set_ylabel('Duration (hours)')
        ax_activity.set_title('Activity Time Distribution', fontsize=14)
        plt.setp(ax_activity.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_activity.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add title and save
        plt.suptitle(f"Ergonomic Assessment Report - Employee {emp_id}", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Use the save_figure utility from base
        save_figure(fig, output_path / f"{emp_id}_ergonomic_report.png")

def analyze_region_ergonomics(data, min_percentage=5):
    """
    Analyze ergonomic patterns for each region
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    min_percentage : float, optional
        Minimum percentage of time an employee must spend at a region to be included
    
    Returns:
    --------
    dict
        Dictionary containing region-level ergonomic analyses
    """
    results = {}
    
    # Get all unique regions
    all_regions = data['region'].unique()
    
    # Process each region
    for region in all_regions:
        region_data = data[data['region'] == region]
        
        if region_data.empty:
            continue
            
        # Total duration in this region across all employees
        total_region_duration = region_data['duration'].sum()
        
        # Identify employees who use this region significantly
        qualified_employees = []
        
        for emp_id, emp_data in data.groupby('id'):
            # Total time for this employee
            emp_total_time = emp_data['duration'].sum()
            
            # Time in this region
            region_time = emp_data[emp_data['region'] == region]['duration'].sum()
            region_percentage = (region_time / emp_total_time * 100) if emp_total_time > 0 else 0
            
            # Check if this region is in the employee's top 5
            emp_top_regions = emp_data.groupby('region')['duration'].sum().sort_values(ascending=False).head(5).index.tolist()
            
            if region_percentage >= min_percentage and region in emp_top_regions:
                qualified_employees.append({
                    'id': emp_id,
                    'time': region_time,
                    'percentage': region_percentage,
                    'in_top5': region in emp_top_regions
                })
        
        # If no qualified employees, skip this region
        if not qualified_employees:
            continue
        
        # Calculate activity distribution in this region (considering only qualified employees)
        qualified_emp_ids = [emp['id'] for emp in qualified_employees]
        qualified_region_data = region_data[region_data['id'].isin(qualified_emp_ids)]
        
        activity_distribution = {}
        for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
            act_data = qualified_region_data[qualified_region_data['activity'] == activity]
            act_duration = act_data['duration'].sum()
            act_percentage = (act_duration / qualified_region_data['duration'].sum() * 100) if not qualified_region_data.empty else 0
            
            activity_distribution[activity] = {
                'duration': act_duration,
                'percentage': act_percentage
            }
        
        # Calculate ergonomic score for this region
        posture_penalties = {
            'Handle up': 0.4,
            'Handle down': 0.5, 
            'Handle center': 0.05,
            'Stand': 0.15,
            'Walk': 0.05
        }
        
        deduction = sum(activity_distribution[act]['percentage'] * penalty 
                        for act, penalty in posture_penalties.items() if act in activity_distribution)
        ergonomic_score = max(100 - deduction, 0)
        
        # Calculate employee contribution to this region
        employee_contributions = []
        for emp in qualified_employees:
            contribution_percentage = (emp['time'] / qualified_region_data['duration'].sum() * 100) if not qualified_region_data.empty else 0
            
            # Get activity breakdown for this employee in this region
            emp_region_data = qualified_region_data[qualified_region_data['id'] == emp['id']]
            emp_activity_breakdown = {}
            
            for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
                act_data = emp_region_data[emp_region_data['activity'] == activity]
                act_duration = act_data['duration'].sum()
                act_percentage = (act_duration / emp_region_data['duration'].sum() * 100) if not emp_region_data.empty else 0
                
                emp_activity_breakdown[activity] = {
                    'duration': act_duration,
                    'percentage': act_percentage
                }
            
            employee_contributions.append({
                'id': emp['id'],
                'time': emp['time'],
                'contribution_percentage': contribution_percentage,
                'region_percentage': emp['percentage'],
                'activity_breakdown': emp_activity_breakdown
            })
        
        # Store results
        results[region] = {
            'qualified_employees': len(qualified_employees),
            'total_duration': qualified_region_data['duration'].sum(),
            'activity_distribution': activity_distribution,
            'ergonomic_score': ergonomic_score,
            'employee_contributions': employee_contributions
        }
    
    return results

def generate_region_ergonomic_report(region_analyses, output_dir):
    """
    Generate ergonomic reports for each analyzed region
    
    Parameters:
    -----------
    region_analyses : dict
        Dictionary of region analyses from analyze_region_ergonomics
    output_dir : Path
        Directory to save the reports
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard activity colors and order
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    employee_colors = get_employee_colors()
    
    # Generate report for each region
    for region, analysis in region_analyses.items():
        # Skip regions with no ergonomic score (should not happen but just in case)
        if 'ergonomic_score' not in analysis:
            continue
            
        score = analysis['ergonomic_score']
        total_duration = analysis['total_duration']
        activity_distribution = analysis['activity_distribution']
        employee_contributions = analysis['employee_contributions']
        
        # Create figure with 3 panels
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1.5])
        
        # 1. Main score display
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.axis('off')
        
        # Display the score prominently
        score_color = 'green' if score >= 75 else ('orange' if score >= 60 else 'red')
        ax_main.text(0.5, 0.7, f"Region Ergonomic Score: {score:.1f}/100", 
                    fontsize=28, ha='center', color=score_color, fontweight='bold')
        
        # Add region info
        hours = total_duration / 3600
        qualified_employees = analysis['qualified_employees']
        ax_main.text(0.5, 0.4, f"Region: {region} | Hours Analyzed: {hours:.1f} | Employees: {qualified_employees}", 
                    fontsize=16, ha='center', fontweight='bold')
        
        # 2. Activity distribution in this region
        ax_activity = fig.add_subplot(gs[1, 0])
        
        # Order activities according to standard order
        ordered_activities = []
        ordered_durations = []
        ordered_colors = []
        
        for activity in activity_order:
            if activity in activity_distribution:
                ordered_activities.append(activity)
                ordered_durations.append(activity_distribution[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Add any remaining activities
        for activity in activity_distribution.keys():
            if activity not in ordered_activities and activity in activity_colors:
                ordered_activities.append(activity)
                ordered_durations.append(activity_distribution[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Create bar chart
        bars = ax_activity.bar(ordered_activities, ordered_durations, color=ordered_colors)
        
        # Add percentage and duration labels
        for i, activity in enumerate(ordered_activities):
            act_duration = activity_distribution[activity]['duration']
            act_percentage = activity_distribution[activity]['percentage']
            
            # Format duration
            formatted_duration = format_seconds_to_hms(act_duration)
            
            # Position label at top of bar
            bar = bars[i]
            height = bar.get_height()
            ax_activity.text(bar.get_x() + bar.get_width()/2., height,
                           f"{act_percentage:.1f}%\n{formatted_duration}", 
                           ha='center', va='bottom', fontsize=10)
        
        ax_activity.set_ylabel('Duration (hours)')
        ax_activity.set_title('Activity Distribution in Region', fontsize=14, pad=15)
        plt.setp(ax_activity.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_activity.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Employee contributions
        ax_contrib = fig.add_subplot(gs[1, 1])
        
        # Sort employees by contribution percentage
        sorted_contributions = sorted(employee_contributions, key=lambda x: x['contribution_percentage'], reverse=True)
        
        emp_ids = [contrib['id'] for contrib in sorted_contributions]
        contribution_pcts = [contrib['contribution_percentage'] for contrib in sorted_contributions]
        
        # Use employee colors
        emp_colors = [employee_colors.get(emp_id, '#2D5F91') for emp_id in emp_ids]
        
        # Create bar chart
        bars = ax_contrib.bar(emp_ids, contribution_pcts, color=emp_colors)
        
        # Add percentage and time labels
        for i, contrib in enumerate(sorted_contributions):
            formatted_duration = format_seconds_to_hms(contrib['time'])
            
            # Position label at top of bar
            bar = bars[i]
            height = bar.get_height()
            ax_contrib.text(bar.get_x() + bar.get_width()/2., height,
                          f"{contrib['contribution_percentage']:.1f}%\n{formatted_duration}", 
                          ha='center', va='bottom', fontsize=10)
        
        ax_contrib.set_ylabel('Contribution (%)')
        ax_contrib.set_title('Employee Contributions to Region', fontsize=14)
        plt.setp(ax_contrib.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_contrib.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Employee activity breakdown in this region
        ax_emp_activity = fig.add_subplot(gs[2, :])
        
        # Create stacked bar chart for employee activity breakdown
        # Only include the top 5 contributing employees if there are more
        top_contributors = sorted_contributions[:min(5, len(sorted_contributions))]
        
        # Prepare data for stacked bar
        emp_ids = [contrib['id'] for contrib in top_contributors]
        
        # Create data for each activity
        activity_data = {}
        for activity in activity_order:
            if activity in activity_distribution:
                percentages = []
                for contrib in top_contributors:
                    if activity in contrib['activity_breakdown']:
                        percentages.append(contrib['activity_breakdown'][activity]['percentage'])
                    else:
                        percentages.append(0)
                activity_data[activity] = percentages
        
        # Create stacked bar
        bottom = np.zeros(len(emp_ids))
        for activity in activity_order:
            if activity in activity_data:
                ax_emp_activity.bar(emp_ids, activity_data[activity], bottom=bottom, 
                                   label=activity, color=activity_colors.get(activity, '#CCCCCC'))
                bottom += activity_data[activity]
        
        ax_emp_activity.set_ylabel('Activity Percentage (%)')
        ax_emp_activity.set_title('Activity Breakdown by Employee in this Region', fontsize=14)
        ax_emp_activity.legend(loc='upper right')
        plt.setp(ax_emp_activity.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_emp_activity.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add title and save
        plt.suptitle(f"Region Ergonomic Assessment - {region}", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Use the save_figure utility from base
        save_figure(fig, output_path / f"{region}_region_report.png")