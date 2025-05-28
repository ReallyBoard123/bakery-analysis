"""
Ergonomics Reports Module

Contains functions for generating comprehensive reports from ergonomic analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.patches as patches
import matplotlib.cm as cm

from src.utils.time_utils import format_seconds_to_hms
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import (
    save_figure, set_visualization_style, 
    get_activity_colors, get_activity_order, 
    get_employee_colors, get_text
)

def generate_ergonomic_report(employee_scores, output_dir, language='en'):
    """
    Generate detailed ergonomic report visualizations for each employee
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries from calculate_ergonomic_score
    output_dir : Path
        Directory to save the reports
    language : str, optional
        Language code ('en' or 'de')
    
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
        ax_main.text(0.5, 0.7, get_text("Ergonomic Score: {0}/100", language).format(f"{score:.1f}"), 
                    fontsize=28, ha='center', color=score_color, fontweight='bold')
        
        # Add comparison to other employees
        comparison_text = get_text("Average: {0:.1f} | Min: {1:.1f} | Max: {2:.1f}", language).format(avg_score, min_score, max_score)
        ax_main.text(0.5, 0.5, comparison_text, fontsize=14, ha='center')
        
        # Add employee ID and total duration
        hours = emp_data['total_duration'] / 3600
        ax_main.text(0.5, 0.3, get_text("Employee: {0} | Hours Analyzed: {1:.1f}", language).format(emp_id, hours), 
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
        bars = ax_deduct.barh([get_text(act, language) for act in activities], values, color=colors)
        ax_deduct.set_xlabel(get_text('Deduction Points', language))
        ax_deduct.set_title(get_text('Ergonomic Deductions by Activity', language), fontsize=14)
        
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
        bars = ax_activity.bar([get_text(act, language) for act in ordered_activities], ordered_durations, color=ordered_colors)
        
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
        
        ax_activity.set_ylabel(get_text('Duration (hours)', language))
        ax_activity.set_title(get_text('Activity Time Distribution', language), fontsize=14)
        plt.setp(ax_activity.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_activity.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add title and save
        plt.suptitle(get_text("Ergonomic Assessment Report - Employee {0}", language).format(emp_id), fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Use the save_figure utility from base
        save_figure(fig, output_path / f"{emp_id}_ergonomic_report.png")

def generate_region_ergonomic_report(region_analyses, output_dir, language='en'):
    """
    Generate ergonomic reports for each analyzed region
    
    Parameters:
    -----------
    region_analyses : dict
        Dictionary of region analyses from analyze_region_ergonomics
    output_dir : Path
        Directory to save the reports
    language : str, optional
        Language code ('en' or 'de')
    
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
        ax_main.text(0.5, 0.7, get_text("Region Ergonomic Score: {0}/100", language).format(f"{score:.1f}"), 
                    fontsize=28, ha='center', color=score_color, fontweight='bold')
        
        # Add region info
        hours = total_duration / 3600
        qualified_employees = analysis['qualified_employees']
        ax_main.text(0.5, 0.4, get_text("Region: {0} | Hours Analyzed: {1:.1f} | Employees: {2}", language).format(
            region, hours, qualified_employees), 
            fontsize=16, ha='center', fontweight='bold')
        
        # 2. Activity distribution in this region (keep the original bar chart)
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
        bars = ax_activity.bar([get_text(act, language) for act in ordered_activities], ordered_durations, color=ordered_colors)
        
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
        
        ax_activity.set_ylabel(get_text('Duration (hours)', language))
        ax_activity.set_title(get_text('Activity Distribution in Region', language), fontsize=14, pad=15)
        plt.setp(ax_activity.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_activity.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Employee contributions as a table
        ax_contrib_table = fig.add_subplot(gs[1, 1])
        ax_contrib_table.axis('off')
        
        # Sort employees by contribution percentage
        sorted_contributions = sorted(employee_contributions, key=lambda x: x['contribution_percentage'], reverse=True)
        
        # Prepare table data
        contrib_table_data = [[
            get_text("Employee", language),
            get_text("Duration", language),
            get_text("Contribution", language),
            get_text("Region %", language)
        ]]
        
        for contrib in sorted_contributions:
            contrib_table_data.append([
                contrib['id'],
                format_seconds_to_hms(contrib['time']),
                f"{contrib['contribution_percentage']:.1f}%",
                f"{contrib['region_percentage']:.1f}%"
            ])
        
        # Create table
        contrib_table = ax_contrib_table.table(
            cellText=contrib_table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        
        # Style table
        contrib_table.auto_set_font_size(False)
        contrib_table.set_fontsize(10)
        contrib_table.scale(1, 1.5)
        
        # Header formatting
        for j in range(4):
            contrib_table[(0, j)].set_facecolor('#4472C4')
            contrib_table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        ax_contrib_table.set_title(get_text('Employee Contributions to Region', language), fontsize=14, pad=15)
        
        # 4. Employee activity breakdown as a table
        ax_breakdown_table = fig.add_subplot(gs[2, :])
        ax_breakdown_table.axis('off')
        
        # Only include the top 5 contributing employees
        top_contributors = sorted_contributions[:min(5, len(sorted_contributions))]
        
        # Prepare table data
        breakdown_table_data = [[get_text("Employee", language)]]
        
        # Add activity columns
        for activity in activity_order:
            if activity in activity_distribution:
                breakdown_table_data[0].append(get_text(activity, language))
        
        # Add data rows for each employee
        for contrib in top_contributors:
            row = [contrib['id']]
            for activity in activity_order:
                if activity in activity_distribution:
                    if activity in contrib['activity_breakdown']:
                        act_data = contrib['activity_breakdown'][activity]
                        row.append(f"{act_data['percentage']:.1f}%\n({format_seconds_to_hms(act_data['duration'])})")
                    else:
                        row.append("0%")
            breakdown_table_data.append(row)
        
        # Create table
        breakdown_table = ax_breakdown_table.table(
            cellText=breakdown_table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.2] + [0.8/(len(breakdown_table_data[0])-1)] * (len(breakdown_table_data[0])-1)
        )
        
        # Style table
        breakdown_table.auto_set_font_size(False)
        breakdown_table.set_fontsize(10)
        breakdown_table.scale(1, 1.5)
        
        # Header formatting
        for j in range(len(breakdown_table_data[0])):
            breakdown_table[(0, j)].set_facecolor('#4472C4')
            breakdown_table[(0, j)].set_text_props(color='white', fontweight='bold')
            
        # Color activity columns based on activity colors
        for j, header in enumerate(breakdown_table_data[0][1:], 1):
            # Get the activity name from the header
            for act in activity_order:
                if get_text(act, language) == header:
                    color = activity_colors.get(act, '#CCCCCC')
                    for i in range(1, len(breakdown_table_data)):
                        breakdown_table[(i, j)].set_facecolor(color)
                        breakdown_table[(i, j)].set_alpha(0.3)
        
        ax_breakdown_table.set_title(get_text('Activity Breakdown by Employee in this Region', language), fontsize=14, pad=15)
        
        # Add title and save
        plt.suptitle(get_text("Region Ergonomic Assessment - {0}", language).format(region), fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Use the save_figure utility from base
        save_figure(fig, output_path / f"{region}_region_report.png")

def generate_all_employees_comparison(employee_scores, output_dir, language='en'):
    """
    Generate a combined visualization comparing all employees' ergonomic scores
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries
    output_dir : Path
        Directory to save the visualization
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard colors
    employee_colors = get_employee_colors()
    activity_colors = get_activity_colors()
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Sort employees by score
    sorted_employees = sorted(employee_scores, key=lambda x: x['final_score'], reverse=True)
    
    # 1. Bar chart of ergonomic scores
    ax_scores = fig.add_subplot(gs[0, 0])
    
    emp_ids = [emp['employee_id'] for emp in sorted_employees]
    scores = [emp['final_score'] for emp in sorted_employees]
    colors = [employee_colors.get(emp_id, '#333333') for emp_id in emp_ids]
    
    # Create horizontal bar chart
    bars = ax_scores.barh(emp_ids, scores, color=colors)
    
    # Add score values
    for bar in bars:
        width = bar.get_width()
        ax_scores.text(width + 1, bar.get_y() + bar.get_height()/2, 
                      f"{width:.1f}", va='center', fontweight='bold')
    
    # Add red-yellow-green background zones
    ax_scores.axvspan(0, 60, alpha=0.1, color='red')
    ax_scores.axvspan(60, 75, alpha=0.1, color='yellow')
    ax_scores.axvspan(75, 100, alpha=0.1, color='green')
    
    # Add text labels for zones
    ax_scores.text(30, -0.5, get_text("Poor", language), ha='center', va='top', color='red', fontsize=10)
    ax_scores.text(67.5, -0.5, get_text("Fair", language), ha='center', va='top', color='orange', fontsize=10)
    ax_scores.text(87.5, -0.5, get_text("Good", language), ha='center', va='top', color='green', fontsize=10)
    
    ax_scores.set_xlabel(get_text('Ergonomic Score', language), fontsize=12)
    ax_scores.set_title(get_text('Employee Ergonomic Scores', language), fontsize=14)
    ax_scores.set_xlim(0, 102)  # Leave room for labels
    
    # 2. Activity distribution heatmap
    ax_heatmap = fig.add_subplot(gs[0, 1])
    
    # Prepare data for heatmap
    heatmap_data = []
    for emp in sorted_employees:
        components = emp['components']
        act_breakdown = components['activity_breakdown']
        row = {'employee': emp['employee_id']}
        
        for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
            if activity in act_breakdown:
                row[activity] = act_breakdown[activity]['percentage']
            else:
                row[activity] = 0
        
        heatmap_data.append(row)
    
    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index('employee', inplace=True)
    
    # Create heatmap
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".1f", 
        cmap="YlOrRd", 
        cbar_kws={'label': get_text('Percentage of Time (%)', language)},
        ax=ax_heatmap
    )
    
    ax_heatmap.set_title(get_text('Activity Distribution by Employee (%)', language), fontsize=14)
    ax_heatmap.set_ylabel('')  # Remove y-label as it's redundant with the bar chart
    
    # 3. Deduction factors stacked bar chart
    ax_deductions = fig.add_subplot(gs[1, 0])
    
    # Prepare deduction data
    deduction_data = []
    for emp in sorted_employees:
        components = emp['components']
        deductions = components['deductions']
        row = {'employee': emp['employee_id']}
        
        for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
            row[activity] = deductions.get(activity, 0)
        
        deduction_data.append(row)
    
    # Convert to DataFrame
    deduction_df = pd.DataFrame(deduction_data)
    deduction_df.set_index('employee', inplace=True)
    
    # Create stacked bar chart
    deduction_df.plot(
        kind='barh', 
        stacked=True, 
        color=[activity_colors.get(col, '#CCCCCC') for col in deduction_df.columns],
        ax=ax_deductions
    )
    
    ax_deductions.set_title(get_text('Ergonomic Deduction Factors by Employee', language), fontsize=14)
    ax_deductions.set_xlabel(get_text('Deduction Points', language), fontsize=12)
    ax_deductions.legend(title=get_text('Activity', language))
    
    # 4. Key metrics table
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_metrics.axis('off')
    
    # Prepare metrics data
    metrics_data = [[
        get_text("Employee", language),
        get_text("Score", language),
        get_text("Hours", language),
        get_text("Handle up %", language),
        get_text("Handle down %", language),
        get_text("Stand %", language)
    ]]
    
    for emp in sorted_employees:
        components = emp['components']
        act_breakdown = components['activity_breakdown']
        
        handle_up_pct = act_breakdown.get('Handle up', {}).get('percentage', 0)
        handle_down_pct = act_breakdown.get('Handle down', {}).get('percentage', 0)
        stand_pct = act_breakdown.get('Stand', {}).get('percentage', 0)
        
        metrics_data.append([
            emp['employee_id'],
            f"{emp['final_score']:.1f}",
            f"{emp['total_duration']/3600:.1f}",
            f"{handle_up_pct:.1f}%",
            f"{handle_down_pct:.1f}%",
            f"{stand_pct:.1f}%"
        ])
    
    # Create table
    metrics_table = ax_metrics.table(
        cellText=metrics_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Style table
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(10)
    metrics_table.scale(1, 1.5)
    
    # Header formatting
    for j in range(len(metrics_data[0])):
        metrics_table[(0, j)].set_facecolor('#4472C4')
        metrics_table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color code scores
    for i in range(1, len(metrics_data)):
        score = float(metrics_data[i][1])
        if score >= 75:
            metrics_table[(i, 1)].set_facecolor('lightgreen')
        elif score >= 60:
            metrics_table[(i, 1)].set_facecolor('lightyellow')
        else:
            metrics_table[(i, 1)].set_facecolor('lightcoral')
    
    ax_metrics.set_title(get_text('Employee Ergonomic Metrics Summary', language), fontsize=14)
    
    # Add main title
    plt.suptitle(get_text("Bakery Employee Ergonomic Comparison", language), fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_figure(fig, output_path / "employee_ergonomic_comparison.png")

def generate_all_regions_comparison(region_analyses, output_dir, language='en'):
    """
    Generate a combined visualization comparing all regions' ergonomic metrics
    
    Parameters:
    -----------
    region_analyses : dict
        Dictionary of region analyses
    output_dir : Path
        Directory to save the visualization
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard colors
    activity_colors = get_activity_colors()
    
    # Filter out regions with no data
    valid_regions = {r: data for r, data in region_analyses.items() if 'ergonomic_score' in data}
    
    # If no valid regions, return
    if not valid_regions:
        print(f"No valid regions found for comparison")
        return
    
    # Sort regions by ergonomic score
    sorted_regions = sorted(valid_regions.items(), key=lambda x: x[1]['ergonomic_score'], reverse=True)
    
    # Get list of regions to display (top 15 to avoid overcrowding)
    display_regions = sorted_regions[:15]
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # 1. Bar chart of ergonomic scores
    ax_scores = fig.add_subplot(gs[0, 0])
    
    region_names = [r[0] for r in display_regions]
    scores = [r[1]['ergonomic_score'] for r in display_regions]
    
    # Truncate long region names
    display_names = [name[:15] + '...' if len(name) > 15 else name for name in region_names]
    
    # Create horizontal bar chart with color gradient based on score
    cmap = plt.cm.get_cmap('RdYlGn')
    score_colors = [cmap(score/100) for score in scores]
    
    bars = ax_scores.barh(display_names, scores, color=score_colors)
    
    # Add score values
    for bar in bars:
        width = bar.get_width()
        ax_scores.text(width + 1, bar.get_y() + bar.get_height()/2, 
                      f"{width:.1f}", va='center', fontweight='bold')
    
    # Add red-yellow-green background zones
    ax_scores.axvspan(0, 60, alpha=0.1, color='red')
    ax_scores.axvspan(60, 75, alpha=0.1, color='yellow')
    ax_scores.axvspan(75, 100, alpha=0.1, color='green')
    
    # Add text labels for zones
    ax_scores.text(30, -0.5, get_text("Poor", language), ha='center', va='top', color='red', fontsize=10)
    ax_scores.text(67.5, -0.5, get_text("Fair", language), ha='center', va='top', color='orange', fontsize=10)
    ax_scores.text(87.5, -0.5, get_text("Good", language), ha='center', va='top', color='green', fontsize=10)
    
    ax_scores.set_xlabel(get_text('Ergonomic Score', language), fontsize=12)
    ax_scores.set_title(get_text('Top Regions by Ergonomic Score', language), fontsize=14)
    ax_scores.set_xlim(0, 102)  # Leave room for labels
    
    # 2. Activity distribution heatmap
    ax_heatmap = fig.add_subplot(gs[0, 1])
    
    # Prepare data for heatmap
    heatmap_data = []
    for region, data in display_regions:
        activity_distribution = data['activity_distribution']
        row = {'region': region}
        
        for activity in ['Handle up', 'Handle down', 'Handle center']:
            if activity in activity_distribution:
                row[activity] = activity_distribution[activity]['percentage']
            else:
                row[activity] = 0
        
        heatmap_data.append(row)
    
    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index('region', inplace=True)
    
    # Truncate index names
    heatmap_df.index = [idx[:15] + '...' if len(idx) > 15 else idx for idx in heatmap_df.index]
    
    # Create heatmap
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".1f", 
        cmap="YlOrRd", 
        cbar_kws={'label': get_text('Percentage of Time (%)', language)},
        ax=ax_heatmap
    )
    
    ax_heatmap.set_title(get_text('Activity Distribution by Region (%)', language), fontsize=14)
    ax_heatmap.set_ylabel('')  # Remove y-label as it's redundant with the bar chart
    
    # 3. Region usage bubble chart
    ax_bubble = fig.add_subplot(gs[1, 0])
    
    # Prepare bubble chart data
    bubble_data = []
    for region, data in display_regions:
        bubble_data.append({
            'region': region[:10] + '...' if len(region) > 10 else region,
            'score': data['ergonomic_score'],
            'hours': data['total_duration'] / 3600,
            'employees': data['qualified_employees']
        })
    
    # Convert to DataFrame
    bubble_df = pd.DataFrame(bubble_data)
    
    # Create bubble chart - score vs hours with bubble size as employee count
    scatter = ax_bubble.scatter(
        bubble_df['hours'], 
        bubble_df['score'], 
        s=bubble_df['employees'] * 100,  # Scale bubble size
        c=bubble_df['score'],  # Color by score
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )
    
    # Add region labels
    for i, row in bubble_df.iterrows():
        ax_bubble.annotate(
            row['region'], 
            (row['hours'], row['score']),
            xytext=(7, 0),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax_bubble)
    cbar.set_label(get_text('Ergonomic Score', language))
    
    # Add a legend for bubble size
    sizes = [1, 3, 5, 7]
    labels = [f"{s} {get_text('employees', language)}" for s in sizes]
    legend_bubbles = []
    for size in sizes:
        legend_bubbles.append(ax_bubble.scatter([], [], s=size*100, c='gray', alpha=0.7, edgecolors='black'))
    
    ax_bubble.legend(legend_bubbles, labels, loc='lower right', title=get_text('Employees', language))
    
    ax_bubble.set_xlabel(get_text('Usage (hours)', language), fontsize=12)
    ax_bubble.set_ylabel(get_text('Ergonomic Score', language), fontsize=12)
    ax_bubble.set_title(get_text('Region Usage vs. Ergonomic Score', language), fontsize=14)
    ax_bubble.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Key metrics table
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_metrics.axis('off')
    
    # Prepare metrics data
    metrics_data = [[
        get_text("Region", language),
        get_text("Score", language),
        get_text("Hours", language),
        get_text("Employees", language),
        get_text("Handle up %", language),
        get_text("Handle down %", language)
    ]]
    
    for region, data in display_regions:
        activity_distribution = data['activity_distribution']
        handle_up_pct = activity_distribution.get('Handle up', {}).get('percentage', 0)
        handle_down_pct = activity_distribution.get('Handle down', {}).get('percentage', 0)
        
        metrics_data.append([
            region[:15] + '...' if len(region) > 15 else region,
            f"{data['ergonomic_score']:.1f}",
            f"{data['total_duration']/3600:.1f}",
            str(data['qualified_employees']),
            f"{handle_up_pct:.1f}%",
            f"{handle_down_pct:.1f}%"
        ])
    
    # Create table
    metrics_table = ax_metrics.table(
        cellText=metrics_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Style table
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(10)
    metrics_table.scale(1, 1.5)
    
    # Header formatting
    for j in range(len(metrics_data[0])):
        metrics_table[(0, j)].set_facecolor('#4472C4')
        metrics_table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color code scores
    for i in range(1, len(metrics_data)):
        score = float(metrics_data[i][1])
        if score >= 75:
            metrics_table[(i, 1)].set_facecolor('lightgreen')
        elif score >= 60:
            metrics_table[(i, 1)].set_facecolor('lightyellow')
        else:
            metrics_table[(i, 1)].set_facecolor('lightcoral')
    
    ax_metrics.set_title(get_text('Region Ergonomic Metrics Summary', language), fontsize=14)
    
    # Add main title
    plt.suptitle(get_text("Bakery Region Ergonomic Comparison", language), fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_figure(fig, output_path / "region_ergonomic_comparison.png")

def generate_circular_packing_visualization(employee_scores, region_analyses, output_dir, language='en'):
    """
    Generate a treemap visualization showing employees and regions 
    grouped by ergonomic score
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries
    region_analyses : dict
        Dictionary of region analyses
    output_dir : Path
        Directory to save the visualization
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    try:
        import squarify
    except ImportError:
        print(f"Warning: squarify module not found. Installing required packages for treemap visualization.")
        import subprocess
        subprocess.check_call(["pip", "install", "squarify"])
        import squarify
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Employee treemap
    ax_emp = fig.add_subplot(1, 2, 1)
    
    # Group employees by score range
    score_ranges = {
        "Good (75-100)": [],
        "Fair (60-75)": [],
        "Poor (0-60)": []
    }
    
    for emp in employee_scores:
        score = emp['final_score']
        emp_id = emp['employee_id']
        hours = emp['total_duration'] / 3600
        
        if score >= 75:
            score_ranges["Good (75-100)"].append((emp_id, score, hours))
        elif score >= 60:
            score_ranges["Fair (60-75)"].append((emp_id, score, hours))
        else:
            score_ranges["Poor (0-60)"].append((emp_id, score, hours))
    
    # Prepare data for treemap
    treemap_data = []
    labels = []
    colors = []
    
    # Create nested data for treemap
    for category, employees in score_ranges.items():
        if not employees:
            continue
            
        # Sort employees by score
        sorted_emps = sorted(employees, key=lambda x: x[1], reverse=True)
        
        # Add entries for each employee
        for emp_id, score, hours in sorted_emps:
            # Value for treemap is hours worked
            treemap_data.append(hours)
            
            # Label is employee ID and score
            labels.append(f"{emp_id}\n{score:.1f}")
            
            # Color based on score
            if score >= 75:
                colors.append('#78C47B')  # Good - Green
            elif score >= 60:
                colors.append('#F0CF57')  # Fair - Yellow
            else:
                colors.append('#E07F70')  # Poor - Red
    
    # Create treemap
    if treemap_data:
        squarify.plot(
            sizes=treemap_data,
            label=labels,
            color=colors,
            alpha=0.8,
            pad=True,
            ax=ax_emp
        )
    
    ax_emp.set_xticks([])
    ax_emp.set_yticks([])
    ax_emp.set_title(get_text("Employee Ergonomic Scores", language), fontsize=16)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#78C47B', edgecolor='k', alpha=0.8, label=get_text('Good (75-100)', language)),
        patches.Patch(facecolor='#F0CF57', edgecolor='k', alpha=0.8, label=get_text('Fair (60-75)', language)),
        patches.Patch(facecolor='#E07F70', edgecolor='k', alpha=0.8, label=get_text('Poor (0-60)', language))
    ]
    
    ax_emp.legend(handles=legend_elements, loc='upper right')
    
    # 2. Region treemap
    ax_region = fig.add_subplot(1, 2, 2)
    
    # Filter valid regions
    valid_regions = {r: data for r, data in region_analyses.items() if 'ergonomic_score' in data}
    
    # Group regions by score range
    region_score_ranges = {
        "Good (75-100)": [],
        "Fair (60-75)": [],
        "Poor (0-60)": []
    }
    
    for region, data in valid_regions.items():
        score = data['ergonomic_score']
        hours = data['total_duration'] / 3600
        employees = data['qualified_employees']
        
        if score >= 75:
            region_score_ranges["Good (75-100)"].append((region, score, hours, employees))
        elif score >= 60:
            region_score_ranges["Fair (60-75)"].append((region, score, hours, employees))
        else:
            region_score_ranges["Poor (0-60)"].append((region, score, hours, employees))
    # Prepare data for treemap
    region_treemap_data = []
    region_labels = []
    region_colors = []
    # Create nested data for treemap
    for category, regions in region_score_ranges.items():
        if not regions:
            continue
            
        # Sort regions by score
        sorted_regions = sorted(regions, key=lambda x: x[1], reverse=True)
        
        # Add entries for each region
        for region, score, hours, employees in sorted_regions:
            # Value for treemap is hours worked
            region_treemap_data.append(hours)
            
            # Label is region name and score
            region_labels.append(f"{region}\n{score:.1f}")
            
            # Color based on score
            if score >= 75:
                region_colors.append('#78C47B')  # Good - Green
            elif score >= 60:
                region_colors.append('#F0CF57')  # Fair - Yellow
            else:
                region_colors.append('#E07F70')  # Poor - Red
    # Create treemap
    if region_treemap_data:
        squarify.plot(
            sizes=region_treemap_data,
            label=region_labels,
            color=region_colors,
            alpha=0.8,
            pad=True,
            ax=ax_region
        )
    ax_region.set_xticks([])
    ax_region.set_yticks([])
    ax_region.set_title(get_text("Region Ergonomic Scores", language), fontsize=16)
    # Add legend
    ax_region.legend(handles=legend_elements, loc='upper right')
    # Add main title
    plt.suptitle(get_text("Bakery Ergonomic Scores Comparison", language), fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save the figure
    save_figure(fig, output_path / "ergonomic_scores_treemap.png")
    