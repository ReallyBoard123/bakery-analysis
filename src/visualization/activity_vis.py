"""
Activity Visualization Module

Functions for creating visualizations of employee activities.
Including function from region_activity_analysis.py.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.gridspec as gridspec

from ..utils.time_utils import format_seconds_to_hms, format_seconds_to_hours
from .base import (save_figure, set_visualization_style, 
                  get_activity_colors, get_activity_order, 
                  get_employee_colors, get_bakery_cmap,
                  add_duration_percentage_label, get_text)

def plot_activity_distribution(data, save_path=None, figsize=(12, 7), language='en'):
    """
    Plot time distribution by activity
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with 'activity', 'duration', 'percentage', 'formatted_time' columns
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get standardized colors and order
    activity_colors = get_activity_colors()
    
    # Reorder data according to standard activity order if possible
    activity_order = get_activity_order()
    sorted_indices = []
    for activity in activity_order:
        if activity in data['activity'].values:
            sorted_indices.append(data[data['activity'] == activity].index[0])
    
    # Add any activities not in our standard order
    for idx in data.index:
        if idx not in sorted_indices:
            sorted_indices.append(idx)
    
    sorted_data = data.loc[sorted_indices].copy()
    
    # Translate activity labels for display
    display_activities = sorted_data['activity'].apply(lambda x: get_text(x, language))
    
    # Convert seconds to hours for better readability
    hours = sorted_data['duration'] / 3600
    
    # Create bar chart with activity-specific colors
    colors = [activity_colors.get(activity, '#CCCCCC') for activity in sorted_data['activity']]
    bars = ax.bar(display_activities, hours, color=colors)
    
    # Add percentage and time labels
    total_seconds = sorted_data['duration'].sum()
    for bar, seconds in zip(bars, sorted_data['duration']):
        add_duration_percentage_label(ax, bar, seconds, total_seconds, language=language)
    
    ax.set_title(get_text('Time Distribution by Activity', language), fontsize=16, fontweight='bold')
    ax.set_xlabel(get_text('Activity', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hours)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12, rotation=30, ha='right')
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_activity_distribution_by_employee(data, save_path=None, figsize=(14, 8), language='en'):
    """
    Plot activity distribution by employee as a heatmap
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with activities
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get standardized activity order
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
    
    # Create pivot table for percentages
    pivot_pct = pd.pivot_table(
        emp_activity,
        values='percentage',
        index='id',
        columns='activity',
        fill_value=0
    )
    
    # Create pivot table for formatted times (as strings)
    pivot_time = pd.pivot_table(
        emp_activity,
        values='formatted_time',
        index='id',
        columns='activity',
        aggfunc=lambda x: x.iloc[0] if len(x) > 0 else '00:00:00',
        fill_value='00:00:00'
    )
    
    # Reorder columns based on standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_pct.columns]
    other_cols = [col for col in pivot_pct.columns if col not in activity_order]
    ordered_cols.extend(other_cols)
    
    pivot_pct = pivot_pct[ordered_cols]
    pivot_time = pivot_time[ordered_cols]
    
    # Create a custom annotation function to show both percentage and time
    def custom_format(val, time_val):
        hours = time_val.split(':')[0]
        mins_secs = ':'.join(time_val.split(':')[1:])
        return f"{val:.1f}%\n{hours}:{mins_secs}"
    
    # Create annotation matrix
    annotations = np.empty_like(pivot_pct.values, dtype=object)
    for i in range(pivot_pct.shape[0]):
        for j in range(pivot_pct.shape[1]):
            annotations[i, j] = custom_format(
                pivot_pct.values[i, j], 
                pivot_time.values[i, j]
            )
    
    # Translate column names for display
    translated_columns = [get_text(col, language) for col in pivot_pct.columns]
    
    # Create heatmap with custom colormap and annotations
    sns.heatmap(
        pivot_pct, 
        cmap=get_bakery_cmap(), 
        annot=annotations,
        fmt="",
        linewidths=0.5,
        ax=ax,
        xticklabels=translated_columns,
        cbar_kws={
            'label': get_text('Percentage of Employee Time (%)', language),
            'shrink': 0.8
        }
    )
    
    ax.set_title(get_text('Activity Distribution by Employee', language), fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(get_text('Employee ID', language), fontsize=14)
    ax.set_xlabel(get_text('Activity', language), fontsize=14)
    
    # Rotate x axis labels for better readability
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_hourly_activity_patterns(hourly_activity, save_path=None, figsize=(16, 8), language='en'):
    """
    Plot activity patterns by hour of day
    
    Parameters:
    -----------
    hourly_activity : pandas.DataFrame
        Dataframe with hourly activity data
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get standardized colors and order
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Create pivot table for visualization
    pivot_data = hourly_activity.pivot_table(
        index='hour',
        columns='activity',
        values='duration',
        fill_value=0
    )
    
    # Reorder columns based on standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in activity_order]
    ordered_cols.extend(other_cols)
    
    pivot_data = pivot_data[ordered_cols]
    
    # Get colors for each activity
    colors = [activity_colors.get(activity, '#CCCCCC') for activity in pivot_data.columns]
    
    # Translate activity labels for legend
    translated_activities = [get_text(activity, language) for activity in pivot_data.columns]
    
    # Plot stacked area chart
    pivot_data.plot(
        kind='area', 
        stacked=True, 
        alpha=0.7, 
        color=colors,
        ax=ax,
        legend=False  # Don't create legend yet
    )
    
    # Create custom legend with translated labels
    handles, original_labels = ax.get_legend_handles_labels()
    ax.legend(handles, translated_activities, title=get_text('Activity', language), 
             fontsize=10, title_fontsize=12)
    
    # Convert seconds to hours on y-axis
    current_yticks = ax.get_yticks()
    ax.set_yticklabels([format_seconds_to_hms(tick) for tick in current_yticks])
    
    ax.set_title(get_text('Activity Patterns by Hour of Day', language), fontsize=16, fontweight='bold')
    ax.set_xlabel(get_text('Hour of Day', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hh:mm:ss)', language), fontsize=14)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    plt.xticks(range(0, 24), fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add total time labels at the top of each hour
    for hour in range(0, 24):
        if hour in pivot_data.index:
            total_seconds = pivot_data.loc[hour].sum()
            if total_seconds > 0:
                formatted_time = format_seconds_to_hms(total_seconds)
                plt.text(hour, total_seconds + (max(current_yticks) * 0.05), 
                        formatted_time, ha='center', va='bottom', 
                        fontsize=9, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def analyze_activities_by_region(data, department=None, top_n_regions=5, save_path=None, 
                               figsize=(16, 12), y_axis_scale='auto', language='en'):
    """
    Create a visualization showing top regions and activity breakdowns for employees in a department
    with both percentage and duration labels
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with tracking data
    department : str, optional
        Department to analyze ('Bread' or 'Cake'), if None, analyze all
    top_n_regions : int, optional
        Number of top regions to display per employee
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    y_axis_scale : str or float, optional
        How to scale the y-axis:
        - 'row': Scale each row (employee) consistently (default)
        - 'auto': Scale each subplot independently
        - 'global': Use the same scale for all subplots
        - float value: Use this specific maximum for all y-axes
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    
    # Filter by department if specified
    if department:
        dept_data = data[data['department'] == department]
    else:
        dept_data = data
    
    # Get standardized activity colors and order
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Get employees in this department
    employees = sorted(dept_data['id'].unique())
    
    if len(employees) == 0:
        print(get_text('No employees found for department: {0}', language).format(department))
        return None
    
    # Calculate max hours for scaling
    max_hours_global = 0
    max_hours_by_employee = {}
    max_hours_by_subplot = {}
    
    # First pass to calculate maximum values for scaling
    for emp_idx, employee_id in enumerate(employees):
        emp_data = dept_data[dept_data['id'] == employee_id]
        
        # Get top regions for this employee
        region_durations = emp_data.groupby('region')['duration'].sum().reset_index()
        region_durations = region_durations.sort_values('duration', ascending=False)
        top_regions = region_durations.head(top_n_regions)
        
        # Calculate maximum activity value for any region for this employee
        max_hours_emp = 0
        
        for reg_idx, (_, region_row) in enumerate(top_regions.iterrows()):
            region = region_row['region']
            region_data = emp_data[emp_data['region'] == region]
            
            # Find max activity value for this region
            activity_max = 0
            for activity in activity_order:
                activity_data = region_data[region_data['activity'] == activity]
                activity_hours = activity_data['duration'].sum() / 3600
                activity_max = max(activity_max, activity_hours)
            
            # Save max for this subplot
            max_hours_by_subplot[(emp_idx, reg_idx)] = activity_max * 1.2  # Add 20% padding
            
            # Update employee max
            max_hours_emp = max(max_hours_emp, activity_max)
        
        # Save employee max with padding
        max_hours_by_employee[employee_id] = max_hours_emp * 1.2  # Add 20% padding
        
        # Update global max
        max_hours_global = max(max_hours_global, max_hours_emp)
    
    # Add 20% padding to global max
    max_hours_global *= 1.2
    
    # Create figure
    fig, axes = plt.subplots(nrows=len(employees), ncols=top_n_regions, figsize=figsize, sharey=False)  # Explicitly disable sharey
    
    # Make sure axes is always a 2D array
    if len(employees) == 1 and top_n_regions == 1:
        axes = np.array([[axes]])
    elif len(employees) == 1:
        axes = np.array([axes])
    elif top_n_regions == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Add a title to the figure
    dept_title = get_text("{0} Department", language).format(get_text(department, language)) if department else get_text("All Departments", language)
    fig.suptitle(get_text("Activity Analysis by Region - {0}", language).format(dept_title), fontsize=20, fontweight='bold', y=0.98, color='#333333')
    
    # Process each employee
    for emp_idx, employee_id in enumerate(employees):
        # Filter data for this employee
        emp_data = dept_data[dept_data['id'] == employee_id]
        
        # Calculate total hours for this employee
        total_duration = emp_data['duration'].sum()
        total_hours = total_duration / 3600
        
        # Calculate time spent in each region
        region_durations = emp_data.groupby('region')['duration'].sum().reset_index()
        region_durations = region_durations.sort_values('duration', ascending=False)
        region_durations['percentage'] = (region_durations['duration'] / total_duration * 100).round(1)
        region_durations['hours'] = region_durations['duration'] / 3600
        
        # Get top N regions
        top_regions = region_durations.head(top_n_regions)
        
        # Employee label at the left side - use employee-specific color
        emp_color = employee_colors.get(employee_id, '#333333')
        if top_n_regions > 0:
            axes[emp_idx, 0].text(-0.5, 0.5, get_text("Employee {0}\n{1:.1f} hours", language).format(employee_id, total_hours), 
                                 ha='center', va='center', fontsize=12, fontweight='bold',
                                 transform=axes[emp_idx, 0].transAxes, 
                                 color=emp_color)
        
        # For each top region
        for reg_idx, (_, region_row) in enumerate(top_regions.iterrows()):
            region = region_row['region']
            region_duration = region_row['duration']
            region_pct = region_row['percentage']
            region_hours = region_row['hours']
            
            # Filter data for this region
            region_data = emp_data[emp_data['region'] == region]
            
            # Calculate activity breakdown within this region
            activity_durations = {}
            activity_seconds = {}
            for activity in activity_order:
                activity_data = region_data[region_data['activity'] == activity]
                activity_seconds[activity] = activity_data['duration'].sum()  # Raw seconds for formatting
                activity_durations[activity] = activity_seconds[activity] / 3600  # Convert to hours for plotting
            
            # Get current axis
            ax = axes[emp_idx, reg_idx]
            
            # Create a bar plot for activities
            activities = []
            hours = []
            colors = []
            seconds = []
            
            for activity in activity_order:
                if activity_durations[activity] > 0:
                    activities.append(activity)
                    hours.append(activity_durations[activity])
                    seconds.append(activity_seconds[activity])
                    colors.append(activity_colors[activity])
            
            # Create bar plot with activities - showing hours
            bars = ax.bar(activities, hours, color=colors)
            
            # Set y-axis limits based on scaling parameter
            if isinstance(y_axis_scale, (int, float)):
                ax.set_ylim([0, float(y_axis_scale)])
            elif y_axis_scale == 'auto':
                ax.set_ylim([0, max_hours_by_subplot.get((emp_idx, reg_idx), 1)])
            elif y_axis_scale == 'row':
                ax.set_ylim([0, max_hours_by_employee[employee_id]])
            else:  # 'global' or any other value
                ax.set_ylim([0, max_hours_global])
            
            # Add percentage labels and duration on top of bars
            for bar, activity, duration_seconds in zip(bars, activities, seconds):
                height = bar.get_height()
                percentage = (height / region_hours * 100).round(1) if region_hours > 0 else 0
                
                # Format duration as hh:mm:ss
                formatted_duration = format_seconds_to_hms(duration_seconds)
                
                if percentage >= 5:  # Only show labels for significant percentages
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{percentage:.1f}%\n{formatted_duration}', 
                           ha='center', va='bottom', fontsize=8)
            
            # Set title with region name and percentage of total time
            region_formatted_time = format_seconds_to_hms(region_duration)
            ax.set_title(f'{region}\n{region_hours:.1f}h ({region_pct:.1f}% {get_text("of total", language)})', fontsize=10, pad=15)
            
            # Remove x-axis labels as we have a legend
            ax.set_xticklabels([])
            
            # Only show y-axis label for first column
            if reg_idx == 0:
                ax.set_ylabel(get_text('Hours', language))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
    
    # Add one legend for the entire figure
    handles = [plt.Rectangle((0, 0), 1, 1, color=activity_colors[activity]) for activity in activity_order]
    labels = [get_text(activity, language) for activity in activity_order]
    fig.legend(handles, labels, loc='lower center', ncol=len(activity_order), 
              bbox_to_anchor=(0.5, 0.01), fontsize=10)
    
    # Add extra space at the bottom for the legend
    plt.subplots_adjust(bottom=0.08)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig