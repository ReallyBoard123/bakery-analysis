"""
Common Visualization Module

Provides common/basic visualizations for the bakery employee tracking analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from .base import (save_figure, set_visualization_style, 
                  get_activity_colors, get_activity_order, 
                  get_employee_colors, get_department_colors,
                  add_duration_percentage_label, get_text)
from ..utils.time_utils import format_seconds_to_hms, format_seconds_to_hours

def plot_activity_by_shift(data, save_path=None, figsize=(15, 8), language='en'):
    """
    Plot total activity duration by shift
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
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
    
    # Use original data without filtering for isPauseData as it might not exist
    filtered_data = data
    
    # Aggregate data by shift
    shift_data = filtered_data.groupby('shift')['duration'].sum().reset_index()
    
    # Sort by shift
    shift_data = shift_data.sort_values('shift')
    
    # Convert seconds to hours
    shift_data['hours'] = shift_data['duration'] / 3600
    
    # Create bar chart - using color from department colors
    department_colors = get_department_colors()
    default_color = '#2D5F91'  # Default blue if no mapping
    bars = ax.bar(shift_data['shift'], shift_data['hours'], color=default_color)
    
    # Add duration and percentage labels
    total_seconds = shift_data['duration'].sum()
    for bar, row in zip(bars, shift_data.iterrows()):
        _, row_data = row
        percentage = (row_data['duration'] / total_seconds * 100).round(1)
        formatted_duration = format_seconds_to_hms(row_data['duration'])
        
        # Larger font for labels
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f"{percentage}%\n{formatted_duration}", 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title(get_text('Total Activity Duration by Shift', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Shift', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hours)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_employee_summary(data, save_path=None, figsize=(12, 8), language='en'):
    """
    Plot total duration by employee, colored by department
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with 'department' column
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
    
    # No filtering for isPauseData or shift=6 as they might not exist
    filtered_data = data
    
    # Get department colors
    department_colors = get_department_colors()
    
    # Aggregate data by employee
    employee_data = filtered_data.groupby(['id', 'department'])['duration'].sum().reset_index()
    
    # Sort by department and total duration
    employee_data = employee_data.sort_values(['department', 'duration'], ascending=[True, False])
    
    # Convert seconds to hours
    employee_data['hours'] = employee_data['duration'] / 3600
    
    # Create bar chart with department-specific colors
    colors = [department_colors.get(dept, '#CCCCCC') for dept in employee_data['department']]
    bars = ax.bar(employee_data['id'], employee_data['hours'], color=colors)
    
    # Add duration and percentage labels
    total_seconds = employee_data['duration'].sum()
    for bar, row in zip(bars, employee_data.iterrows()):
        _, row_data = row
        percentage = (row_data['duration'] / total_seconds * 100).round(1)
        formatted_duration = format_seconds_to_hms(row_data['duration'])
        
        # Larger font for labels
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f"{percentage}%\n{formatted_duration}", 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add department legend - horizontal placement at bottom
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=department_colors[dept]) 
                     for dept in sorted(employee_data['department'].unique())]
    legend_labels = [get_text(dept, language) for dept in sorted(employee_data['department'].unique())]
    ax.legend(legend_elements, legend_labels, title=get_text('Department', language), 
             loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    ax.set_title(get_text('Total Duration by Employee', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hours)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_department_comparison(data, save_path=None, figsize=(14, 8), language='en'):
    """
    Compare activity breakdown between departments
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with 'department' column
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
    
    # No filtering for isPauseData or shift as they might not exist
    filtered_data = data
    
    # Get colors
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Aggregate data by department and activity
    dept_activity = filtered_data.groupby(['department', 'activity'])['duration'].sum().reset_index()
    
    # Calculate total by department for percentages
    dept_total = filtered_data.groupby('department')['duration'].sum().reset_index()
    dept_total = dept_total.rename(columns={'duration': 'dept_total'})
    
    # Merge to get department totals
    dept_activity = pd.merge(dept_activity, dept_total, on='department')
    
    # Calculate percentage
    dept_activity['percentage'] = (dept_activity['duration'] / dept_activity['dept_total'] * 100).round(1)
    
    # Create a table to store duration information for labeling
    duration_info = {}
    for _, row in dept_activity.iterrows():
        key = (row['department'], row['activity'])
        duration_info[key] = {
            'duration': row['duration'],
            'percentage': row['percentage'],
            'formatted_time': format_seconds_to_hms(row['duration'])
        }
    
    # Pivot data for plotting
    pivot_data = dept_activity.pivot_table(
        index='department',
        columns='activity',
        values='percentage',
        fill_value=0
    )
    
    # Reorder columns based on standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in ordered_cols]
    ordered_cols.extend(other_cols)
    
    pivot_data = pivot_data[ordered_cols]
    
    # Get colors for the stacked bars - using the defined activity colors
    bar_colors = [activity_colors.get(act, '#CCCCCC') for act in pivot_data.columns]
    
    # Translate activity names for legend
    translated_activities = [get_text(act, language) for act in pivot_data.columns]
    
    # Plot stacked bars
    ax = pivot_data.plot(kind='bar', stacked=True, color=bar_colors, ax=ax)
    
    # Update legend with translated labels - horizontal placement at bottom
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, translated_activities, title=get_text('Activity', language),
             loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(pivot_data.columns))
    
    # Translate department names on x-axis
    ax.set_xticklabels([get_text(dept, language) for dept in pivot_data.index])
    
    # Add percentage and duration labels on segments
    for i, dept in enumerate(pivot_data.index):
        cumulative_height = 0
        for activity in pivot_data.columns:
            height = pivot_data.loc[dept, activity]
            if height > 5:  # Only label segments larger than 5%
                # Get duration info
                info = duration_info.get((dept, activity), {})
                formatted_time = info.get('formatted_time', '')
                
                # Add label with percentage and time
                ax.text(i, cumulative_height + height/2, 
                       f"{height:.1f}%\n{formatted_time}", 
                       ha='center', va='center', fontsize=12, fontweight='bold')
            cumulative_height += height
    
    ax.set_title(get_text('Activity Distribution by Department', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Department', language), fontsize=14)
    ax.set_ylabel(get_text('Percentage of Time (%)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_shift_comparison(data, save_path=None, figsize=(14, 8), language='en'):
    """
    Compare activity breakdown between shifts
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
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
    
    # No filtering for isPauseData
    filtered_data = data
    
    # Get colors
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Aggregate data by shift and activity
    shift_activity = filtered_data.groupby(['shift', 'activity'])['duration'].sum().reset_index()
    
    # Calculate total by shift for percentages
    shift_total = filtered_data.groupby('shift')['duration'].sum().reset_index()
    shift_total = shift_total.rename(columns={'duration': 'shift_total'})
    
    # Merge to get shift totals
    shift_activity = pd.merge(shift_activity, shift_total, on='shift')
    
    # Calculate percentage
    shift_activity['percentage'] = (shift_activity['duration'] / shift_activity['shift_total'] * 100).round(1)
    
    # Create a table to store duration information for labeling
    duration_info = {}
    for _, row in shift_activity.iterrows():
        key = (row['shift'], row['activity'])
        duration_info[key] = {
            'duration': row['duration'],
            'percentage': row['percentage'],
            'formatted_time': format_seconds_to_hms(row['duration'])
        }
    
    # Pivot data for plotting
    pivot_data = shift_activity.pivot_table(
        index='shift',
        columns='activity',
        values='percentage',
        fill_value=0
    )
    
    # Reorder columns based on standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in ordered_cols]
    ordered_cols.extend(other_cols)
    
    pivot_data = pivot_data[ordered_cols]
    
    # Get colors for the stacked bars - using the defined activity colors
    bar_colors = [activity_colors.get(act, '#CCCCCC') for act in pivot_data.columns]
    
    # Translate activity names for legend
    translated_activities = [get_text(act, language) for act in pivot_data.columns]
    
    # Plot stacked bars
    ax = pivot_data.plot(kind='bar', stacked=True, color=bar_colors, ax=ax)
    
    # Update legend with translated labels - horizontal placement at bottom
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, translated_activities, title=get_text('Activity', language),
             loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(pivot_data.columns))
    
    # Add percentage and duration labels on segments
    for i, shift in enumerate(pivot_data.index):
        cumulative_height = 0
        for activity in pivot_data.columns:
            height = pivot_data.loc[shift, activity]
            if height > 5:  # Only label segments larger than 5%
                # Get duration info
                info = duration_info.get((shift, activity), {})
                formatted_time = info.get('formatted_time', '')
                
                # Add label with percentage and time
                ax.text(i, cumulative_height + height/2, 
                       f"{height:.1f}%\n{formatted_time}", 
                       ha='center', va='center', fontsize=12, fontweight='bold')
            cumulative_height += height
    
    ax.set_title(get_text('Activity Distribution by Shift', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Shift', language), fontsize=14)
    ax.set_ylabel(get_text('Percentage of Time (%)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_employee_activities_by_shift(data, employee_id=None, save_path=None, figsize=(15, 8), language='en'):
    """
    Plot activities for an employee (or all employees) by shift
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    employee_id : str, optional
        Employee ID to focus on (if None, show all employees)
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
    
    # No filtering for isPauseData
    filtered_data = data
    
    # Filter by employee if specified
    if employee_id:
        filtered_data = filtered_data[filtered_data['id'] == employee_id]
    
    # Get colors
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Aggregate data by shift and activity
    shift_activity = filtered_data.groupby(['shift', 'activity'])['duration'].sum().reset_index()
    
    # Convert seconds to hours
    shift_activity['hours'] = shift_activity['duration'] / 3600
    
    # Create a table to store duration information for labeling
    duration_info = {}
    for _, row in shift_activity.iterrows():
        key = (row['shift'], row['activity'])
        duration_info[key] = {
            'duration': row['duration'],
            'hours': row['hours'],
            'formatted_time': format_seconds_to_hms(row['duration'])
        }
    
    # Pivot data for plotting
    pivot_data = shift_activity.pivot_table(
        index='shift',
        columns='activity',
        values='hours',
        fill_value=0
    )
    
    # Sort by shift
    pivot_data = pivot_data.sort_index()
    
    # Reorder columns based on standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in ordered_cols]
    ordered_cols.extend(other_cols)
    
    # Make sure all columns exist (or create empty ones)
    for col in ordered_cols:
        if col not in pivot_data.columns:
            pivot_data[col] = 0
    
    # Select only the columns that exist
    existing_cols = [col for col in ordered_cols if col in pivot_data.columns]
    pivot_data = pivot_data[existing_cols]
    
    # Get colors for the stacked bars - using the defined activity colors
    bar_colors = [activity_colors.get(act, '#CCCCCC') for act in pivot_data.columns]
    
    # Translate activity names for legend
    translated_activities = [get_text(act, language) for act in pivot_data.columns]
    
    # Plot stacked bars
    ax = pivot_data.plot(kind='bar', stacked=True, color=bar_colors, ax=ax)
    
    # Update legend with translated labels - horizontal placement at bottom
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, translated_activities, title=get_text('Activity', language),
             loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(pivot_data.columns))
    
    # Calculate total hours and percentages for each shift
    shift_totals = pivot_data.sum(axis=1)
    overall_total = shift_totals.sum()
    
    # Add percentage and duration labels on the bars
    for i, shift in enumerate(pivot_data.index):
        shift_total = shift_totals[shift]
        shift_pct = (shift_total / overall_total * 100).round(1) if overall_total > 0 else 0
        cumulative_height = 0
        
        # Add total percentage at the top
        ax.text(i, shift_total + 0.5, 
               f"{shift_pct}%\n{format_seconds_to_hms(shift_total*3600)}", 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add activity-specific labels
        for activity in pivot_data.columns:
            height = pivot_data.loc[shift, activity]
            if height > 0.5:  # Only label segments larger than 0.5 hours
                # Calculate percentage of this activity within this shift
                activity_pct = (height / shift_total * 100).round(1) if shift_total > 0 else 0
                
                # Get formatted time
                formatted_time = format_seconds_to_hms(height * 3600)
                
                # Add label with percentage and time
                ax.text(i, cumulative_height + height/2, 
                       f"{activity_pct:.1f}%\n{formatted_time}", 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            cumulative_height += height
    
    # Set title based on whether filtering for a specific employee
    if employee_id:
        title = get_text('Activities by Shift - Employee {0}', language).format(employee_id)
    else:
        title = get_text('Activities by Shift - All Employees', language)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Shift', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hours)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_region_activity_heatmap(data, top_n_regions=10, save_path=None, figsize=(14, 10), language='en'):
    """
    Create a heatmap showing activities across top regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    top_n_regions : int, optional
        Number of top regions to include
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
    
    # No filtering for isPauseData or shift
    filtered_data = data
    
    # Get standard activity order
    activity_order = get_activity_order()
    
    # Find top regions by total duration
    region_total = filtered_data.groupby('region')['duration'].sum().reset_index()
    region_total = region_total.sort_values('duration', ascending=False)
    top_regions = region_total.head(top_n_regions)['region'].tolist()
    
    # Filter data to top regions
    region_data = filtered_data[filtered_data['region'].isin(top_regions)]
    
    # Aggregate data by region and activity
    region_activity = region_data.groupby(['region', 'activity'])['duration'].sum().reset_index()
    
    # Calculate total by region for percentages
    region_total = region_data.groupby('region')['duration'].sum().reset_index()
    region_total = region_total.rename(columns={'duration': 'region_total'})
    
    # Merge to get region totals
    region_activity = pd.merge(region_activity, region_total, on='region')
    
    # Calculate percentage
    region_activity['percentage'] = (region_activity['duration'] / region_activity['region_total'] * 100).round(1)
    region_activity['formatted_time'] = region_activity['duration'].apply(format_seconds_to_hms)
    
    # Store duration and percentage info for annotations
    annotation_data = {}
    for _, row in region_activity.iterrows():
        key = (row['region'], row['activity'])
        annotation_data[key] = {
            'percentage': row['percentage'],
            'formatted_time': row['formatted_time']
        }
    
    # Pivot data for heatmap
    pivot_data = region_activity.pivot_table(
        index='region',
        columns='activity',
        values='percentage',
        fill_value=0
    )
    
    # Reorder columns based on standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in ordered_cols]
    ordered_cols.extend(other_cols)
    
    # Select only the columns that exist
    existing_cols = [col for col in ordered_cols if col in pivot_data.columns]
    if not existing_cols:
        existing_cols = pivot_data.columns
    
    pivot_data = pivot_data[existing_cols]
    
    # Reorder rows by total duration
    pivot_data = pivot_data.reindex(top_regions)
    
    # Create a custom function for annotations
    def heatmap_annotation(val, region, activity):
        if val < 1:  # Skip very small values
            return ""
        
        info = annotation_data.get((region, activity), {})
        formatted_time = info.get('formatted_time', '')
        return f"{val:.1f}%\n{formatted_time}"
    
    # Create annotations matrix
    annotations = np.empty_like(pivot_data.values, dtype=object)
    for i, region in enumerate(pivot_data.index):
        for j, activity in enumerate(pivot_data.columns):
            val = pivot_data.iloc[i, j]
            annotations[i, j] = heatmap_annotation(val, region, activity)
    
    # Translate column names
    pivot_data.columns = [get_text(col, language) for col in pivot_data.columns]
    
    # Create heatmap with custom annotations
    sns.heatmap(
        pivot_data, 
        annot=annotations,
        fmt="",
        cmap="YlOrRd", 
        cbar_kws={'label': get_text('Percentage of Time (%)', language)},
        ax=ax
    )
    
    ax.set_title(get_text('Activity Distribution Across Top Regions', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel(get_text('Region', language), fontsize=14)
    ax.set_xlabel(get_text('Activity', language), fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_activity_duration_bar(data, filter_by=None, filter_value=None, save_path=None, figsize=(12, 8), language='en'):
    """
    Create horizontal bar chart of activity durations
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    filter_by : str, optional
        Column to filter by (e.g., 'id', 'department', 'shift')
    filter_value : str, optional
        Value to filter for
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
    
    # No filtering for isPauseData or shift
    filtered_data = data
    
    # Apply additional filter if specified
    if filter_by and filter_value:
        filtered_data = filtered_data[filtered_data[filter_by] == filter_value]
    
    # Get colors and order
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Aggregate data by activity
    activity_data = filtered_data.groupby('activity')['duration'].sum().reset_index()
    
    # Calculate percentage
    total_duration = activity_data['duration'].sum()
    activity_data['percentage'] = (activity_data['duration'] / total_duration * 100).round(1)
    
    # Format duration
    activity_data['formatted_time'] = activity_data['duration'].apply(format_seconds_to_hms)
    activity_data['hours'] = activity_data['duration'] / 3600
    
    # Reorder by standard activity order
    ordered_activities = []
    for activity in activity_order:
        matching = activity_data[activity_data['activity'] == activity]
        if not matching.empty:
            ordered_activities.append(matching.iloc[0])
    
    # Add any remaining activities
    for _, row in activity_data.iterrows():
        if row['activity'] not in activity_order:
            ordered_activities.append(row)
    
    # Convert to DataFrame
    if ordered_activities:
        ordered_df = pd.DataFrame(ordered_activities)
    else:
        return None  # No data to plot
    
    # Prepare data for horizontal bar chart
    labels = [get_text(act, language) for act in ordered_df['activity']]
    values = ordered_df['hours']
    colors = [activity_colors.get(act, '#CCCCCC') for act in ordered_df['activity']]
    
    # Create horizontal bar chart
    bars = ax.barh(labels, values, color=colors)
    
    # Add percentage and time labels
    for bar, percentage, formatted_time in zip(bars, ordered_df['percentage'], ordered_df['formatted_time']):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
               f"{percentage:.1f}%\n{formatted_time}", 
               va='center', fontsize=12, fontweight='bold')
    
    # Set title based on filter
    if filter_by and filter_value:
        filter_name = get_text(filter_by.capitalize(), language)
        filter_text = get_text(filter_value, language) if filter_by == 'department' else filter_value
        title = get_text('Activity Duration - {0}: {1}', language).format(filter_name, filter_text)
    else:
        title = get_text('Activity Duration - All Data', language)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Duration (hours)', language), fontsize=14)
    ax.set_ylabel(get_text('Activity', language), fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig