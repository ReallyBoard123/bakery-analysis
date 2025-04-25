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

def plot_activity_by_day(data, save_path=None, figsize=(15, 8), language='en'):
    """
    Plot total activity duration by day
    
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
    
    # Aggregate data by date
    daily_data = data.groupby('date')['duration'].sum().reset_index()
    
    # Sort by date
    daily_data = daily_data.sort_values('date')
    
    # Convert seconds to hours
    daily_data['hours'] = daily_data['duration'] / 3600
    
    # Create bar chart
    bars = ax.bar(daily_data['date'], daily_data['hours'], color='#2D5F91')
    
    # Add duration and percentage labels
    total_seconds = daily_data['duration'].sum()
    for i, row in enumerate(daily_data.iterrows()):
        _, row_data = row
        percentage = (row_data['duration'] / total_seconds * 100).round(1)
        formatted_duration = format_seconds_to_hms(row_data['duration'])
        
        ax.text(i, row_data['hours'] + 0.5, 
               f"{percentage}%\n{formatted_duration}", 
               ha='center', va='bottom', fontsize=10)
    
    ax.set_title(get_text('Total Activity Duration by Day', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Date', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hours)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45, ha='right')
    
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
    
    # Get department colors
    department_colors = get_department_colors()
    
    # Aggregate data by employee
    employee_data = data.groupby(['id', 'department'])['duration'].sum().reset_index()
    
    # Sort by department and total duration
    employee_data = employee_data.sort_values(['department', 'duration'], ascending=[True, False])
    
    # Convert seconds to hours
    employee_data['hours'] = employee_data['duration'] / 3600
    
    # Create bar chart with department-specific colors
    colors = [department_colors.get(dept, '#CCCCCC') for dept in employee_data['department']]
    bars = ax.bar(employee_data['id'], employee_data['hours'], color=colors)
    
    # Add duration and percentage labels
    total_seconds = employee_data['duration'].sum()
    for i, row in enumerate(employee_data.iterrows()):
        _, row_data = row
        percentage = (row_data['duration'] / total_seconds * 100).round(1)
        formatted_duration = format_seconds_to_hms(row_data['duration'])
        
        ax.text(i, row_data['hours'] + 0.5, 
               f"{percentage}%\n{formatted_duration}", 
               ha='center', va='bottom', fontsize=10)
    
    # Add department legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=department_colors[dept]) 
                     for dept in sorted(employee_data['department'].unique())]
    legend_labels = [get_text(dept, language) for dept in sorted(employee_data['department'].unique())]
    ax.legend(legend_elements, legend_labels, title=get_text('Department', language))
    
    ax.set_title(get_text('Total Duration by Employee', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hours)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
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
    
    # Get colors
    department_colors = get_department_colors()
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Aggregate data by department and activity
    dept_activity = data.groupby(['department', 'activity'])['duration'].sum().reset_index()
    
    # Calculate total by department for percentages
    dept_total = data.groupby('department')['duration'].sum().reset_index()
    dept_total = dept_total.rename(columns={'duration': 'dept_total'})
    
    # Merge to get department totals
    dept_activity = pd.merge(dept_activity, dept_total, on='department')
    
    # Calculate percentage
    dept_activity['percentage'] = (dept_activity['duration'] / dept_activity['dept_total'] * 100).round(1)
    
    # Pivot data for plotting
    pivot_data = dept_activity.pivot_table(
        index='department',
        columns='activity',
        values='percentage',
        fill_value=0
    )
    
    # Reorder columns by standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in activity_order]
    ordered_cols.extend(other_cols)
    
    pivot_data = pivot_data[ordered_cols]
    
    # Get colors for the stacked bars
    bar_colors = [activity_colors.get(act, '#CCCCCC') for act in pivot_data.columns]
    
    # Translate activity names for legend
    translated_activities = [get_text(act, language) for act in pivot_data.columns]
    
    # Plot stacked bars
    pivot_data.plot(kind='bar', stacked=True, color=bar_colors, ax=ax)
    
    # Update legend with translated labels
    handles, old_labels = ax.get_legend_handles_labels()
    ax.legend(handles, translated_activities, title=get_text('Activity', language))
    
    # Translate department names on x-axis
    ax.set_xticklabels([get_text(dept, language) for dept in pivot_data.index])
    
    # Add percentage labels on segments
    for i, dept in enumerate(pivot_data.index):
        cumulative_height = 0
        for activity in pivot_data.columns:
            height = pivot_data.loc[dept, activity]
            if height > 5:  # Only label segments larger than 5%
                ax.text(i, cumulative_height + height/2, 
                       f"{height:.1f}%", 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            cumulative_height += height
    
    ax.set_title(get_text('Activity Distribution by Department', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Department', language), fontsize=14)
    ax.set_ylabel(get_text('Percentage of Time (%)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
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
    
    # Get colors
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Aggregate data by shift and activity
    shift_activity = data.groupby(['shift', 'activity'])['duration'].sum().reset_index()
    
    # Calculate total by shift for percentages
    shift_total = data.groupby('shift')['duration'].sum().reset_index()
    shift_total = shift_total.rename(columns={'duration': 'shift_total'})
    
    # Merge to get shift totals
    shift_activity = pd.merge(shift_activity, shift_total, on='shift')
    
    # Calculate percentage
    shift_activity['percentage'] = (shift_activity['duration'] / shift_activity['shift_total'] * 100).round(1)
    
    # Pivot data for plotting
    pivot_data = shift_activity.pivot_table(
        index='shift',
        columns='activity',
        values='percentage',
        fill_value=0
    )
    
    # Reorder columns by standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in activity_order]
    ordered_cols.extend(other_cols)
    
    pivot_data = pivot_data[ordered_cols]
    
    # Get colors for the stacked bars
    bar_colors = [activity_colors.get(act, '#CCCCCC') for act in pivot_data.columns]
    
    # Translate activity names for legend
    translated_activities = [get_text(act, language) for act in pivot_data.columns]
    
    # Plot stacked bars
    pivot_data.plot(kind='bar', stacked=True, color=bar_colors, ax=ax)
    
    # Update legend with translated labels
    handles, old_labels = ax.get_legend_handles_labels()
    ax.legend(handles, translated_activities, title=get_text('Activity', language))
    
    # Add percentage labels on segments
    for i, shift in enumerate(pivot_data.index):
        cumulative_height = 0
        for activity in pivot_data.columns:
            height = pivot_data.loc[shift, activity]
            if height > 5:  # Only label segments larger than 5%
                ax.text(i, cumulative_height + height/2, 
                       f"{height:.1f}%", 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            cumulative_height += height
    
    ax.set_title(get_text('Activity Distribution by Shift', language), fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Shift', language), fontsize=14)
    ax.set_ylabel(get_text('Percentage of Time (%)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_employee_activities_by_date(data, employee_id=None, save_path=None, figsize=(15, 8), language='en'):
    """
    Plot activities for an employee (or all employees) by date
    
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
    
    # Filter by employee if specified
    if employee_id:
        filtered_data = data[data['id'] == employee_id]
    else:
        filtered_data = data
    
    # Get colors
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Aggregate data by date and activity
    date_activity = filtered_data.groupby(['date', 'activity'])['duration'].sum().reset_index()
    
    # Convert seconds to hours
    date_activity['hours'] = date_activity['duration'] / 3600
    
    # Pivot data for plotting
    pivot_data = date_activity.pivot_table(
        index='date',
        columns='activity',
        values='hours',
        fill_value=0
    )
    
    # Sort by date
    pivot_data = pivot_data.sort_index()
    
    # Reorder columns by standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in activity_order]
    ordered_cols.extend(other_cols)
    
    # Make sure all columns exist (or create empty ones)
    for col in ordered_cols:
        if col not in pivot_data.columns:
            pivot_data[col] = 0
    
    # Select only the columns that exist
    existing_cols = [col for col in ordered_cols if col in pivot_data.columns]
    pivot_data = pivot_data[existing_cols]
    
    # Get colors for the stacked bars
    bar_colors = [activity_colors.get(act, '#CCCCCC') for act in pivot_data.columns]
    
    # Translate activity names for legend
    translated_activities = [get_text(act, language) for act in pivot_data.columns]
    
    # Plot stacked bars
    pivot_data.plot(kind='bar', stacked=True, color=bar_colors, ax=ax)
    
    # Update legend with translated labels
    handles, old_labels = ax.get_legend_handles_labels()
    ax.legend(handles, translated_activities, title=get_text('Activity', language))
    
    # Set title based on whether filtering for a specific employee
    if employee_id:
        title = get_text('Daily Activities - Employee {0}', language).format(employee_id)
    else:
        title = get_text('Daily Activities - All Employees', language)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(get_text('Date', language), fontsize=14)
    ax.set_ylabel(get_text('Duration (hours)', language), fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_activity_breakdown_pie(data, filter_by=None, filter_value=None, save_path=None, figsize=(10, 8), language='en'):
    """
    Create pie chart of activity breakdown
    
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
    
    # Filter data if specified
    if filter_by and filter_value:
        filtered_data = data[data[filter_by] == filter_value]
    else:
        filtered_data = data
    
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
    
    # Prepare data for pie chart
    labels = [get_text(act, language) for act in ordered_df['activity']]
    sizes = ordered_df['percentage']
    colors = [activity_colors.get(act, '#CCCCCC') for act in ordered_df['activity']]
    
    # Add formatted time and percentage to labels
    labels_with_info = [f"{label} ({time}, {pct:.1f}%)" 
                      for label, time, pct in zip(labels, ordered_df['formatted_time'], sizes)]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,  # No labels on the pie itself
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Improve the appearance of percentage text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # Add legend with custom labels
    ax.legend(wedges, labels_with_info, title=get_text('Activity', language), 
             loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Set title based on filter
    if filter_by and filter_value:
        filter_name = get_text(filter_by.capitalize(), language)
        filter_text = get_text(filter_value, language) if filter_by == 'department' else filter_value
        title = get_text('Activity Breakdown - {0}: {1}', language).format(filter_name, filter_text)
    else:
        title = get_text('Activity Breakdown - All Data', language)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')
    
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
    
    # Get colors and order
    activity_order = get_activity_order()
    
    # Find top regions by total duration
    region_total = data.groupby('region')['duration'].sum().reset_index()
    region_total = region_total.sort_values('duration', ascending=False)
    top_regions = region_total.head(top_n_regions)['region'].tolist()
    
    # Filter data to top regions
    filtered_data = data[data['region'].isin(top_regions)]
    
    # Aggregate data by region and activity
    region_activity = filtered_data.groupby(['region', 'activity'])['duration'].sum().reset_index()
    
    # Calculate total by region for percentages
    region_total = filtered_data.groupby('region')['duration'].sum().reset_index()
    region_total = region_total.rename(columns={'duration': 'region_total'})
    
    # Merge to get region totals
    region_activity = pd.merge(region_activity, region_total, on='region')
    
    # Calculate percentage
    region_activity['percentage'] = (region_activity['duration'] / region_activity['region_total'] * 100).round(1)
    
    # Pivot data for heatmap
    pivot_data = region_activity.pivot_table(
        index='region',
        columns='activity',
        values='percentage',
        fill_value=0
    )
    
    # Reorder columns by standard activity order
    ordered_cols = [col for col in activity_order if col in pivot_data.columns]
    other_cols = [col for col in pivot_data.columns if col not in activity_order]
    ordered_cols.extend(other_cols)
    
    # Make sure all columns exist (or create empty ones)
    for col in ordered_cols:
        if col not in pivot_data.columns:
            pivot_data[col] = 0
    
    # Select only the columns that exist
    existing_cols = [col for col in ordered_cols if col in pivot_data.columns]
    pivot_data = pivot_data[existing_cols]
    
    # Reorder rows by total duration
    pivot_data = pivot_data.reindex(top_regions)
    
    # Translate column names
    pivot_data.columns = [get_text(col, language) for col in pivot_data.columns]
    
    # Create heatmap
    sns.heatmap(
        pivot_data, 
        annot=True, 
        fmt=".1f", 
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