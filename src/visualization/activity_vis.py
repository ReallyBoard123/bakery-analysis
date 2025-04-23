"""
Activity Visualization Module

Functions for creating visualizations of employee activities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

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