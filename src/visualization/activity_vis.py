"""
Activity Visualization Module

Functions for creating visualizations of employee activities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from ..utils.time_utils import format_seconds_to_hms, format_seconds_to_hours
from .base import get_color_palette, get_bakery_cmap, save_figure, set_visualization_style

def plot_activity_distribution(data, save_path=None, figsize=(12, 7)):
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
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert seconds to hours for better readability
    hours = data['duration'] / 3600
    
    # Create bar chart with custom colors
    bars = ax.bar(data['activity'], hours, color=get_color_palette(len(data)))
    
    # Add percentage and time labels
    for bar, perc, time in zip(bars, data['percentage'], data['formatted_time']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{perc:.1f}%\n({time})', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Time Distribution by Activity', fontsize=16, fontweight='bold')
    ax.set_xlabel('Activity', fontsize=14)
    ax.set_ylabel('Duration (hours)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12, rotation=30, ha='right')
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_activity_distribution_by_employee(activity_profile, save_path=None, figsize=(14, 8)):
    """
    Plot activity distribution by employee as a heatmap
    
    Parameters:
    -----------
    activity_profile : pandas.DataFrame
        Pivot table with employees as index and activities as columns
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with custom colormap
    sns.heatmap(
        activity_profile, 
        cmap=get_bakery_cmap(), 
        annot=True, 
        fmt='.1f', 
        linewidths=0.5,
        ax=ax,
        cbar_kws={
            'label': 'Percentage of Employee Time (%)',
            'shrink': 0.8
        }
    )
    
    ax.set_title('Activity Distribution by Employee', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Employee ID', fontsize=14)
    ax.set_xlabel('Activity', fontsize=14)
    
    # Rotate x axis labels for better readability
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_hourly_activity_patterns(hourly_activity, save_path=None, figsize=(16, 8)):
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
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create pivot table for visualization
    pivot_data = hourly_activity.pivot_table(
        index='hour',
        columns='activity',
        values='duration',
        fill_value=0
    )
    
    # Plot stacked area chart
    pivot_data.plot(
        kind='area', 
        stacked=True, 
        alpha=0.7, 
        color=get_color_palette(len(pivot_data.columns)),
        ax=ax
    )
    
    # Convert seconds to hours on y-axis
    current_yticks = ax.get_yticks()
    ax.set_yticklabels([format_seconds_to_hms(tick) for tick in current_yticks])
    
    ax.set_title('Activity Patterns by Hour of Day', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Duration (hh:mm:ss)', fontsize=14)
    ax.grid(axis='both', linestyle='--', alpha=0.7)
    plt.xticks(range(0, 24), fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(title='Activity', fontsize=10, title_fontsize=12)
    
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