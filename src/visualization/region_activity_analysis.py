"""
Region Activity Analysis Module

Analyzes and visualizes employee activities within their top regions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.gridspec as gridspec

from .base import save_figure, set_visualization_style, get_color_palette
from ..utils.time_utils import format_seconds_to_hms

def analyze_activities_by_region(data, department=None, top_n_regions=5, save_path=None, figsize=(16, 12)):
    """
    Create a visualization showing top regions and activity breakdowns for employees in a department
    
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
    
    # Define the order of activities for consistent plotting
    activity_order = ['Handle up', 'Handle center', 'Handle down', 'Walk', 'Stand']
    
    # Define activity colors - using more attractive palette
    activity_colors = {
        'Handle up': '#FF6B6B',    # Red
        'Handle center': '#4ECDC4', # Teal
        'Handle down': '#FFD166',  # Yellow
        'Walk': '#06D6A0',         # Green
        'Stand': '#F7B267'         # Orange
    }
    
    # Get employees in this department
    employees = sorted(dept_data['id'].unique())
    
    if len(employees) == 0:
        print(f"No employees found for department: {department}")
        return None
    
    # Create figure
    fig, axes = plt.subplots(nrows=len(employees), ncols=top_n_regions, figsize=figsize, sharey=True)
    
    # Add a title to the figure
    dept_title = f"{department} Department" if department else "All Departments"
    fig.suptitle(f"Activity Analysis by Region - {dept_title}", fontsize=20, fontweight='bold', y=0.98)
    
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
        
        # Employee label at the left side
        if top_n_regions > 0:
            axes[emp_idx, 0].text(-0.5, 0.5, f"Employee {employee_id}\n{total_hours:.1f} hours", 
                                 ha='center', va='center', fontsize=12, fontweight='bold',
                                 transform=axes[emp_idx, 0].transAxes)
        
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
            for activity in activity_order:
                activity_data = region_data[region_data['activity'] == activity]
                activity_durations[activity] = activity_data['duration'].sum() / 3600  # Convert to hours
            
            # Get current axis
            ax = axes[emp_idx, reg_idx]
            
            # Create a bar plot for activities
            activities = []
            hours = []
            colors = []
            
            for activity in activity_order:
                if activity_durations[activity] > 0:
                    activities.append(activity)
                    hours.append(activity_durations[activity])
                    colors.append(activity_colors[activity])
            
            # Create bar plot with activities - showing hours
            bars = ax.bar(activities, hours, color=colors)
            
            # Add percentage labels on top of bars
            for bar, activity in zip(bars, activities):
                height = bar.get_height()
                percentage = (height / region_hours * 100).round(1) if region_hours > 0 else 0
                if percentage >= 5:  # Only show labels for significant percentages
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Set title with region name and percentage of total time
            ax.set_title(f'{region}\n{region_hours:.1f}h ({region_pct:.1f}% of total)', fontsize=10)
            
            # Format x-axis labels
            ax.set_xticklabels(activities, rotation=45, ha='right', fontsize=8)
            
            # Only show y-axis label for first column
            if reg_idx == 0:
                ax.set_ylabel('Hours')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
    
    # Add one legend for the entire figure
    handles = [plt.Rectangle((0, 0), 1, 1, color=activity_colors[activity]) for activity in activity_order]
    fig.legend(handles, activity_order, loc='lower center', ncol=len(activity_order), 
              bbox_to_anchor=(0.5, 0.01), fontsize=10)
    
    # Add extra space at the bottom for the legend
    plt.subplots_adjust(bottom=0.08)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig