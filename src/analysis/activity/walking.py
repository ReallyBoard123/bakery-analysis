"""
Walking Pattern Analysis Module

Provides functions for analyzing walking patterns of bakery employees across the 24-hour timeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ...utils.time_utils import format_seconds_to_hms
from ...visualization.base import set_visualization_style, get_employee_colors, save_figure, get_text

def analyze_walking_patterns(data, output_dir=None, time_slot_minutes=30, language='en'):
    """
    Analyze walking patterns of bakery employees across the 24-hour timeline
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with tracking data
    output_dir : str or Path, optional
        Directory to save output files
    time_slot_minutes : int, optional
        Size of time slots in minutes (default: 30)
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary containing results of various analyses
    """
    # Filter walking data
    walking_data = data[data['activity'] == 'Walk'].copy()
    
    if walking_data.empty:
        print(f"No walking data found in the dataset.")
        return {}
    
    # Ensure output directory exists
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Dictionary to store results
    results = {}
    
    # Create density plots of walking patterns over 24 hours
    results['timeline_analysis'] = analyze_walking_timeline(
        walking_data, output_dir, time_slot_minutes, language
    )
    
    # Add overall walking statistics for completeness
    results['overall_stats'] = analyze_overall_walking(walking_data, output_dir, language)
    
    return results

def analyze_walking_timeline(walking_data, output_dir=None, time_slot_minutes=30, language='en'):
    """
    Create density plots showing walking patterns across the 24-hour timeline
    
    Parameters:
    -----------
    walking_data : pandas.DataFrame
        Filtered dataframe with just walking data
    output_dir : Path, optional
        Directory to save output files
    time_slot_minutes : int, optional
        Size of time slots in minutes
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with timeline analysis results
    """
    # Extract hour of day from startTime (assuming it's in seconds from midnight)
    walking_data['hour_of_day'] = walking_data['startTime'] / 3600  # Convert to hours
    
    # Calculate number of slots in a day based on requested time_slot_minutes
    slots_per_day = 24 * 60 // time_slot_minutes
    
    # Create time slot index for each record (0 to slots_per_day-1)
    walking_data['time_slot_idx'] = (walking_data['hour_of_day'] * 60 / time_slot_minutes).astype(int) % slots_per_day
    
    # Create human-readable time slot labels
    time_slot_labels = []
    for i in range(slots_per_day):
        minutes_from_midnight = i * time_slot_minutes
        hours = minutes_from_midnight // 60
        minutes = minutes_from_midnight % 60
        time_slot_labels.append(f"{hours:02d}:{minutes:02d}")
    
    # Add time slot label column
    walking_data['time_slot'] = walking_data['time_slot_idx'].apply(lambda x: time_slot_labels[x])
    
    # Calculate walking duration and count by employee and time slot
    time_slot_data = walking_data.groupby(['id', 'time_slot_idx']).agg({
        'duration': ['sum', 'count'],
    }).reset_index()
    
    # Flatten multi-index columns
    time_slot_data.columns = ['id', 'time_slot_idx', 'total_duration', 'walk_count']
    
    # Add human-readable time slot
    time_slot_data['time_slot'] = time_slot_data['time_slot_idx'].apply(lambda x: time_slot_labels[x])
    
    # Convert to minutes for better visualization
    time_slot_data['duration_minutes'] = time_slot_data['total_duration'] / 60
    
    # Create directory for timeline plots
    if output_dir:
        timeline_dir = output_dir / 'timeline'
        timeline_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate a combined plot with all employees
        create_combined_density_plot(
            time_slot_data, 
            slots_per_day, 
            time_slot_labels,
            timeline_dir,
            time_slot_minutes,
            language
        )
        
        # Generate individual plots for each employee
        for emp_id in walking_data['id'].unique():
            create_employee_density_plot(
                time_slot_data,
                emp_id,
                slots_per_day,
                time_slot_labels,
                timeline_dir,
                time_slot_minutes,
                language
            )
    
    return {
        'time_slot_data': time_slot_data,
        'time_slot_labels': time_slot_labels,
        'slots_per_day': slots_per_day
    }

def create_combined_density_plot(time_slot_data, slots_per_day, time_slot_labels, 
                               output_dir, time_slot_minutes, language='en'):
    """
    Create a combined density plot showing walking patterns for all employees
    
    Parameters:
    -----------
    time_slot_data : pandas.DataFrame
        Data with walking stats by employee and time slot
    slots_per_day : int
        Number of time slots in a day
    time_slot_labels : list
        List of human-readable time slot labels
    output_dir : Path
        Directory to save the plot
    time_slot_minutes : int
        Size of time slots in minutes
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # For each employee, plot their walking pattern across time slots
    for emp_id in time_slot_data['id'].unique():
        # Filter data for this employee
        emp_data = time_slot_data[time_slot_data['id'] == emp_id]
        
        # Create full time series with all slots
        full_slots = pd.DataFrame({
            'time_slot_idx': range(slots_per_day),
            'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
        })
        
        # Merge with employee data to ensure all time slots exist
        emp_full_data = pd.merge(full_slots, emp_data, on=['time_slot_idx', 'time_slot'], how='left')
        
        # Fill NAs with zeros
        emp_full_data = emp_full_data.fillna(0)
        
        # Sort by time slot
        emp_full_data = emp_full_data.sort_values('time_slot_idx')
        
        # Get color for this employee
        color = employee_colors.get(emp_id, None)  # Use default if not found
        
        # Create the walking pattern line
        plt.plot(
            emp_full_data['time_slot_idx'],
            emp_full_data['duration_minutes'],
            '-', 
            label=f"Employee {emp_id}", 
            linewidth=2.5,
            color=color
        )
    
    # Improve x-axis with better tick labels
    if slots_per_day <= 48:  # For 30-min or longer slots, label every slot
        tick_every = 1
    else:  # For shorter slots, label less frequently
        tick_every = slots_per_day // 24  # Approximately hourly
    
    plt.xticks(
        range(0, slots_per_day, tick_every),
        [time_slot_labels[i] for i in range(0, slots_per_day, tick_every)],
        rotation=45
    )
    
    # Set labels and title
    slot_text = f"{time_slot_minutes}-minute" if time_slot_minutes != 60 else "hourly"
    plt.title(f"Walking Patterns Throughout the Day ({slot_text} intervals)", fontsize=16, fontweight='bold')
    plt.xlabel("Time of Day", fontsize=14)
    plt.ylabel("Minutes Spent Walking", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / f"all_employees_walking_patterns_{time_slot_minutes}min.png")
    plt.close()

def create_employee_density_plot(time_slot_data, employee_id, slots_per_day, 
                               time_slot_labels, output_dir, time_slot_minutes, language='en'):
    """
    Create a density plot showing walking patterns for a specific employee
    
    Parameters:
    -----------
    time_slot_data : pandas.DataFrame
        Data with walking stats by employee and time slot
    employee_id : str
        ID of the employee to analyze
    slots_per_day : int
        Number of time slots in a day
    time_slot_labels : list
        List of human-readable time slot labels
    output_dir : Path
        Directory to save the plot
    time_slot_minutes : int
        Size of time slots in minutes
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Get employee colors
    employee_colors = get_employee_colors()
    emp_color = employee_colors.get(employee_id, '#1f77b4')
    
    # Filter data for this employee
    emp_data = time_slot_data[time_slot_data['id'] == employee_id]
    
    # Create full time series with all slots
    full_slots = pd.DataFrame({
        'time_slot_idx': range(slots_per_day),
        'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
    })
    
    # Merge with employee data to ensure all time slots exist
    emp_full_data = pd.merge(full_slots, emp_data, on=['time_slot_idx', 'time_slot'], how='left')
    
    # Fill NAs with zeros
    emp_full_data = emp_full_data.fillna(0)
    
    # Sort by time slot
    emp_full_data = emp_full_data.sort_values('time_slot_idx')
    
    # Create a more advanced plot with multiple visualizations in one
    fig, (ax_line, ax_bar) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # 1. Line plot with area fill (top subplot)
    ax_line.plot(
        emp_full_data['time_slot_idx'],
        emp_full_data['duration_minutes'],
        '-', 
        linewidth=2.5,
        color=emp_color
    )
    
    # Add area fill below the line
    ax_line.fill_between(
        emp_full_data['time_slot_idx'],
        emp_full_data['duration_minutes'],
        alpha=0.3,
        color=emp_color
    )
    
    # Add data points for non-zero values
    non_zero_data = emp_full_data[emp_full_data['duration_minutes'] > 0]
    ax_line.scatter(
        non_zero_data['time_slot_idx'],
        non_zero_data['duration_minutes'],
        s=50,
        color=emp_color,
        edgecolor='white',
        alpha=0.8
    )
    
    # Label significant points
    for _, row in non_zero_data.nlargest(5, 'duration_minutes').iterrows():
        ax_line.annotate(
            f"{row['duration_minutes']:.1f}m",
            xy=(row['time_slot_idx'], row['duration_minutes']),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black')
        )
    
    ax_line.set_title(f"Walking Pattern for Employee {employee_id} ({time_slot_minutes}-minute intervals)", 
                    fontsize=16, fontweight='bold')
    ax_line.set_ylabel("Minutes Spent Walking", fontsize=14)
    ax_line.grid(True, alpha=0.3)
    
    # 2. Bar chart for walk count (bottom subplot)
    ax_bar.bar(
        emp_full_data['time_slot_idx'],
        emp_full_data['walk_count'],
        color=emp_color,
        alpha=0.7
    )
    
    ax_bar.set_ylabel("Number of Walks", fontsize=14)
    ax_bar.set_xlabel("Time of Day", fontsize=14)
    ax_bar.grid(True, alpha=0.3)
    
    # Improve x-axis with better tick labels
    if slots_per_day <= 48:  # For 30-min or longer slots, label every slot
        tick_every = 1
    else:  # For shorter slots, label less frequently
        tick_every = slots_per_day // 24  # Approximately hourly
    
    ax_bar.set_xticks(range(0, slots_per_day, tick_every))
    ax_bar.set_xticklabels([time_slot_labels[i] for i in range(0, slots_per_day, tick_every)], rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / f"employee_{employee_id}_walking_pattern_{time_slot_minutes}min.png")
    plt.close()

def analyze_overall_walking(walking_data, output_dir=None, language='en'):
    """
    Analyze overall walking statistics by employee
    
    Parameters:
    -----------
    walking_data : pandas.DataFrame
        Filtered dataframe with just walking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    pandas.DataFrame
        Overall walking statistics
    """
    # Group by employee
    employee_stats = walking_data.groupby('id').agg({
        'duration': ['sum', 'mean', 'median', 'count', 'min', 'max']
    }).reset_index()
    
    # Flatten multi-index columns
    employee_stats.columns = ['id', 'total_duration', 'mean_duration', 'median_duration', 
                             'walk_count', 'min_duration', 'max_duration']
    
    # Convert durations to human-readable formats
    employee_stats['total_hours'] = employee_stats['total_duration'] / 3600
    employee_stats['formatted_total'] = employee_stats['total_duration'].apply(format_seconds_to_hms)
    employee_stats['formatted_mean'] = employee_stats['mean_duration'].apply(format_seconds_to_hms)
    
    # Sort by total walking time
    employee_stats = employee_stats.sort_values('total_duration', ascending=False)
    
    # Save to CSV if output_dir is provided
    if output_dir:
        employee_stats.to_csv(output_dir / 'walking_stats_by_employee.csv', index=False)
    
    # Create summary visualization
    if output_dir:
        create_walking_summary(employee_stats, output_dir, language)
    
    return employee_stats

def create_walking_summary(walking_stats, output_dir, language='en'):
    """
    Create visualization summarizing walking statistics for all employees
    
    Parameters:
    -----------
    walking_stats : pandas.DataFrame
        Walking statistics by employee
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot total walking time by employee
    bars = ax.bar(
        walking_stats['id'], 
        walking_stats['total_hours'],
        color=[employee_colors.get(emp, '#1f77b4') for emp in walking_stats['id']]
    )
    
    # Add formatted time labels
    for bar, formatted_time in zip(bars, walking_stats['formatted_total']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
              formatted_time, ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Total Walking Time by Employee", fontsize=16, fontweight='bold')
    ax.set_xlabel("Employee ID", fontsize=14)
    ax.set_ylabel("Hours", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / 'total_walking_time.png')
    plt.close()