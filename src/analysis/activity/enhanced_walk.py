"""
Enhanced Walking Pattern Analysis Module

Provides functions for analyzing walking patterns of bakery employees with focus on:
1. Dynamic time range detection rather than fixed 24-hour periods
2. Region analysis during peak walking times
3. Statistical correlation between walking patterns and regions/time
4. Detection of repetitive movement patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d
from collections import Counter, defaultdict
from scipy.stats import pearsonr
import colorsys
from datetime import datetime, timedelta

from ...utils.time_utils import format_seconds_to_hms
from ...visualization.base import set_visualization_style, get_employee_colors, save_figure, get_text


def analyze_walking_patterns(data, output_dir=None, time_slot_minutes=30, language='en'):
    """
    Analyze walking patterns of bakery employees dynamically based on actual time ranges
    
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
    
    # Determine the actual time range for each employee
    time_ranges = determine_employee_time_range(data)
    results['time_ranges'] = time_ranges
    
    # Create directory for timeline plots
    if output_dir:
        timeline_dir = output_dir / 'timeline'
        timeline_dir.mkdir(exist_ok=True, parents=True)
    
    # Create density plots of walking patterns
    results['timeline_analysis'] = analyze_walking_timeline(
        walking_data, time_ranges, output_dir, time_slot_minutes, language
    )
    
    # Add overall walking statistics
    results['overall_stats'] = analyze_overall_walking(walking_data, output_dir, language)
    
    # Analyze region patterns
    results['region_patterns'] = analyze_region_walking_patterns(data, walking_data, output_dir, language)
    
    # Detect repetitive patterns
    results['repetitive_patterns'] = detect_repetitive_walking(walking_data, output_dir, language)
    
    # Analyze walking by shift if shift information is available
    if 'shift' in walking_data.columns:
        results['shift_analysis'] = analyze_walking_by_shift(walking_data, output_dir, time_slot_minutes, language)
    
    return results


def determine_employee_time_range(data):
    """
    Determine the actual time range for each employee from the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with tracking data
    
    Returns:
    --------
    dict
        Dictionary with employee IDs as keys and tuples of (start_time, end_time) as values
    """
    time_ranges = {}
    
    for emp_id in data['id'].unique():
        emp_data = data[data['id'] == emp_id]
        
        # Find the minimum startTime and maximum endTime
        min_start = emp_data['startTime'].min()
        max_end = emp_data['endTime'].max()
        
        # Convert to hours for easier visualization
        min_start_hour = min_start / 3600
        max_end_hour = max_end / 3600
        
        # Calculate total duration in hours
        total_duration_hours = (max_end - min_start) / 3600
        
        time_ranges[emp_id] = {
            'start_time': min_start,
            'end_time': max_end,
            'start_hour': min_start_hour,
            'end_hour': max_end_hour,
            'total_hours': total_duration_hours,
            'formatted_start': format_seconds_to_hms(min_start),
            'formatted_end': format_seconds_to_hms(max_end)
        }
    
    return time_ranges


def analyze_walking_timeline(walking_data, time_ranges, output_dir=None, time_slot_minutes=30, language='en'):
    """
    Create density plots showing walking patterns across the timeline,
    focusing the visualization on the actual time periods with region information
    
    Parameters:
    -----------
    walking_data : pandas.DataFrame
        Filtered dataframe with just walking data
    time_ranges : dict
        Dictionary with employee time ranges
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
    # Create time slots based on time slot minutes
    slots_per_day = 24 * 60 // time_slot_minutes
    
    # Extract hour of day from startTime and calculate time slot index
    walking_data['hour_of_day'] = walking_data['startTime'] / 3600  # Convert to hours
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
    
    # Find active time periods for each employee (time slots with activity)
    active_periods = {}
    for emp_id in walking_data['id'].unique():
        emp_data = walking_data[walking_data['id'] == emp_id]
        active_slots = emp_data['time_slot_idx'].unique().tolist()
        if active_slots:
            active_periods[emp_id] = (min(active_slots), max(active_slots))
    
    # Calculate walking duration, count, and most visited region by employee and time slot
    time_slot_data = walking_data.groupby(['id', 'time_slot_idx']).agg({
        'duration': ['sum', 'count'],
        'region': lambda x: x.value_counts().index[0] if len(x) > 0 else None  # Most frequent region
    }).reset_index()
    
    # Flatten multi-index columns
    time_slot_data.columns = ['id', 'time_slot_idx', 'total_duration', 'walk_count', 'most_visited_region']
    
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
            active_periods,
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
                active_periods.get(emp_id, (0, slots_per_day-1)),
                slots_per_day,
                time_slot_labels,
                timeline_dir,
                time_slot_minutes,
                language
            )
            
            # Create region-focused plots for this employee
            create_region_focused_plot(
                walking_data[walking_data['id'] == emp_id],
                emp_id,
                timeline_dir,
                language
            )
    
    return {
        'time_slot_data': time_slot_data,
        'time_slot_labels': time_slot_labels,
        'slots_per_day': slots_per_day,
        'active_periods': active_periods
    }


def create_employee_density_plot(time_slot_data, employee_id, active_period, slots_per_day, 
                               time_slot_labels, output_dir, time_slot_minutes, language='en'):
    """
    Create a density plot showing walking patterns for a specific employee,
    focusing on active time periods and including region information
    
    Parameters:
    -----------
    time_slot_data : pandas.DataFrame
        Data with walking stats by employee and time slot
    employee_id : str
        ID of the employee to analyze
    active_period : tuple
        (min_slot, max_slot) indicating the active time period
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
    
    # Merge with employee data to ensure all slots exist
    emp_full_data = pd.merge(full_slots, emp_data, on=['time_slot_idx', 'time_slot'], how='left')
    
    # Fill NAs with zeros
    emp_full_data = emp_full_data.fillna(0)
    
    # Sort by time slot
    emp_full_data = emp_full_data.sort_values('time_slot_idx')
    
    # Create a more advanced plot with multiple visualizations in one
    fig, (ax_line, ax_bar, ax_region) = plt.subplots(3, 1, figsize=(16, 12), 
                                                   gridspec_kw={'height_ratios': [3, 1, 1]}, 
                                                   sharex=True)
    
    # Apply smoothing for more readable patterns
    y_values = emp_full_data['duration_minutes'].values
    smoothed_y = gaussian_filter1d(y_values, sigma=1.5)
    
    # 1. Line plot with area fill (top subplot)
    ax_line.plot(
        emp_full_data['time_slot_idx'],
        smoothed_y,
        '-', 
        linewidth=2.5,
        color=emp_color
    )
    
    # Add area fill below the line
    ax_line.fill_between(
        emp_full_data['time_slot_idx'],
        smoothed_y,
        alpha=0.3,
        color=emp_color
    )
    
    # Add data points for non-zero values
    non_zero_data = emp_full_data[emp_full_data['duration_minutes'] > 0].copy()
    non_zero_data['smoothed_value'] = [smoothed_y[idx] for idx in non_zero_data['time_slot_idx']]
    
    ax_line.scatter(
        non_zero_data['time_slot_idx'],
        non_zero_data['smoothed_value'],
        s=50,
        color=emp_color,
        edgecolor='white',
        alpha=0.8
    )
    
    # Label significant points and their regions
    for _, row in non_zero_data.nlargest(5, 'duration_minutes').iterrows():
        smoothed_value = smoothed_y[row['time_slot_idx']]
        region_text = f"\n({row['most_visited_region']})" if row['most_visited_region'] else ""
        
        ax_line.annotate(
            f"{row['duration_minutes']:.1f}m{region_text}",
            xy=(row['time_slot_idx'], smoothed_value),
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
    
    # 2. Bar chart for walk count (middle subplot)
    ax_bar.bar(
        emp_full_data['time_slot_idx'],
        emp_full_data['walk_count'],
        color=emp_color,
        alpha=0.7
    )
    
    ax_bar.set_ylabel("Number of Walks", fontsize=14)
    ax_bar.grid(True, alpha=0.3)
    
    # 3. Region information (bottom subplot)
    # Get all regions visited
    regions = []
    slot_region_pairs = []
    
    for _, row in non_zero_data.iterrows():
        if row['most_visited_region'] and row['most_visited_region'] not in regions:
            regions.append(row['most_visited_region'])
        slot_region_pairs.append((row['time_slot_idx'], row['most_visited_region']))
    
    # Create a colormap for regions
    region_colors = plt.cm.tab20(np.linspace(0, 1, len(regions))) if regions else []
    region_color_map = {region: color for region, color in zip(regions, region_colors)}
    
    # Plot regions
    for time_slot_idx, region in slot_region_pairs:
        if region and region in region_color_map:
            ax_region.bar(
                time_slot_idx, 
                1, 
                width=0.8, 
                color=region_color_map[region],
                alpha=0.8
            )
    
    # Add region legend
    if regions:
        region_handles = [plt.Rectangle((0, 0), 1, 1, color=region_color_map[region]) for region in regions]
        ax_region.legend(region_handles, regions, loc='upper right', ncol=min(4, len(regions)), 
                       fontsize=10, title='Most Visited Regions')
    
    ax_region.set_ylabel("Region", fontsize=14)
    ax_region.set_xlabel("Time of Day", fontsize=14)
    ax_region.set_yticks([])
    
    # Focus x-axis on active period with padding
    min_slot, max_slot = active_period
    padding = max(2, (max_slot - min_slot) * 0.1)  # 10% padding on each side, at least 2 slots
    padded_min = max(0, min_slot - padding)
    padded_max = min(slots_per_day - 1, max_slot + padding)
    
    ax_line.set_xlim(padded_min, padded_max)
    
    # Improve x-axis with better tick labels
    if (max_slot - min_slot) <= 24:  # For shorter active periods, show more detail
        tick_every = 1
    else:
        tick_every = max(1, (max_slot - min_slot) // 12)  # About 12 ticks
    
    tick_positions = range(int(padded_min), int(padded_max) + 1, max(1, tick_every))
    ax_region.set_xticks(tick_positions)
    ax_region.set_xticklabels([time_slot_labels[i % len(time_slot_labels)] for i in tick_positions], rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / f"employee_{employee_id}_walking_pattern_{time_slot_minutes}min.png")
    plt.close()


def create_combined_density_plot(time_slot_data, active_periods, slots_per_day, time_slot_labels, 
                               output_dir, time_slot_minutes, language='en'):
    """
    Create a combined density plot showing walking patterns for all employees,
    focusing on the active time periods with region information at peak points
    
    Parameters:
    -----------
    time_slot_data : pandas.DataFrame
        Data with walking stats by employee and time slot
    active_periods : dict
        Dictionary with employee IDs as keys and (min_slot, max_slot) as values
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
    
    # Find global active period across all employees
    if active_periods:
        global_min_slot = min([period[0] for period in active_periods.values()])
        global_max_slot = max([period[1] for period in active_periods.values()])
        padding = max(2, (global_max_slot - global_min_slot) * 0.05)  # 5% padding, at least 2 slots
        global_min_with_padding = max(0, global_min_slot - padding)
        global_max_with_padding = min(slots_per_day - 1, global_max_slot + padding)
    else:
        global_min_with_padding = 0
        global_max_with_padding = slots_per_day - 1
    
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
        
        # Apply smoothing for more readable patterns
        y_values = emp_full_data['duration_minutes'].values
        smoothed_y = gaussian_filter1d(y_values, sigma=1.5)
        
        # Get color for this employee
        color = employee_colors.get(emp_id, None)  # Use default if not found
        
        # Create the walking pattern line with smoothed data
        plt.plot(
            emp_full_data['time_slot_idx'],
            smoothed_y,
            '-', 
            label=f"Employee {emp_id}", 
            linewidth=2.5,
            color=color
        )
        
        # Add markers for peak walking times with region information
        peak_indices = find_peaks(smoothed_y)  # Get indices of local peaks
        
        # Filter to top 3 peaks
        if len(peak_indices) > 3:
            peak_values = [smoothed_y[idx] for idx in peak_indices]
            sorted_peaks = sorted(zip(peak_indices, peak_values), key=lambda x: x[1], reverse=True)
            peak_indices = [idx for idx, _ in sorted_peaks[:3]]
        
        for peak_idx in peak_indices:
            if smoothed_y[peak_idx] > 0.1:  # Only mark significant peaks
                time_slot = emp_full_data.iloc[peak_idx]['time_slot_idx']
                original_data = emp_data[emp_data['time_slot_idx'] == time_slot]
                
                if not original_data.empty:
                    region = original_data.iloc[0]['most_visited_region']
                    region_text = f"\n({region})" if region else ""
                    
                    plt.scatter(
                        time_slot,
                        smoothed_y[peak_idx],
                        s=100,
                        color=color,
                        edgecolor='white',
                        zorder=10
                    )
                    
                    plt.annotate(
                        f"{smoothed_y[peak_idx]:.1f}m{region_text}",
                        xy=(time_slot, smoothed_y[peak_idx]),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontweight='bold',
                        color=color
                    )
    
    # Focus x-axis on the global active period
    plt.xlim(global_min_with_padding, global_max_with_padding)
    
    # Improve x-axis with better tick labels
    span = global_max_with_padding - global_min_with_padding
    if span <= 24:  # For shorter spans, show more detail
        tick_every = 1
    else:
        tick_every = max(1, int(span / 12))  # About 12 ticks
    
    tick_positions = range(int(global_min_with_padding), int(global_max_with_padding) + 1, max(1, tick_every))
    plt.xticks(
        tick_positions,
        [time_slot_labels[i % len(time_slot_labels)] for i in tick_positions],
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


def find_peaks(y_values, min_height=0.1, min_distance=3):
    """
    Find peaks in a 1D array
    
    Parameters:
    -----------
    y_values : array-like
        Array of values to find peaks in
    min_height : float
        Minimum height to be considered a peak
    min_distance : int
        Minimum samples between peaks
    
    Returns:
    --------
    list
        List of peak indices
    """
    peaks = []
    
    # Simple peak detection algorithm
    for i in range(1, len(y_values) - 1):
        if (y_values[i] > y_values[i-1] and y_values[i] > y_values[i+1] and y_values[i] >= min_height):
            # Check if it's far enough from existing peaks
            if not peaks or min(abs(i - p) for p in peaks) >= min_distance:
                peaks.append(i)
    
    return peaks


def create_region_focused_plot(emp_walking_data, employee_id, output_dir, language='en'):
    """
    Create a visualization focused on regions where the employee walks the most
    
    Parameters:
    -----------
    emp_walking_data : pandas.DataFrame
        Walking data for a specific employee
    employee_id : str
        ID of the employee
    output_dir : Path
        Directory to save the plot
    language : str, optional
        Language code ('en' or 'de')
    """
    if emp_walking_data.empty:
        return
    
    set_visualization_style()
    
    # Get employee color
    employee_colors = get_employee_colors()
    emp_color = employee_colors.get(employee_id, '#1f77b4')
    
    # Calculate walking time by region
    region_walking = emp_walking_data.groupby('region')['duration'].agg(['sum', 'count']).reset_index()
    region_walking.columns = ['region', 'total_duration', 'walk_count']
    region_walking['duration_minutes'] = region_walking['total_duration'] / 60
    region_walking['walk_freq'] = (region_walking['duration_minutes'] / region_walking['walk_count']).fillna(0)
    
    # Sort by total duration
    region_walking = region_walking.sort_values('total_duration', ascending=False)
    
    # Get top regions (up to 10)
    top_regions = region_walking.head(10)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Plot total duration by region
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(top_regions)))
    
    bars = ax1.barh(
        top_regions['region'],
        top_regions['duration_minutes'],
        color=colors
    )
    
    # Add labels
    for bar, duration, count in zip(bars, top_regions['duration_minutes'], top_regions['walk_count']):
        ax1.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{duration:.1f}m ({count} walks)",
            va='center',
            fontweight='bold'
        )
    
    ax1.set_title(f"Top Walking Regions for Employee {employee_id}", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Minutes Spent Walking", fontsize=14)
    ax1.set_ylabel("Region", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Plot average walk duration by region
    bars2 = ax2.barh(
        top_regions['region'],
        top_regions['walk_freq'],
        color=colors,
        alpha=0.8
    )
    
    # Add labels
    for bar, freq in zip(bars2, top_regions['walk_freq']):
        ax2.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height()/2,
            f"{freq:.1f}m/walk",
            va='center'
        )
    
    ax2.set_title(f"Average Walking Duration per Visit", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Minutes per Walk", fontsize=14)
    ax2.set_ylabel("Region", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / f"employee_{employee_id}_region_focused.png")
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
    employee_stats['avg_duration_seconds'] = employee_stats['mean_duration'].round(1)
    
    # Sort by total walking time
    employee_stats = employee_stats.sort_values('total_duration', ascending=False)
    
    # Save to CSV if output_dir is provided
    if output_dir:
        employee_stats.to_csv(output_dir / 'walking_stats_by_employee.csv', index=False)
    
    # Create summary visualization
    if output_dir:
        create_walking_summary(employee_stats, output_dir, language)
        
        # Create comparative visualization
        create_employee_walking_comparison(employee_stats, output_dir, language)
    
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
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Plot total walking time by employee (top subplot)
    bars = ax1.bar(
        walking_stats['id'], 
        walking_stats['total_hours'],
        color=[employee_colors.get(emp, '#1f77b4') for emp in walking_stats['id']]
    )
    
    # Add formatted time labels
    for bar, formatted_time, hours in zip(bars, 
                                         walking_stats['formatted_total'], 
                                         walking_stats['total_hours']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
              f"{formatted_time}\n({hours:.1f} hrs)",
              ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title("Total Walking Time by Employee", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Hours", fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot average walk duration (bottom subplot)
    bars2 = ax2.bar(
        walking_stats['id'], 
        walking_stats['avg_duration_seconds'],
        color=[employee_colors.get(emp, '#1f77b4') for emp in walking_stats['id']],
        alpha=0.8
    )
    
    # Add walk count labels
    for bar, avg_duration, count in zip(bars2, 
                                      walking_stats['avg_duration_seconds'], 
                                      walking_stats['walk_count']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
              f"{avg_duration:.1f}s\n({count} walks)",
              ha='center', va='bottom')
    
    ax2.set_title("Average Walk Duration by Employee", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Employee ID", fontsize=14)
    ax2.set_ylabel("Seconds", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / 'total_walking_time.png')
    plt.close()


def create_employee_walking_comparison(walking_stats, output_dir, language='en'):
    """
    Create a comprehensive comparison of walking patterns across employees
    
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
    
    # Create scatter plot comparing walk counts and average durations
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with bubble size proportional to total time
    for _, row in walking_stats.iterrows():
        emp_id = row['id']
        x = row['walk_count']
        y = row['avg_duration_seconds']
        size = row['total_hours'] * 100  # Scale for better visibility
        color = employee_colors.get(emp_id, '#1f77b4')
        
        plt.scatter(
            x, y, 
            s=size, 
            color=color, 
            alpha=0.7, 
            edgecolor='black', 
            linewidth=1
        )
        
        # Add employee label
        plt.annotate(
            emp_id,
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontweight='bold'
        )
    
    # Add reference lines
    plt.axhline(walking_stats['avg_duration_seconds'].mean(), color='gray', linestyle='--', alpha=0.7)
    plt.axvline(walking_stats['walk_count'].mean(), color='gray', linestyle='--', alpha=0.7)
    
    # Annotate quadrants
    x_mean = walking_stats['walk_count'].mean()
    y_mean = walking_stats['avg_duration_seconds'].mean()
    
    plt.annotate(
        'Few, Long Walks',
        (0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.7)
    )
    
    plt.annotate(
        'Many, Long Walks',
        (0.95, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        ha='right',
        va='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.7)
    )
    
    plt.annotate(
        'Few, Short Walks',
        (0.05, 0.05),
        xycoords='axes fraction',
        fontsize=12,
        ha='left',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.7)
    )
    
    plt.annotate(
        'Many, Short Walks',
        (0.95, 0.05),
        xycoords='axes fraction',
        fontsize=12,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", alpha=0.7)
    )
    
    plt.title("Walking Pattern Comparison Across Employees", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Walks", fontsize=14)
    plt.ylabel("Average Walk Duration (seconds)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add explanation of bubble size
    plt.figtext(
        0.5, 0.01, 
        "Bubble size indicates total walking time", 
        ha='center', 
        fontsize=12, 
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / 'employee_walking_comparison.png')
    plt.close()


def analyze_region_walking_patterns(data, walking_data, output_dir=None, language='en'):
    """
    Analyze walking patterns with relation to specific regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete tracking data
    walking_data : pandas.DataFrame
        Filtered walking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with region pattern analysis results
    """
    results = {}
    
    # Get all regions and all employees
    all_regions = data['region'].unique()
    all_employees = data['id'].unique()
    
    # Create directory for region analysis
    if output_dir:
        region_dir = output_dir / 'regions'
        region_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate region transitions
    transitions = []
    transition_counts = defaultdict(int)
    
    # Group by employee
    for emp_id in all_employees:
        emp_data = data[data['id'] == emp_id].sort_values('startTime')
        
        # Track transitions between regions
        prev_region = None
        
        for _, row in emp_data.iterrows():
            curr_region = row['region']
            activity = row['activity']
            duration = row['duration']
            
            if prev_region and prev_region != curr_region:
                transition = (prev_region, curr_region)
                transitions.append({
                    'employee': emp_id,
                    'from_region': prev_region,
                    'to_region': curr_region,
                    'activity': activity,
                    'duration': duration
                })
                transition_counts[transition] += 1
            
            prev_region = curr_region
    
    # Convert transition data to DataFrame
    transitions_df = pd.DataFrame(transitions)
    
    # Calculate walk frequency between regions
    region_walk_freq = {}
    
    for region in all_regions:
        # Calculate walks entering this region
        entering_walks = walking_data[walking_data['region'] == region].shape[0]
        
        # Store in dictionary
        region_walk_freq[region] = entering_walks
    
    # Sort transitions by count
    sorted_transitions = sorted(
        [(k, v) for k, v in transition_counts.items()], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Top transitions
    top_transitions = sorted_transitions[:20]
    
    # Extract walking transition data
    walking_transitions = []
    
    for (from_region, to_region), count in top_transitions:
        # Filter transitions for this pair
        transition_walks = transitions_df[
            (transitions_df['from_region'] == from_region) & 
            (transitions_df['to_region'] == to_region) & 
            (transitions_df['activity'] == 'Walk')
        ]
        
        # Calculate average duration
        avg_duration = transition_walks['duration'].mean() if not transition_walks.empty else 0
        
        walking_transitions.append({
            'from_region': from_region,
            'to_region': to_region,
            'count': count,
            'walk_count': transition_walks.shape[0],
            'avg_duration': avg_duration,
            'total_duration': transition_walks['duration'].sum() if not transition_walks.empty else 0
        })
    
    # Convert to DataFrame
    walking_trans_df = pd.DataFrame(walking_transitions)
    
    # Save to CSV
    if output_dir:
        walking_trans_df.to_csv(region_dir / 'top_walking_transitions.csv', index=False)
    
    # Create visualization for top transitions
    if output_dir and not walking_trans_df.empty:
        create_transition_visualization(walking_trans_df, region_dir, language)
    
    # Calculate region statistics
    region_stats = []
    
    for region in all_regions:
        region_data = walking_data[walking_data['region'] == region]
        
        if not region_data.empty:
            total_duration = region_data['duration'].sum()
            walk_count = region_data.shape[0]
            unique_employees = region_data['id'].nunique()
            
            # Calculate correlation with other regions
            region_corr = calculate_region_correlation(data, region)
            
            region_stats.append({
                'region': region,
                'walking_time': total_duration,
                'walking_minutes': total_duration / 60,
                'formatted_time': format_seconds_to_hms(total_duration),
                'walk_count': walk_count,
                'unique_employees': unique_employees,
                'avg_duration': total_duration / walk_count if walk_count > 0 else 0,
                'highest_correlation': region_corr['highest'] if region_corr else None,
                'highest_corr_value': region_corr['highest_value'] if region_corr else 0
            })
    
    # Convert to DataFrame
    region_stats_df = pd.DataFrame(region_stats)
    
    # Sort by walking time
    region_stats_df = region_stats_df.sort_values('walking_time', ascending=False)
    
    # Save to CSV
    if output_dir:
        region_stats_df.to_csv(region_dir / 'region_walking_stats.csv', index=False)
    
    # Create visualization for region stats
    if output_dir and not region_stats_df.empty:
        create_region_stats_visualization(region_stats_df, region_dir, language)
    
    results['transitions'] = transitions_df
    results['region_stats'] = region_stats_df
    results['walking_transitions'] = walking_trans_df
    
    return results


def calculate_region_correlation(data, target_region):
    """
    Calculate correlation between time spent in target region vs other regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete tracking data
    target_region : str
        Region to calculate correlations for
    
    Returns:
    --------
    dict
        Dictionary with correlation results
    """
    # Get all regions
    all_regions = data['region'].unique()
    
    # Group by employee and region
    emp_region_time = data.groupby(['id', 'region'])['duration'].sum().reset_index()
    
    # Pivot to get time per region for each employee
    region_matrix = emp_region_time.pivot(index='id', columns='region', values='duration').fillna(0)
    
    # Only proceed if target region is in the matrix
    if target_region not in region_matrix.columns:
        return None
    
    # Calculate correlation with other regions
    correlations = {}
    
    for region in all_regions:
        if region != target_region and region in region_matrix.columns:
            # Calculate Pearson correlation
            corr, _ = pearsonr(region_matrix[target_region], region_matrix[region])
            correlations[region] = corr
    
    # Find highest correlation
    if correlations:
        highest_region = max(correlations.items(), key=lambda x: abs(x[1]))
        result = {
            'highest': highest_region[0],
            'highest_value': highest_region[1],
            'all_correlations': correlations
        }
    else:
        result = None
    
    return result


def create_transition_visualization(transitions_df, output_dir, language='en'):
    """
    Create visualization for top region transitions
    
    Parameters:
    -----------
    transitions_df : pandas.DataFrame
        DataFrame with transition statistics
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Filter to top 10 transitions
    top_trans = transitions_df.head(10)
    
    # Create labels for transitions
    labels = [f"{row['from_region']} → {row['to_region']}" for _, row in top_trans.iterrows()]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # 1. Bar chart for transition counts
    bars1 = ax1.barh(
        labels,
        top_trans['count'],
        color=plt.cm.viridis(np.linspace(0, 0.8, len(top_trans)))
    )
    
    # Add labels
    for bar, count in zip(bars1, top_trans['count']):
        ax1.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{count}",
            va='center',
            fontweight='bold'
        )
    
    ax1.set_title("Top Region Transitions", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Number of Transitions", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Bar chart for average walking duration
    bars2 = ax2.barh(
        labels,
        top_trans['avg_duration'],
        color=plt.cm.viridis(np.linspace(0, 0.8, len(top_trans)))
    )
    
    # Add labels
    for bar, avg_dur in zip(bars2, top_trans['avg_duration']):
        ax2.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{avg_dur:.1f}s",
            va='center'
        )
    
    ax2.set_title("Average Walking Duration Between Regions", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Seconds", fontsize=14)
    ax2.set_ylabel("Region Transition", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / 'top_walking_transitions.png')
    plt.close()


def create_region_stats_visualization(region_stats, output_dir, language='en'):
    """
    Create visualization for region walking statistics
    
    Parameters:
    -----------
    region_stats : pandas.DataFrame
        DataFrame with region statistics
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Filter to top 15 regions by walking time
    top_regions = region_stats.head(15)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create bar chart for walking time
    bars = plt.barh(
        top_regions['region'],
        top_regions['walking_minutes'],
        color=plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(top_regions)))
    )
    
    # Add labels with walk count
    for bar, minutes, count, employees in zip(
        bars, 
        top_regions['walking_minutes'],
        top_regions['walk_count'],
        top_regions['unique_employees']
    ):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{minutes:.1f}m ({count} walks, {employees} emp)",
            va='center',
            fontweight='bold'
        )
    
    plt.title("Top Regions by Walking Time", fontsize=16, fontweight='bold')
    plt.xlabel("Minutes Spent Walking", fontsize=14)
    plt.ylabel("Region", fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / 'top_walking_regions.png')
    plt.close()
    
    # Create correlation network visualization
    create_region_correlation_network(region_stats, output_dir, language)


def create_region_correlation_network(region_stats, output_dir, language='en'):
    """
    Create a network visualization showing correlations between regions
    
    Parameters:
    -----------
    region_stats : pandas.DataFrame
        DataFrame with region statistics including correlations
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    try:
        import networkx as nx
        
        # Filter to regions with valid correlation data
        corr_data = region_stats[region_stats['highest_correlation'].notna()]
        
        if len(corr_data) < 3:  # Need at least a few nodes for a meaningful network
            return
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (regions)
        for _, row in corr_data.iterrows():
            G.add_node(row['region'], size=row['walking_minutes'], walks=row['walk_count'])
        
        # Add edges (correlations)
        for _, row in corr_data.iterrows():
            if row['highest_correlation'] in G.nodes and abs(row['highest_corr_value']) > 0.4:
                G.add_edge(
                    row['region'],
                    row['highest_correlation'],
                    weight=abs(row['highest_corr_value']),
                    sign=1 if row['highest_corr_value'] > 0 else -1
                )
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Set positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get node sizes based on walking time
        node_sizes = [G.nodes[n]['size'] * 10 for n in G.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='skyblue',
            alpha=0.8,
            edgecolors='black'
        )
        
        # Draw edges with varying thickness based on correlation strength
        # Positive correlations (green)
        pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] > 0]
        pos_weights = [G[u][v]['weight'] * 3 for u, v in pos_edges]
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=pos_edges,
            width=pos_weights,
            edge_color='green',
            alpha=0.7
        )
        
        # Negative correlations (red)
        neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] < 0]
        neg_weights = [G[u][v]['weight'] * 3 for u, v in neg_edges]
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=neg_edges,
            width=neg_weights,
            edge_color='red',
            alpha=0.7,
            style='dashed'
        )
        
        # Add labels with smaller font
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold'
        )
        
        plt.title("Region Correlation Network", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add legend
        plt.figtext(0.01, 0.01, "Node size = Walking time | Green edges = Positive correlation | Red dashed = Negative correlation",
                  fontsize=10, ha='left',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), output_dir / 'region_correlation_network.png')
        plt.close()
        
    except ImportError:
        print("NetworkX not available for correlation network visualization")


def detect_repetitive_walking(walking_data, output_dir=None, language='en'):
    """
    Detect repetitive walking patterns between regions
    
    Parameters:
    -----------
    walking_data : pandas.DataFrame
        Filtered walking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with repetitive pattern analysis results
    """
    # Results dictionary
    results = {}
    
    # Create directory for pattern analysis
    if output_dir:
        pattern_dir = output_dir / 'patterns'
        pattern_dir.mkdir(exist_ok=True, parents=True)
    
    # Analyze each employee separately
    all_patterns = []
    
    for emp_id in walking_data['id'].unique():
        emp_data = walking_data[walking_data['id'] == emp_id].sort_values('startTime')
        
        # Create sequence of regions
        region_sequence = list(emp_data['region'])
        
        # Find repeated sequences of regions (minimum length 2)
        repeated_patterns = find_repeated_sequences(region_sequence, min_length=2, min_repeats=3)
        
        # Store results
        if repeated_patterns:
            for pattern, occurrences in repeated_patterns.items():
                pattern_regions = pattern.split(' → ')
                
                # Calculate typical duration for this pattern
                pattern_durations = []
                
                for i in range(len(region_sequence) - len(pattern_regions) + 1):
                    if region_sequence[i:i+len(pattern_regions)] == pattern_regions:
                        # Extract the walking segments
                        pattern_segment = emp_data.iloc[i:i+len(pattern_regions)]
                        pattern_durations.append(pattern_segment['duration'].sum())
                
                avg_duration = sum(pattern_durations) / len(pattern_durations) if pattern_durations else 0
                
                all_patterns.append({
                    'employee': emp_id,
                    'pattern': pattern,
                    'regions': len(pattern_regions),
                    'occurrences': len(occurrences),
                    'avg_duration': avg_duration,
                    'formatted_duration': format_seconds_to_hms(avg_duration)
                })
    
    # Convert to DataFrame
    pattern_df = pd.DataFrame(all_patterns) if all_patterns else pd.DataFrame()
    
    # Sort by occurrences
    if not pattern_df.empty:
        pattern_df = pattern_df.sort_values(['employee', 'occurrences'], ascending=[True, False])
    
    # Save to CSV
    if output_dir and not pattern_df.empty:
        pattern_df.to_csv(pattern_dir / 'repetitive_walking_patterns.csv', index=False)
        
        # Create visualization
        create_pattern_visualization(pattern_df, pattern_dir, language)
    
    results['patterns'] = pattern_df
    
    return results


def find_repeated_sequences(sequence, min_length=2, min_repeats=3):
    """
    Find repeated subsequences in a sequence
    
    Parameters:
    -----------
    sequence : list
        Input sequence
    min_length : int
        Minimum length of subsequence to consider
    min_repeats : int
        Minimum number of repetitions to qualify
    
    Returns:
    --------
    dict
        Dictionary with patterns as keys and lists of occurrence indices as values
    """
    patterns = {}
    
    # Check each possible subsequence length
    for length in range(min_length, min(10, len(sequence) // 2 + 1)):
        # Check each possible starting position
        for start in range(len(sequence) - length + 1):
            subseq = sequence[start:start+length]
            pattern = ' → '.join(subseq)
            
            # Skip if already found
            if pattern in patterns:
                continue
            
            # Find all occurrences
            occurrences = []
            
            for i in range(len(sequence) - length + 1):
                if sequence[i:i+length] == subseq:
                    occurrences.append(i)
            
            # Add if it meets minimum repetitions
            if len(occurrences) >= min_repeats:
                patterns[pattern] = occurrences
    
    return patterns


def create_pattern_visualization(pattern_df, output_dir, language='en'):
    """
    Create visualization for repetitive walking patterns
    
    Parameters:
    -----------
    pattern_df : pandas.DataFrame
        DataFrame with pattern information
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Group by employee
    grouped = pattern_df.groupby('employee')
    
    # Create a figure for each employee
    for emp_id, group in grouped:
        # Get top patterns (up to 10)
        top_patterns = group.head(10)
        
        if len(top_patterns) < 3:  # Skip if too few patterns
            continue
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Get employee color
        emp_color = employee_colors.get(emp_id, '#1f77b4')
        
        # Create bar chart for pattern occurrences
        bars = plt.barh(
            top_patterns['pattern'],
            top_patterns['occurrences'],
            color=emp_color,
            alpha=0.7
        )
        
        # Add labels
        for bar, count, duration in zip(bars, top_patterns['occurrences'], top_patterns['formatted_duration']):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{count} times (avg {duration})",
                va='center',
                fontweight='bold'
            )
        
        plt.title(f"Repetitive Walking Patterns for Employee {emp_id}", fontsize=16, fontweight='bold')
        plt.xlabel("Number of Occurrences", fontsize=14)
        plt.ylabel("Region Pattern", fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), output_dir / f"employee_{emp_id}_repetitive_patterns.png")
        plt.close()
    
    # Create summary for all employees
    create_pattern_summary(pattern_df, output_dir, language)


def create_pattern_summary(pattern_df, output_dir, language='en'):
    """
    Create a summary visualization for repetitive patterns across employees
    
    Parameters:
    -----------
    pattern_df : pandas.DataFrame
        DataFrame with pattern information
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    if pattern_df.empty:
        return
    
    set_visualization_style()
    
    # Count pattern occurrences by employee
    pattern_counts = pattern_df.groupby('employee')['occurrences'].agg(['count', 'sum']).reset_index()
    pattern_counts.columns = ['employee', 'unique_patterns', 'total_repetitions']
    
    # Sort by total repetitions
    pattern_counts = pattern_counts.sort_values('total_repetitions', ascending=False)
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    
    # 1. Unique patterns per employee
    bars1 = ax1.bar(
        pattern_counts['employee'],
        pattern_counts['unique_patterns'],
        color=[employee_colors.get(emp, '#1f77b4') for emp in pattern_counts['employee']]
    )
    
    # Add labels
    for bar, count in zip(bars1, pattern_counts['unique_patterns']):
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{count}",
            ha='center',
            fontweight='bold'
        )
    
    ax1.set_title("Unique Repetitive Patterns by Employee", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Number of Patterns", fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Total repetitions per employee
    bars2 = ax2.bar(
        pattern_counts['employee'],
        pattern_counts['total_repetitions'],
        color=[employee_colors.get(emp, '#1f77b4') for emp in pattern_counts['employee']]
    )
    
    # Add labels
    for bar, count in zip(bars2, pattern_counts['total_repetitions']):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{count}",
            ha='center',
            fontweight='bold'
        )
    
    ax2.set_title("Total Pattern Repetitions by Employee", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Employee ID", fontsize=14)
    ax2.set_ylabel("Total Repetitions", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / 'pattern_summary.png')
    plt.close()


def analyze_walking_by_shift(walking_data, output_dir=None, time_slot_minutes=30, language='en'):
    """
    Analyze walking patterns by shift for each employee
    
    Parameters:
    -----------
    walking_data : pandas.DataFrame
        Filtered walking data
    output_dir : Path, optional
        Directory to save output files
    time_slot_minutes : int, optional
        Size of time slots in minutes
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with shift analysis results
    """
    # Check if shift column exists
    if 'shift' not in walking_data.columns:
        print("No shift information found in the dataset.")
        return {}
    
    # Create shift-based directory
    if output_dir:
        shift_dir = output_dir / 'shifts'
        shift_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract hour of day from startTime 
    walking_data['hour_of_day'] = walking_data['startTime'] / 3600  # Convert to hours
    
    # Calculate number of slots in a day
    slots_per_day = 24 * 60 // time_slot_minutes
    
    # Create time slot index for each record
    walking_data['time_slot_idx'] = (walking_data['hour_of_day'] * 60 / time_slot_minutes).astype(int) % slots_per_day
    
    # Create time slot labels
    time_slot_labels = []
    for i in range(slots_per_day):
        minutes_from_midnight = i * time_slot_minutes
        hours = minutes_from_midnight // 60
        minutes = minutes_from_midnight % 60
        time_slot_labels.append(f"{hours:02d}:{minutes:02d}")
    
    walking_data['time_slot'] = walking_data['time_slot_idx'].apply(lambda x: time_slot_labels[x])
    
    # Calculate walking statistics by shift
    shift_stats = walking_data.groupby(['shift']).agg({
        'duration': ['sum', 'mean', 'count'],
        'id': 'nunique'
    }).reset_index()
    
    # Flatten multi-index columns
    shift_stats.columns = ['shift', 'total_duration', 'mean_duration', 'walk_count', 'employee_count']
    
    # Add formatted time and convert to minutes/hours
    shift_stats['walking_minutes'] = shift_stats['total_duration'] / 60
    shift_stats['walking_hours'] = shift_stats['total_duration'] / 3600
    shift_stats['formatted_total'] = shift_stats['total_duration'].apply(format_seconds_to_hms)
    shift_stats['formatted_mean'] = shift_stats['mean_duration'].apply(format_seconds_to_hms)
    
    # Calculate walking statistics by shift and region
    shift_region_stats = walking_data.groupby(['shift', 'region']).agg({
        'duration': 'sum',
        'id': 'nunique'
    }).reset_index()
    
    shift_region_stats.columns = ['shift', 'region', 'walking_duration', 'employee_count']
    shift_region_stats['walking_minutes'] = shift_region_stats['walking_duration'] / 60
    
    # Save statistics to CSV
    if output_dir:
        shift_stats.to_csv(shift_dir / 'walking_by_shift.csv', index=False)
        shift_region_stats.to_csv(shift_dir / 'walking_by_shift_region.csv', index=False)
    
    # Process each employee
    results = {}
    for emp_id in walking_data['id'].unique():
        emp_data = walking_data[walking_data['id'] == emp_id]
        shifts = emp_data['shift'].unique()
        
        if len(shifts) <= 1:
            continue  # Skip employees with only one shift
        
        # Create visualization comparing shifts
        create_shift_comparison_plot(
            emp_data, 
            emp_id, 
            shifts, 
            slots_per_day,
            time_slot_labels,
            shift_dir,
            time_slot_minutes,
            language
        )
        
        # Store results
        results[emp_id] = {
            'shifts': shifts.tolist(),
            'shift_counts': emp_data.groupby('shift').size().to_dict()
        }
    
    # Create all-employee shift comparisons
    for shift in walking_data['shift'].unique():
        shift_data = walking_data[walking_data['shift'] == shift]
        if len(shift_data['id'].unique()) > 1:
            create_shift_employee_comparison(
                shift_data,
                shift,
                slots_per_day,
                time_slot_labels,
                shift_dir,
                time_slot_minutes,
                language
            )
    
    # Create visualizations for shift statistics
    if output_dir:
        create_shift_stats_visualization(shift_stats, shift_region_stats, shift_dir, language)
    
    return {
        'shift_stats': shift_stats,
        'shift_region_stats': shift_region_stats,
        'employee_shifts': results
    }


def create_shift_comparison_plot(emp_data, emp_id, shifts, slots_per_day, 
                              time_slot_labels, output_dir, time_slot_minutes, language='en'):
    """
    Create an improved visualization comparing walking patterns across different shifts
    
    Parameters:
    -----------
    emp_data : pandas.DataFrame
        Walking data for a specific employee
    emp_id : str
        ID of the employee
    shifts : list
        List of shifts to compare
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
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get employee color
    employee_colors = get_employee_colors()
    base_color = employee_colors.get(emp_id, '#1f77b4')
    
    # Track active time periods
    all_active_slots = []
    
    # Create a color palette
    from matplotlib.colors import to_rgba
    base_rgba = to_rgba(base_color)
    
    # Calculate walking stats by shift and time slot
    all_shift_data = []
    max_minutes = 0
    
    for shift_idx, shift in enumerate(shifts):
        # Filter data for this shift
        shift_data = emp_data[emp_data['shift'] == shift]
        
        # Calculate time slot statistics
        slot_data = shift_data.groupby(['time_slot_idx', 'region']).agg({
            'duration': ['sum', 'count']
        }).reset_index()
        
        # Flatten multi-index
        slot_data.columns = ['time_slot_idx', 'region', 'total_duration', 'walk_count']
        
        # Find most frequent region per time slot
        most_common_regions = shift_data.groupby('time_slot_idx')['region'].apply(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).reset_index()
        most_common_regions.columns = ['time_slot_idx', 'most_common_region']
        
        # Track which slots have activity (for focusing the x-axis)
        active_slots = slot_data[slot_data['total_duration'] > 0]['time_slot_idx'].unique().tolist()
        all_active_slots.extend(active_slots)
        
        # Create summary by time slot
        time_summary = slot_data.groupby('time_slot_idx').agg({
            'total_duration': 'sum',
            'walk_count': 'sum'
        }).reset_index()
        
        # Create full time series with all slots
        full_slots = pd.DataFrame({
            'time_slot_idx': range(slots_per_day),
            'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
        })
        
        # Merge with slot data and region data
        full_data = pd.merge(full_slots, time_summary, on=['time_slot_idx'], how='left')
        full_data = pd.merge(full_data, most_common_regions, on=['time_slot_idx'], how='left')
        
        # Fill NAs
        full_data = full_data.fillna(0)
        full_data['duration_minutes'] = full_data['total_duration'] / 60
        
        all_shift_data.append((shift, full_data))
        max_minutes = max(max_minutes, full_data['duration_minutes'].max())
    
    # Determine colors for each shift (with nice gradient)
    shift_colors = {}
    for i, shift in enumerate(shifts):
        # Create pleasing gradient
        opacity = 0.8
        hue_shift = i / max(1, len(shifts) - 1)  # 0 to 1 range
        
        # Create a color that varies in hue but maintains the base saturation/value
        import colorsys
        h, s, v = colorsys.rgb_to_hsv(*base_rgba[:3])
        new_h = (h + 0.1 + hue_shift * 0.3) % 1.0  # Shift hue in pleasing range
        r, g, b = colorsys.hsv_to_rgb(new_h, s, v)
        
        shift_colors[shift] = (r, g, b, opacity)
    
    # Plot each shift's data with enhanced smoothing
    for shift, shift_data in all_shift_data:
        # Apply stronger smoothing for nicer curve appearance
        # Extract the y-values and apply Gaussian smoothing
        y_values = shift_data['duration_minutes'].values
        # Apply stronger Gaussian smoothing (sigma controls smoothness)
        smoothed_y = gaussian_filter1d(y_values, sigma=2.0) 
        
        # Plot with thicker, smoother lines
        ax.plot(
            shift_data['time_slot_idx'],
            smoothed_y,
            '-', 
            label=f"Shift {shift}",
            linewidth=3.0,  # Thicker line
            color=shift_colors[shift],
            alpha=0.9  # High opacity for rich color
        )
        
        # Add subtle fill below the line for density-like appearance
        ax.fill_between(
            shift_data['time_slot_idx'],
            smoothed_y,
            alpha=0.2,  # Light fill
            color=shift_colors[shift]
        )
        
        # Mark peaks for each shift with region information
        peak_indices = find_peaks(smoothed_y, min_height=0.1)
        
        # Sort peaks by height and take top 3
        peak_values = [(idx, smoothed_y[idx]) for idx in peak_indices]
        peak_values.sort(key=lambda x: x[1], reverse=True)
        
        for idx, height in peak_values[:3]:
            if height > 0:
                # Get region information
                region = shift_data.iloc[idx]['most_common_region']
                region_text = f"\n({region})" if region and region != 0 else ""
                
                ax.scatter(
                    idx,
                    height,
                    s=80,
                    color=shift_colors[shift],
                    edgecolor='white',
                    zorder=10
                )
                ax.annotate(
                    f"{height:.1f}m{region_text}",
                    xy=(idx, height),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontweight='bold',
                    color=shift_colors[shift],
                    arrowprops=dict(arrowstyle='->', color=shift_colors[shift])
                )
    
    # Focus x-axis on relevant time periods
    if all_active_slots:
        # Find earliest and latest active slots with padding
        min_slot = max(0, min(all_active_slots) - 2)  # Add padding
        max_slot = min(slots_per_day - 1, max(all_active_slots) + 2)  # Add padding
        
        # Set x-axis limits to focus only on active periods
        ax.set_xlim(min_slot, max_slot)
        
        # Create focused x-ticks within this range
        span = max_slot - min_slot
        if span <= 24:  # For smaller spans, show more detail
            tick_every = 1
        else:
            tick_every = span // 12  # About 12 ticks regardless of span
        
        focused_ticks = range(min_slot, max_slot + 1, tick_every)
        ax.set_xticks(focused_ticks)
        ax.set_xticklabels([time_slot_labels[i % len(time_slot_labels)] for i in focused_ticks], rotation=45)
    
    # Set y-axis limit with 20% padding
    ax.set_ylim(0, max_minutes * 1.2)
    
    # Set labels and title with more styling
    ax.set_title(f"Walking Patterns Across Shifts - Employee {emp_id}", 
                fontsize=18, fontweight='bold', color=base_rgba)
    ax.set_xlabel("Time of Day", fontsize=14, fontweight='bold')
    ax.set_ylabel("Minutes Spent Walking", fontsize=14, fontweight='bold')
    
    # Enhanced grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # More attractive legend
    legend = ax.legend(title="Shift", loc="upper right", frameon=True, fancybox=True, 
                      framealpha=0.9, shadow=True, fontsize=12)
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, output_dir / f"employee_{emp_id}_shift_comparison_{time_slot_minutes}min.png")
    
    plt.close()


def create_shift_employee_comparison(shift_data, shift, slots_per_day, 
                                  time_slot_labels, output_dir, time_slot_minutes, language='en'):
    """
    Create a visualization comparing all employees within a single shift
    with region information at peak times
    
    Parameters:
    -----------
    shift_data : pandas.DataFrame
        Walking data for a specific shift
    shift : str
        Shift identifier
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
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Get all employees in this shift
    employees = shift_data['id'].unique()
    
    # Calculate walking stats by employee and time slot
    all_emp_data = []
    
    for emp_id in employees:
        # Filter data for this employee
        emp_shift_data = shift_data[shift_data['id'] == emp_id]
        
        # Calculate time slot statistics
        slot_data = emp_shift_data.groupby(['time_slot_idx']).agg({
            'duration': ['sum', 'count'],
            'region': lambda x: x.value_counts().index[0] if len(x) > 0 else None  # Most frequent region
        }).reset_index()
        
        # Flatten columns
        slot_data.columns = ['time_slot_idx', 'total_duration', 'walk_count', 'most_common_region']
        slot_data['time_slot'] = slot_data['time_slot_idx'].apply(lambda x: time_slot_labels[x])
        slot_data['duration_minutes'] = slot_data['total_duration'] / 60
        
        # Create full time series with all slots
        full_slots = pd.DataFrame({
            'time_slot_idx': range(slots_per_day),
            'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
        })
        
        # Merge with slot data to ensure all slots exist
        full_data = pd.merge(full_slots, slot_data, on=['time_slot_idx', 'time_slot'], how='left')
        full_data = full_data.fillna(0)
        full_data = full_data.sort_values('time_slot_idx')
        
        all_emp_data.append((emp_id, full_data))
    
    # Plot each employee's data
    for emp_id, emp_data in all_emp_data:
        # Apply smoothing to make the curves look nicer
        y_values = emp_data['duration_minutes'].values
        # Apply Gaussian smoothing
        smoothed_y = gaussian_filter1d(y_values, sigma=1.5)
        
        # Get employee color
        color = employee_colors.get(emp_id, None)
        
        # Plot line
        ax.plot(
            emp_data['time_slot_idx'],
            smoothed_y,
            '-', 
            label=f"Employee {emp_id}",
            linewidth=2.5,
            color=color
        )
        
        # Mark peaks with region information
        peak_indices = find_peaks(smoothed_y, min_height=0.1)
        
        # Sort peaks by height and take top 2 per employee
        peak_values = [(idx, smoothed_y[idx]) for idx in peak_indices]
        peak_values.sort(key=lambda x: x[1], reverse=True)
        
        for idx, height in peak_values[:2]:
            if height > 0:
                # Get region information
                region = emp_data.iloc[idx]['most_common_region']
                region_text = f"\n({region})" if region and region != 0 else ""
                
                ax.scatter(
                    idx,
                    height,
                    s=70,
                    color=color,
                    edgecolor='white',
                    zorder=10
                )
                ax.annotate(
                    f"{height:.1f}m{region_text}",
                    xy=(idx, height),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontweight='bold',
                    color=color,
                    arrowprops=dict(arrowstyle='->', color=color)
                )
    
    # Find global active period
    all_active_slots = []
    for _, emp_data in all_emp_data:
        active_slots = emp_data[emp_data['duration_minutes'] > 0]['time_slot_idx'].tolist()
        all_active_slots.extend(active_slots)
    
    # Focus x-axis on relevant time periods
    if all_active_slots:
        # Find earliest and latest active slots with padding
        min_slot = max(0, min(all_active_slots) - 2)  # Add padding
        max_slot = min(slots_per_day - 1, max(all_active_slots) + 2)  # Add padding
        
        # Set x-axis limits to focus only on active periods
        ax.set_xlim(min_slot, max_slot)
        
        # Create focused x-ticks within this range
        span = max_slot - min_slot
        if span <= 24:  # For smaller spans, show more detail
            tick_every = 1
        else:
            tick_every = span // 12  # About 12 ticks regardless of span
        
        focused_ticks = range(min_slot, max_slot + 1, tick_every)
        ax.set_xticks(focused_ticks)
        ax.set_xticklabels([time_slot_labels[i % len(time_slot_labels)] for i in focused_ticks], rotation=45)
    
    # Set labels and title
    ax.set_title(f"Walking Patterns for Shift {shift}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time of Day", fontsize=14)
    ax.set_ylabel("Minutes Spent Walking", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Employee", loc="upper right")
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, output_dir / f"shift_{shift}_employee_comparison_{time_slot_minutes}min.png")
    
    plt.close()


def create_shift_stats_visualization(shift_stats, shift_region_stats, output_dir, language='en'):
    """
    Create visualizations for shift statistics
    
    Parameters:
    -----------
    shift_stats : pandas.DataFrame
        Walking statistics by shift
    shift_region_stats : pandas.DataFrame
        Walking statistics by shift and region
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # 1. Create bar chart for walking time by shift
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(
        shift_stats['shift'],
        shift_stats['walking_hours'],
        color=plt.cm.viridis(np.linspace(0, 0.8, len(shift_stats)))
    )
    
    # Add labels
    for bar, hours, count, emp_count in zip(
        bars, 
        shift_stats['walking_hours'],
        shift_stats['walk_count'],
        shift_stats['employee_count']
    ):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f"{hours:.1f}h\n{count} walks\n{emp_count} employees",
            ha='center',
            fontweight='bold'
        )
    
    plt.title("Walking Time by Shift", fontsize=16, fontweight='bold')
    plt.xlabel("Shift", fontsize=14)
    plt.ylabel("Hours Spent Walking", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / 'walking_time_by_shift.png')
    plt.close()
    
    # 2. Create heatmap for walking time by shift and region
    if not shift_region_stats.empty:
        # Pivot the data
        pivot_data = shift_region_stats.pivot_table(
            index='shift',
            columns='region',
            values='walking_minutes',
            fill_value=0
        )
        
        # Sort columns by total walking time
        col_sums = pivot_data.sum(axis=0)
        pivot_data = pivot_data[col_sums.sort_values(ascending=False).index]
        
        # Get top 15 regions
        pivot_data = pivot_data.iloc[:, :15]
        
        # Create heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(
            pivot_data,
            cmap='YlOrRd',
            annot=True,
            fmt='.1f',
            linewidths=0.5,
            cbar_kws={'label': 'Walking Minutes'}
        )
        
        plt.title("Walking Time by Shift and Region", fontsize=16, fontweight='bold')
        plt.ylabel("Shift", fontsize=14)
        plt.xlabel("Region", fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), output_dir / 'walking_heatmap_shift_region.png')
        plt.close()