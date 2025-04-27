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

def analyze_walking_by_shift(data, output_dir=None, time_slot_minutes=30, language='en'):
    """
    Analyze walking patterns by shift for each employee
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with tracking data
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
    # Filter walking data
    walking_data = data[data['activity'] == 'Walk'].copy()
    
    if walking_data.empty or 'shift' not in walking_data.columns:
        print("No walking data or shift information found in the dataset.")
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
    
    return results

def create_shift_comparison_plot(emp_data, emp_id, shifts, slots_per_day, 
                              time_slot_labels, output_dir, time_slot_minutes, language='en'):
    """
    Create an improved visualization comparing walking patterns across different shifts
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
        slot_data = shift_data.groupby('time_slot_idx')['duration'].agg(['sum', 'count']).reset_index()
        slot_data.columns = ['time_slot_idx', 'total_duration', 'walk_count']
        
        # Track which slots have activity (for focusing the x-axis)
        active_slots = slot_data[slot_data['total_duration'] > 0]['time_slot_idx'].tolist()
        all_active_slots.extend(active_slots)
        
        # Create full time series with all slots
        full_slots = pd.DataFrame({
            'time_slot_idx': range(slots_per_day),
            'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
        })
        
        # Merge with slot data to ensure all slots exist
        full_data = pd.merge(full_slots, slot_data, on=['time_slot_idx'], how='left')
        full_data = full_data.fillna(0).sort_values('time_slot_idx')
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
        from scipy.ndimage import gaussian_filter1d
        
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
        
        # Mark peaks for each shift (using the smoothed data)
        peaks = []
        for i in range(1, len(smoothed_y)-1):
            if smoothed_y[i] > max(smoothed_y[i-1], smoothed_y[i+1]) and smoothed_y[i] > 5:  # Only significant peaks
                peaks.append((i, smoothed_y[i]))
        
        # Sort peaks by height and take top 3
        peaks.sort(key=lambda x: x[1], reverse=True)
        for idx, height in peaks[:3]:
            if height > 0:
                ax.scatter(
                    idx,
                    height,
                    s=80,
                    color=shift_colors[shift],
                    edgecolor='white',
                    zorder=10
                )
                ax.annotate(
                    f"{height:.1f}m",
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
        ax.set_xticklabels([time_slot_labels[i] for i in focused_ticks], rotation=45)
    
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
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Calculate walking stats by employee and time slot
    all_emp_data = []
    
    for emp_id in shift_data['id'].unique():
        # Filter data for this employee
        emp_shift_data = shift_data[shift_data['id'] == emp_id]
        
        # Calculate time slot statistics
        slot_data = emp_shift_data.groupby('time_slot_idx')['duration'].agg(['sum', 'count']).reset_index()
        slot_data.columns = ['time_slot_idx', 'total_duration', 'walk_count']
        slot_data['time_slot'] = slot_data['time_slot_idx'].apply(lambda x: time_slot_labels[x])
        slot_data['duration_minutes'] = slot_data['total_duration'] / 60
        
        # Create full time series with all slots
        full_slots = pd.DataFrame({
            'time_slot_idx': range(slots_per_day),
            'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
        })
        
        # Merge with slot data to ensure all slots exist
        full_data = pd.merge(full_slots, slot_data, on=['time_slot_idx', 'time_slot'], how='left')
        full_data = full_data.fillna(0).sort_values('time_slot_idx')
        
        all_emp_data.append((emp_id, full_data))
    
    # Plot each employee's data
    for emp_id, emp_data in all_emp_data:
        # Apply smoothing to make the curves look nicer
        window_size = 3
        if len(emp_data) > window_size:
            emp_data['smoothed_minutes'] = emp_data['duration_minutes'].rolling(
                window=window_size, center=True).mean().fillna(emp_data['duration_minutes'])
        else:
            emp_data['smoothed_minutes'] = emp_data['duration_minutes']
        
        color = employee_colors.get(emp_id, None)
        ax.plot(
            emp_data['time_slot_idx'],
            emp_data['smoothed_minutes'],  # Use smoothed data for nicer curves
            '-', 
            label=f"Employee {emp_id}",
            linewidth=2.5,
            color=color
        )
    
    # Improve x-axis tick labels
    if slots_per_day <= 48:
        tick_every = 1
    else:
        tick_every = slots_per_day // 24
    
    ax.set_xticks(range(0, slots_per_day, tick_every))
    ax.set_xticklabels([time_slot_labels[i] for i in range(0, slots_per_day, tick_every)], rotation=45)
    
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
    
    
def create_walking_density_plot(data, output_dir=None, language='en'):
    """
    Create density plots showing the distribution of walking times by employee
    """
    import seaborn as sns
    from scipy import stats
    
    # Filter for walking data
    walking_data = data[data['activity'] == 'Walk'].copy()
    
    # Create directory for density plots
    if output_dir:
        density_dir = output_dir / 'density'
        density_dir.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(12, 8))
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # For each employee, create a KDE of their walking durations
    for emp_id in walking_data['id'].unique():
        # Filter for this employee
        emp_data = walking_data[walking_data['id'] == emp_id]
        
        # Get walking durations in minutes
        durations = emp_data['duration'] / 60
        
        # Use Gaussian KDE to estimate the probability density
        if len(durations) > 10:  # Need enough data for meaningful KDE
            # Estimate the probability density
            density = stats.gaussian_kde(durations)
            
            # Plot range (from min to max with padding)
            x_min = max(0, durations.min() - 1)
            x_max = durations.max() + 1
            x = np.linspace(x_min, x_max, 1000)
            
            # Get the color for this employee
            color = employee_colors.get(emp_id, 'blue')
            
            # Plot the density curve
            plt.plot(x, density(x), '-', lw=2.5, color=color, label=f"Employee {emp_id}")
            
            # Fill the area under the curve
            plt.fill_between(x, density(x), alpha=0.3, color=color)
    
    plt.title('Distribution of Walking Durations by Employee', fontsize=16, fontweight='bold')
    plt.xlabel('Duration (minutes)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(density_dir / "walking_duration_density.png", dpi=300)
    
    plt.close()
    
    # For ridge plot (like your third image), you need to reshape the data
    # and use Seaborn's specialized functions
    
    # Prepare data in long format
    ridge_data = []
    for emp_id in walking_data['id'].unique():
        emp_data = walking_data[walking_data['id'] == emp_id]
        durations = emp_data['duration'] / 60
        
        for d in durations:
            ridge_data.append({
                'employee': f"Employee {emp_id}",
                'duration': d
            })
    
    if ridge_data:
        ridge_df = pd.DataFrame(ridge_data)
        
        plt.figure(figsize=(12, 8))
        
        # Create ridge plot
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        # Initialize the FacetGrid object
        g = sns.FacetGrid(ridge_df, row="employee", height=1.5, aspect=6)
        
        # Draw the densities in a loop
        for i, (name, color) in enumerate(employee_colors.items()):
            g.map_dataframe(sns.kdeplot, x="duration", 
                           fill=True, alpha=.5, linewidth=1.5)
        
        # Add titles and labels
        g.fig.suptitle('Distribution of Walking Durations', fontsize=16, y=.95)
        g.set_axis_labels("Duration (minutes)", "")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.fig.subplots_adjust(hspace=0.5)
        g.set(xlim=(0, ridge_df['duration'].max() * 1.1))
        
        if output_dir:
            plt.savefig(density_dir / "walking_duration_ridgeplot.png", dpi=300)
        
        plt.close()