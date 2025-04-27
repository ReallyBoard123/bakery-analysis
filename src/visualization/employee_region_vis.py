"""
Enhanced Employee Region Visualization Module

Creates employee-specific region heatmaps with walking path analysis, transition visualization,
and comprehensive region visit analysis. Distinguishes between meaningful region presence versus
transitory visits, and provides both individual shift and combined analyses.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.gridspec as gridspec

from .base import (save_figure, set_visualization_style, 
                  get_employee_colors, get_text)
from ..utils.time_utils import format_seconds_to_hms

# Threshold to determine if a visit is meaningful or just passing through (seconds)
MEANINGFUL_PRESENCE_THRESHOLD = 15

def create_employee_region_heatmap(data, floor_plan_data, employee_id, save_path=None, 
                                  top_n=10, min_transitions=2, figsize=(20, 14), language='en'):
    """
    Create employee-specific region heatmap with walking path analysis, transition visualization,
    and region visit analysis. Distinguishes between meaningful region presence and transitory visits.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    floor_plan_data : dict
        Dictionary containing floor plan data
    employee_id : str
        ID of the employee to analyze
    save_path : str or Path, optional
        Path to save the visualization
    top_n : int, optional
        Number of top regions to display
    min_transitions : int, optional
        Minimum number of transitions to include
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
    
    # Get employee colors
    employee_colors = get_employee_colors()
    emp_color = employee_colors.get(employee_id, '#2D5F91')
    
    # Filter data for the specified employee
    employee_data = data[data['id'] == employee_id]
    if employee_data.empty:
        print(get_text('No data found for employee {0}', language).format(employee_id))
        return None
    
    # Extract department info
    dept = employee_data['department'].iloc[0] if 'department' in employee_data.columns else 'Unknown'
    
    # Calculate total duration for this employee
    total_duration = employee_data['duration'].sum()
    total_duration_formatted = format_seconds_to_hms(total_duration)
    
    # Extract floor plan elements
    floor_plan = floor_plan_data['floor_plan']
    region_coordinates = floor_plan_data['region_coordinates']
    region_dimensions = floor_plan_data['region_dimensions']
    name_connections = floor_plan_data['name_connections']
    
    # Perform region visit analysis
    visit_analysis = analyze_region_visits(employee_data)
    
    # Get top regions by total duration
    top_regions_data = []
    for region, stats in visit_analysis['region_stats'].items():
        if stats['meaningful_visits'] > 0:  # Only include regions with meaningful visits
            top_regions_data.append({
                'region': region,
                'total_duration': stats['total_duration'],
                'meaningful_visits': stats['meaningful_visits'],
                'avg_meaningful_duration': stats['avg_meaningful_duration'],
                'formatted_duration': format_seconds_to_hms(stats['total_duration'])
            })
    
    # Sort by total duration
    top_regions_data.sort(key=lambda x: x['total_duration'], reverse=True)
    top_regions_data = top_regions_data[:top_n]
    top_regions = [r['region'] for r in top_regions_data]
    
    # Calculate transitions between regions
    transition_counts = {}
    for from_region in top_regions:
        for to_region in top_regions:
            if from_region != to_region:
                key = (from_region, to_region)
                count = visit_analysis['transition_counts'].get(key, 0)
                if count >= min_transitions:
                    transition_counts[key] = count
    
    # Calculate region percentiles for coloring
    region_durations = [r['total_duration'] for r in top_regions_data]
    region_percentiles = {}
    for i, region_info in enumerate(top_regions_data):
        # Calculate percentile (1-10)
        percentile = np.ceil(10 * (len(top_regions_data) - i) / len(top_regions_data))
        region_percentiles[region_info['region']] = percentile
    
    # Calculate transition percentiles for line thickness
    transition_values = list(transition_counts.values())
    transition_percentiles = {}
    
    if transition_values:
        # Sort transitions by count
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1])
        
        # Calculate percentiles (1-10)
        for i, ((from_region, to_region), count) in enumerate(sorted_transitions):
            percentile = np.ceil(10 * (i + 1) / len(sorted_transitions))
            transition_percentiles[(from_region, to_region)] = percentile
    
    # Calculate transition time between region pairs (bidirectional)
    transition_table_data = []
    
    # Process all region pairs
    for i, region1 in enumerate(top_regions):
        for region2 in top_regions[i+1:]:
            # Find all transitions involving these regions
            transition_duration = 0
            transition_count = 0
            
            # Count transitions from region1 to region2
            count_1_to_2 = visit_analysis['transition_counts'].get((region1, region2), 0)
            count_2_to_1 = visit_analysis['transition_counts'].get((region2, region1), 0)
            transition_count = count_1_to_2 + count_2_to_1
            
            if transition_count >= min_transitions:
                # Find all transitions where either of these regions is transitory
                for visit in visit_analysis['transitions']:
                    if visit['region'] == region1 or visit['region'] == region2:
                        transition_duration += visit['duration']
                
                # Add to table data
                transition_table_data.append({
                    'regions': f"{region1} ↔ {region2}",
                    'count': transition_count,
                    'duration': transition_duration,
                    'formatted_duration': format_seconds_to_hms(transition_duration),
                    'percentage': (transition_duration / total_duration * 100) if total_duration > 0 else 0
                })
    
    # Sort by count (descending)
    transition_table_data.sort(key=lambda x: x['count'], reverse=True)
    
    # Create figure with gridspec to have map, region table, and transition table
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 3, 1], height_ratios=[3, 1])
    
    # Region table axis (on the left)
    ax_region_table = fig.add_subplot(gs[0, 0])
    ax_region_table.axis('off')
    
    # Main heatmap axis (in the center)
    ax_map = fig.add_subplot(gs[0, 1:])
    
    # Transition table axis (bottom)
    ax_transition_table = fig.add_subplot(gs[1, :])
    ax_transition_table.axis('off')
    
    # Display the floor plan
    img = np.array(floor_plan)
    ax_map.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Create a custom colormap for regions
    region_cmap = plt.cm.get_cmap('YlOrRd')
    
    # Create a custom colormap for transitions
    transition_cmap = LinearSegmentedColormap.from_list(
        'transition_cmap', 
        ['#87CEEB', '#44A4DF', '#FFD700', '#FF8C00', '#FF4500']
    )
    
    # Draw regions as rectangles with color based on percentile
    for region_info in top_regions_data:
        region = region_info['region']
        if region in region_coordinates and region in region_dimensions:
            x, y = region_coordinates[region]
            width, height = region_dimensions[region]
            
            # Scale coordinates and dimensions to match image
            x_scaled = x * img_width - (width * img_width / 2)
            y_scaled = y * img_height - (height * img_height / 2)
            width_scaled = width * img_width
            height_scaled = height * img_height
            
            # Get color based on percentile
            percentile = region_percentiles[region]
            color_intensity = percentile / 10.0
            color = region_cmap(color_intensity)
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_scaled, y_scaled),
                width_scaled, height_scaled,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax_map.add_patch(rect)
            
            # Format time display
            duration_str = region_info['formatted_duration']
            visits = region_info['meaningful_visits']
            avg_duration_sec = int(region_info['avg_meaningful_duration'])
            
            # Add label with formatted info (no region name)
            ax_map.text(
                x * img_width, y * img_height,
                f"{duration_str}\n{visits} visits - {avg_duration_sec}sec",
                fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
    
    # Sort transitions by count to ensure higher counts are drawn on top
    sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1])
    
    # Draw transitions between regions with thickness based on percentile
    for (from_region, to_region), count in sorted_transitions:
        if from_region in region_coordinates and to_region in region_coordinates:
            from_x, from_y = region_coordinates[from_region]
            to_x, to_y = region_coordinates[to_region]
            
            # Scale coordinates
            from_x_scaled = from_x * img_width
            from_y_scaled = from_y * img_height
            to_x_scaled = to_x * img_width
            to_y_scaled = to_y * img_height
            
            # Get percentile (1-10)
            percentile = transition_percentiles.get((from_region, to_region), 1)
            
            # Calculate line width based on percentile (1-10 scale)
            line_width = percentile * 0.5  # Scale from 0.5 to 5.0
            
            # Get color based on percentile
            color_intensity = percentile / 10.0
            color = transition_cmap(color_intensity)
            
            # Draw arrow with solid line
            ax_map.arrow(
                from_x_scaled, from_y_scaled,
                (to_x_scaled - from_x_scaled) * 0.8,  # Shorten arrow to not overlap with regions
                (to_y_scaled - from_y_scaled) * 0.8,
                head_width=15,
                head_length=15,
                fc=color,
                ec=color,
                linewidth=line_width,
                length_includes_head=True,
                alpha=0.7,
                zorder=int(percentile)  # Higher percentile = higher z-index
            )
    
    # Create region usage table
    if top_regions_data:
        # Format table data
        table_data = [[get_text('Region', language), get_text('Duration', language), 
                      get_text('Visits', language), '%']]
        
        # Add data rows
        top_regions_duration = sum(r['total_duration'] for r in top_regions_data)
        for region_info in top_regions_data:
            percentage = (region_info['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            table_data.append([
                region_info['region'],
                region_info['formatted_duration'],
                str(region_info['meaningful_visits']),
                f"{percentage:.1f}%"
            ])
        
        # Add total row
        top_percentage = (top_regions_duration / total_duration * 100) if total_duration > 0 else 0
        table_data.append([
            get_text('Total', language),
            format_seconds_to_hms(top_regions_duration),
            str(sum(r['meaningful_visits'] for r in top_regions_data)),
            f"{top_percentage:.1f}%"
        ])
        
        # Create table
        region_table = ax_region_table.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.25, 0.15, 0.2]
        )
        
        # Style table
        region_table.auto_set_font_size(False)
        region_table.set_fontsize(11)
        region_table.scale(1, 1.5)
        
        # Add header formatting
        for i in range(4):
            region_table[(0, i)].set_facecolor(emp_color)  # Use employee color for header
            region_table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Add total row formatting
        last_row = len(table_data) - 1
        for i in range(4):
            region_table[(last_row, i)].set_facecolor('#E8655F')
            region_table[(last_row, i)].set_text_props(fontweight='bold')
    
    # Create transition table
    if transition_table_data:
        # Format table data
        trans_table_data = [[get_text('Regions (bidirectional)', language), 
                           get_text('Transitions', language), 
                           get_text('Duration', language), '%']]
        
        # Add data rows
        for transition in transition_table_data[:10]:  # Top 10 transitions
            trans_table_data.append([
                transition['regions'],
                str(transition['count']),
                transition['formatted_duration'],
                f"{transition['percentage']:.1f}%"
            ])
        
        # Calculate total
        total_trans_count = sum(t['count'] for t in transition_table_data)
        total_trans_duration = sum(t['duration'] for t in transition_table_data)
        total_trans_percentage = (total_trans_duration / total_duration * 100) if total_duration > 0 else 0
        
        # Add total row
        trans_table_data.append([
            get_text('Total Transitions', language),
            str(total_trans_count),
            format_seconds_to_hms(total_trans_duration),
            f"{total_trans_percentage:.1f}%"
        ])
        
        # Create table
        transition_table = ax_transition_table.table(
            cellText=trans_table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.4, 0.2, 0.25, 0.15]
        )
        
        # Style table
        transition_table.auto_set_font_size(False)
        transition_table.set_fontsize(11)
        transition_table.scale(1, 1.5)
        
        # Add header formatting
        for i in range(4):
            transition_table[(0, i)].set_facecolor('#2D5F91')  # Blue for transition header
            transition_table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Add total row formatting
        last_row = len(trans_table_data) - 1
        for i in range(4):
            transition_table[(last_row, i)].set_facecolor('#E8655F')
            transition_table[(last_row, i)].set_text_props(fontweight='bold')
    
    # Add title with total duration - use employee color
    ax_map.set_title(get_text('Region Analysis: Employee {0} ({1}) - Total Duration: {2}', language).format(
        employee_id, get_text(dept, language), total_duration_formatted), 
        fontsize=16, fontweight='bold', color=emp_color)
    
    # Turn off axis
    ax_map.axis('off')
    
    # Add note about meaningful presence
    plt.figtext(0.01, 0.01, get_text('Note: Only regions with meaningful presence (>{0}s) are displayed.', language).format(
        MEANINGFUL_PRESENCE_THRESHOLD), fontsize=9, ha='left')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def create_combined_region_analysis(data, floor_plan_data, employee_ids, output_dir=None, 
                                   top_n=10, min_transitions=2, figsize=(20, 14), language='en'):
    """
    Create combined region analysis for multiple shifts or employees.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with tracking data
    floor_plan_data : dict
        Dictionary containing floor plan data
    employee_ids : list or str
        List of employee IDs or single employee ID
    output_dir : Path, optional
        Output directory
    top_n : int, optional
        Number of top regions to display
    min_transitions : int, optional
        Minimum number of transitions to include
    figsize : tuple, optional
        Figure size (width, height) in inches
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    list
        List of created figure paths
    """
    # Ensure employee_ids is a list
    if isinstance(employee_ids, str):
        employee_ids = [employee_ids]
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    figure_paths = []
    
    # Process each employee
    for emp_id in employee_ids:
        # Filter data for this employee
        employee_data = data[data['id'] == emp_id]
        if employee_data.empty:
            print(get_text('No data found for employee {0}', language).format(emp_id))
            continue
        
        # Generate path for combined analysis
        if output_dir:
            combined_path = output_dir / f"{emp_id}_combined_analysis.png"
            figure_paths.append(combined_path)
        else:
            combined_path = None
        
        # Create combined visualization
        fig = create_employee_region_heatmap(
            data, floor_plan_data, emp_id, 
            save_path=combined_path,
            top_n=top_n, min_transitions=min_transitions,
            figsize=figsize, language=language
        )
        
        # Analyze by shift if 'shift' column exists
        if 'shift' in employee_data.columns:
            shifts = employee_data['shift'].unique()
            
            for shift in shifts:
                # Filter data for this shift
                shift_data = data[(data['id'] == emp_id) & (data['shift'] == shift)]
                
                # Skip if not enough data
                if len(shift_data) < 10:
                    continue
                
                # Generate path for shift analysis
                if output_dir:
                    shift_path = output_dir / f"{emp_id}_shift_{shift}_analysis.png"
                    figure_paths.append(shift_path)
                else:
                    shift_path = None
                
                # Create shift-specific visualization
                create_employee_region_heatmap(
                    shift_data, floor_plan_data, emp_id, 
                    save_path=shift_path,
                    top_n=top_n, min_transitions=min_transitions,
                    figsize=figsize, language=language
                )
    
    return figure_paths

def analyze_region_visits(employee_data):
    """
    Analyze an employee's visits to regions, distinguishing between meaningful presence and transitions
    
    Parameters:
    -----------
    employee_data : pandas.DataFrame
        Data for a single employee
    
    Returns:
    --------
    dict
        Dictionary with region visit analysis
    """
    results = {
        'meaningful_visits': [],
        'transitions': [],
        'all_visits': []
    }
    
    # Sort data by time
    sorted_data = employee_data.sort_values('startTime')
    
    current_region = None
    visit_start_time = None
    visit_rows = []
    
    # Process each record sequentially
    for _, row in sorted_data.iterrows():
        if current_region is None:
            # First record - start tracking this region
            current_region = row['region']
            visit_start_time = row['startTime']
            visit_rows = [row]
        elif row['region'] != current_region:
            # Region changed - analyze the completed visit
            visit_end_time = visit_rows[-1]['endTime']
            
            # Calculate visit duration and activities
            visit_duration = sum(r['duration'] for r in visit_rows)
            activity_durations = {}
            for r in visit_rows:
                activity = r['activity']
                if activity not in activity_durations:
                    activity_durations[activity] = 0
                activity_durations[activity] += r['duration']
            
            # Create visit record
            visit = {
                'region': current_region,
                'start_time': visit_start_time,
                'end_time': visit_end_time,
                'duration': visit_duration,
                'activity_durations': activity_durations,
                'rows': visit_rows,
                'row_count': len(visit_rows)
            }
            
            # Categorize as meaningful visit or transition
            if visit_duration >= MEANINGFUL_PRESENCE_THRESHOLD:
                results['meaningful_visits'].append(visit)
            else:
                results['transitions'].append(visit)
            
            results['all_visits'].append(visit)
            
            # Start tracking new region
            current_region = row['region']
            visit_start_time = row['startTime']
            visit_rows = [row]
        else:
            # Same region - continue tracking this visit
            visit_rows.append(row)
    
    # Process the final visit if there is one
    if current_region is not None and visit_rows:
        visit_duration = sum(r['duration'] for r in visit_rows)
        activity_durations = {}
        for r in visit_rows:
            activity = r['activity']
            if activity not in activity_durations:
                activity_durations[activity] = 0
            activity_durations[activity] += r['duration']
        
        visit = {
            'region': current_region,
            'start_time': visit_start_time,
            'end_time': visit_rows[-1]['endTime'],
            'duration': visit_duration,
            'activity_durations': activity_durations,
            'rows': visit_rows,
            'row_count': len(visit_rows)
        }
        
        if visit_duration >= MEANINGFUL_PRESENCE_THRESHOLD:
            results['meaningful_visits'].append(visit)
        else:
            results['transitions'].append(visit)
        
        results['all_visits'].append(visit)
    
    # Calculate region statistics
    region_stats = defaultdict(lambda: {
        'total_duration': 0,
        'meaningful_visits': 0,
        'transition_visits': 0,
        'total_visits': 0,
        'meaningful_durations': [],
        'transition_durations': [],
        'all_durations': []
    })
    
    for visit in results['meaningful_visits']:
        region = visit['region']
        region_stats[region]['total_duration'] += visit['duration']
        region_stats[region]['meaningful_visits'] += 1
        region_stats[region]['total_visits'] += 1
        region_stats[region]['meaningful_durations'].append(visit['duration'])
        region_stats[region]['all_durations'].append(visit['duration'])
    
    for visit in results['transitions']:
        region = visit['region']
        region_stats[region]['total_duration'] += visit['duration']
        region_stats[region]['transition_visits'] += 1
        region_stats[region]['total_visits'] += 1
        region_stats[region]['transition_durations'].append(visit['duration'])
        region_stats[region]['all_durations'].append(visit['duration'])
    
    # Calculate averages and other statistics
    for region, stats in region_stats.items():
        if stats['meaningful_durations']:
            stats['avg_meaningful_duration'] = sum(stats['meaningful_durations']) / len(stats['meaningful_durations'])
            stats['max_meaningful_duration'] = max(stats['meaningful_durations'])
            stats['min_meaningful_duration'] = min(stats['meaningful_durations'])
        else:
            stats['avg_meaningful_duration'] = 0
            stats['max_meaningful_duration'] = 0
            stats['min_meaningful_duration'] = 0
        
        if stats['transition_durations']:
            stats['avg_transition_duration'] = sum(stats['transition_durations']) / len(stats['transition_durations'])
        else:
            stats['avg_transition_duration'] = 0
        
        if stats['all_durations']:
            stats['avg_duration'] = sum(stats['all_durations']) / len(stats['all_durations'])
        else:
            stats['avg_duration'] = 0
    
    results['region_stats'] = dict(region_stats)
    
    # Calculate transitions between regions
    region_transitions = []
    for i in range(len(results['all_visits']) - 1):
        from_region = results['all_visits'][i]['region']
        to_region = results['all_visits'][i + 1]['region']
        region_transitions.append((from_region, to_region))
    
    results['region_transitions'] = region_transitions
    results['transition_counts'] = Counter(region_transitions)
    
    return results

def create_employee_transition_visualization(data, floor_plan_data, employee_id, shift=None, 
                                            save_path=None, min_transitions=3, figsize=(16, 12),
                                            language='en'):
    """
    Create visualization of region transitions for a specific employee.
    This function is kept for backward compatibility.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    floor_plan_data : dict
        Dictionary containing floor plan data
    employee_id : str
        ID of the employee to analyze
    shift : int, optional
        Shift to analyze (if None, includes all shifts)
    save_path : str or Path, optional
        Path to save the visualization
    min_transitions : int, optional
        Minimum number of transitions to include
    figsize : tuple, optional
        Figure size (width, height) in inches
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Filter data
    filtered_data = data[(data['id'] == employee_id)]
    if shift is not None:
        filtered_data = filtered_data[filtered_data['shift'] == shift]
    
    # Create enhanced version using the new function
    fig = create_employee_region_heatmap(
        filtered_data, floor_plan_data, employee_id,
        save_path=save_path, min_transitions=min_transitions,
        figsize=figsize, language=language
    )
    
    return fig

def plot_movement_transitions_chart(transition_counts, save_path=None, top_n=10, figsize=(15, 10), language='en'):
    """
    Plot top movement transitions as horizontal bar charts.
    This function is kept for backward compatibility.
    
    Parameters:
    -----------
    transition_counts : dict
        Dictionary with transition counts by employee
    save_path : str or Path, optional
        Path to save the visualization
    top_n : int, optional
        Number of top transitions to show
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
    fig, axes = plt.subplots(len(transition_counts), 1, figsize=figsize, sharex=True)
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Ensure axes is always a list
    if len(transition_counts) == 1:
        axes = [axes]
    
    for i, (emp_id, counts) in enumerate(transition_counts.items()):
        # Get top transitions
        top_transitions = counts.sort_values(ascending=False).head(top_n)
        
        # Create transition labels
        labels = [f"{src} → {dst}" for (src, dst) in top_transitions.index]
        
        # Get employee-specific color
        color = employee_colors.get(emp_id, '#2D5F91')
        
        # Plot horizontal bar chart
        bars = axes[i].barh(labels, top_transitions.values, color=color)
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"{width:.0f}", ha='left', va='center', fontweight='bold')
        
        axes[i].set_title(get_text('Top {0} Movement Transitions - Employee {1}', language).format(top_n, emp_id), 
                          fontsize=14, fontweight='bold')
        axes[i].set_xlabel(get_text('Number of Transitions', language), fontsize=12)
        axes[i].set_ylabel(get_text('Transition Path', language), fontsize=12)
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def run_employee_heatmaps(data, floor_plan_data, output_dir, employee_id=None, language='en'):
    """
    Run employee heatmap generation for one or all employees.
    This is a utility function that can be called from main.py.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    floor_plan_data : dict
        Dictionary containing floor plan data
    output_dir : Path
        Output directory
    employee_id : str, optional
        If provided, generate heatmap only for this employee
    language : str, optional
        Language code ('en' or 'de')
    """
    from ..utils.file_utils import ensure_dir_exists
    
    # Create directory for employee region heatmaps
    employee_heatmaps_dir = ensure_dir_exists(output_dir / 'employee_heatmaps')
    
    # Get employees to process
    if employee_id:
        if employee_id not in data['id'].unique():
            print(get_text('Error: Employee {0} not found in the data', language).format(employee_id))
            return
        employees = [employee_id]
    else:
        employees = sorted(data['id'].unique())
    
    print(f"\n{get_text('Generating heatmaps for {0} employees:', language).format(len(employees))}")
    
    # Generate region heatmap for each employee
    for emp_id in employees:
        print(f"  {get_text('Processing employee {0}...', language).format(emp_id)}")
        
        # Create directory for this employee
        emp_dir = ensure_dir_exists(employee_heatmaps_dir / emp_id)
        
        # Generate combined analysis for all shifts
        create_combined_region_analysis(
            data, 
            floor_plan_data,
            emp_id,
            output_dir=emp_dir,
            language=language
        )
    
    print(f"  {get_text('Saved employee region analyses to {0}/employee_heatmaps/', language).format(output_dir)}")