"""
Improved Employee Region Visualization Module

Creates employee-specific region heatmaps with simplified transition lines and region duration table.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.gridspec as gridspec

from .base import (save_figure, set_visualization_style, 
                  get_employee_colors, get_text)
from ..utils.time_utils import format_seconds_to_hms

def create_employee_region_heatmap(data, floor_plan_data, employee_id, save_path=None, 
                                  top_n=10, min_transitions=2, figsize=(20, 14), language='en'):
    """
    Create employee-specific region heatmap with simplified transition lines and region duration table
    
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
    
    # Calculate total duration for this employee across all shifts
    total_duration = employee_data['duration'].sum()
    total_duration_formatted = format_seconds_to_hms(total_duration)
    
    # Calculate time spent in each region
    region_durations = employee_data.groupby('region')['duration'].sum().reset_index()
    region_durations = region_durations.sort_values('duration', ascending=False)
    
    # Add formatted time column and percentage of total time
    region_durations['formatted_time'] = region_durations['duration'].apply(format_seconds_to_hms)
    region_durations['percentage'] = (region_durations['duration'] / total_duration * 100).round(1)
    
    # Get top N regions by duration
    top_regions_data = region_durations.head(top_n)
    top_regions = top_regions_data['region'].tolist()
    
    # Calculate total duration of top N regions
    top_regions_total_duration = top_regions_data['duration'].sum()
    top_regions_total_formatted = format_seconds_to_hms(top_regions_total_duration)
    
    # Create figure with gridspec to have both map and table
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 3, 0.1, 0.5, 0.5])
    
    # Table axis (on the left)
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')
    
    # Main heatmap axis (in the center)
    ax_map = fig.add_subplot(gs[1:3])
    
    # Extract floor plan elements
    floor_plan = floor_plan_data['floor_plan']
    region_coordinates = floor_plan_data['region_coordinates']
    region_dimensions = floor_plan_data['region_dimensions']
    name_connections = floor_plan_data['name_connections']
    
    # Calculate transitions between top regions only
    transitions = []
    prev_region = None
    for _, row in employee_data.sort_values('startTime').iterrows():
        current_region = row['region']
        if prev_region and prev_region != current_region:
            if prev_region in top_regions and current_region in top_regions:
                transitions.append((prev_region, current_region))
        prev_region = current_region
    
    # Count transitions
    transition_counts = Counter(transitions)
    
    # Filter to transitions with count >= min_transitions
    filtered_transitions = {t: count for t, count in transition_counts.items() if count >= min_transitions}
    
    # Display the floor plan
    img = np.array(floor_plan)
    ax_map.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Create a custom colormap for transitions
    transition_cmap = LinearSegmentedColormap.from_list(
        'transition_cmap', 
        ['#87CEEB', '#44A4DF', '#FFD700', '#FF8C00', '#FF4500']
    )
    
    # Create bucketed line thickness ranges for the legend
    line_thickness_legend = []
    line_thickness_ranges = []
    
    # Get max transition count for line thickness scaling
    if filtered_transitions:
        max_count = max(filtered_transitions.values())
        min_count = min(filtered_transitions.values())
        
        # Create 5 equal-width buckets for transition counts
        if max_count > min_count:
            bucket_width = max(1, (max_count - min_count) / 5)
            
            for i in range(5):
                bucket_min = int(min_count + i * bucket_width)
                bucket_max = int(min_count + (i + 1) * bucket_width)
                if i < 4:
                    line_thickness_ranges.append((bucket_min, bucket_max))
                else:
                    line_thickness_ranges.append((bucket_min, max_count + 1))  # Include max value
        else:
            # Only one bucket if all counts are the same
            line_thickness_ranges.append((min_count, max_count + 1))
        
        # Draw transitions as simplified lines
        for (src, dst), count in filtered_transitions.items():
            if src in region_coordinates and dst in region_coordinates:
                src_x, src_y = region_coordinates[src]
                dst_x, dst_y = region_coordinates[dst]
                
                # Scale coordinates
                src_x_scaled = src_x * img_width
                src_y_scaled = src_y * img_height
                dst_x_scaled = dst_x * img_width
                dst_y_scaled = dst_y * img_height
                
                # Calculate which bucket this count falls into
                bucket_idx = 0
                for i, (bucket_min, bucket_max) in enumerate(line_thickness_ranges):
                    if bucket_min <= count < bucket_max:
                        bucket_idx = i
                        break
                
                # Line width scales from 1 to 4 based on bucket
                line_width = 1 + bucket_idx * (3.0 / max(1, len(line_thickness_ranges) - 1))
                
                # Get color based on count
                color = transition_cmap(count / max_count)
                
                # Calculate if regions are adjacent
                is_adjacent = dst in name_connections.get(src, [])
                linestyle = '-' if is_adjacent else ':'
                
                # Draw line
                line = ax_map.plot([src_x_scaled, dst_x_scaled], [src_y_scaled, dst_y_scaled], 
                                  color=color, linewidth=line_width, linestyle=linestyle, alpha=0.7)[0]
                
                # Save for legend
                if bucket_idx >= len(line_thickness_legend):
                    line_thickness_legend.append({
                        'line': line,
                        'bucket_min': line_thickness_ranges[bucket_idx][0],
                        'bucket_max': line_thickness_ranges[bucket_idx][1] - 1 if bucket_idx < len(line_thickness_ranges) - 1 else max_count,
                    })
    
    # Create heatmap overlay for top regions
    cmap = plt.cm.get_cmap('YlOrRd')
    max_duration = region_durations.iloc[0]['duration'] if not region_durations.empty else 1
    
    # Draw regions as rectangles
    for _, row in top_regions_data.iterrows():
        region = row['region']
        duration = row['duration']
        
        if region in region_coordinates and region in region_dimensions:
            x, y = region_coordinates[region]
            width, height = region_dimensions[region]
            
            # Scale coordinates and dimensions to match image
            x_scaled = x * img_width - (width * img_width / 2)
            y_scaled = y * img_height - (height * img_height / 2)
            width_scaled = width * img_width
            height_scaled = height * img_height
            
            # Get color based on duration
            color = cmap(duration / max_duration)
            
            # Format duration as hh:mm:ss
            duration_formatted = format_seconds_to_hms(duration)
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_scaled, y_scaled),
                width_scaled, height_scaled,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax_map.add_patch(rect)
            
            # Add region label and duration
            ax_map.text(
                x * img_width, y * img_height,
                f"{region}\n{duration_formatted}",
                fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
    
    # Add title with total duration - use employee color
    ax_map.set_title(get_text('Region Heatmap: Employee {0} ({1}) - Total Duration: {2}', language).format(employee_id, get_text(dept, language), total_duration_formatted), 
                   fontsize=16, fontweight='bold', color=emp_color)
    
    # Turn off axis
    ax_map.axis('off')
    
    # Format table data
    table_data = [[get_text('Region', language), get_text('Duration', language), '%']]
    
    # Add data rows
    for _, row in top_regions_data.iterrows():
        table_data.append([
            row['region'], 
            row['formatted_time'],
            f"{row['percentage']:.1f}%"
        ])
    
    # Add total row for top regions
    top_regions_percentage = (top_regions_total_duration / total_duration * 100).round(1)
    table_data.append([
        get_text('Total', language), 
        top_regions_total_formatted,
        f"{top_regions_percentage:.1f}%"
    ])
    
    # Create table
    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.5, 0.3, 0.2]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Add header formatting
    for i in range(3):
        table[(0, i)].set_facecolor(emp_color)  # Use employee color for header
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Add total row formatting
    last_row = len(table_data) - 1
    for i in range(3):
        table[(last_row, i)].set_facecolor('#E8655F')
        table[(last_row, i)].set_text_props(fontweight='bold')
    
    # Add line thickness legend at the bottom of the figure
    if line_thickness_legend:
        # Create bins for the legend
        legend_ranges = []
        for item in line_thickness_legend:
            if item['bucket_min'] == item['bucket_max']:
                legend_ranges.append(get_text("{0} transitions", language).format(item['bucket_min']))
            else:
                legend_ranges.append(get_text("{0}-{1} transitions", language).format(item['bucket_min'], item['bucket_max']))
        
        # Create a colorbar for the transitions
        transition_legend_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
        
        # Create a gradient of colors for the colorbar
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        
        # Plot the gradient
        transition_legend_ax.imshow(gradient, aspect='auto', cmap=transition_cmap)
        
        # Customize the colorbar
        transition_legend_ax.set_yticks([])
        
        # Set custom ticks and labels based on the ranges
        tick_positions = np.linspace(0, 255, len(legend_ranges))
        transition_legend_ax.set_xticks(tick_positions)
        transition_legend_ax.set_xticklabels(legend_ranges)
        
        # Add a title to the legend
        transition_legend_ax.set_title(get_text('Transition Frequency', language), fontsize=10)
    
    # Add note about line styles
    plt.figtext(0.1, 0.01, get_text('Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions', language), 
              fontsize=10, ha='left')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig