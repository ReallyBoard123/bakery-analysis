"""
Employee Region Visualization Module

Function for creating employee-specific region heatmaps with transition visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from collections import Counter

from .base import save_figure, set_visualization_style, get_color_palette
from ..utils.time_utils import format_seconds_to_hms

def create_employee_region_heatmap(data, floor_plan_data, employee_id, save_path=None, 
                                  top_n=10, min_transitions=2, figsize=(16, 12)):
    """
    Create employee-specific region heatmap with transition lines
    
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
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter data for the specified employee
    employee_data = data[data['id'] == employee_id]
    if employee_data.empty:
        print(f"No data found for employee {employee_id}")
        return None
    
    # Extract floor plan elements
    floor_plan = floor_plan_data['floor_plan']
    region_coordinates = floor_plan_data['region_coordinates']
    name_connections = floor_plan_data['name_connections']
    
    # Calculate time spent in each region
    region_durations = employee_data.groupby('region')['duration'].sum().reset_index()
    region_durations = region_durations.sort_values('duration', ascending=False)
    
    # Get top N regions by duration
    top_regions = region_durations.head(top_n)['region'].tolist()
    
    # Calculate transitions
    transitions = []
    prev_region = None
    for _, row in employee_data.sort_values('startTime').iterrows():
        if prev_region and prev_region != row['region']:
            transitions.append((prev_region, row['region']))
        prev_region = row['region']
    
    # Count transitions
    transition_counts = Counter(transitions)
    
    # Filter to transitions involving top regions and with count >= min_transitions
    filtered_transitions = {t: count for t, count in transition_counts.items() 
                          if (t[0] in top_regions or t[1] in top_regions) and count >= min_transitions}
    
    # Display the floor plan
    img = np.array(floor_plan)
    ax.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Create a custom colormap for transitions
    transition_cmap = LinearSegmentedColormap.from_list(
        'transition_cmap', 
        ['#87CEEB', '#44A4DF', '#FFD700', '#FF8C00', '#FF4500']
    )
    
    # Create a custom colormap for regions
    region_cmap = plt.cm.get_cmap('YlOrRd')
    
    # Calculate max values for normalization
    max_count = max(filtered_transitions.values()) if filtered_transitions else 1
    
    # Get top regions data
    top_regions_data = region_durations[region_durations['region'].isin(top_regions)]
    
    # Calculate max duration for normalization
    max_duration = top_regions_data['duration'].max() if not top_regions_data.empty else 1
    
    # Draw transitions first (lower z-index)
    for (src, dst), count in filtered_transitions.items():
        if src in region_coordinates and dst in region_coordinates:
            src_x, src_y = region_coordinates[src]
            dst_x, dst_y = region_coordinates[dst]
            
            # Scale coordinates
            src_x_scaled = src_x * img_width
            src_y_scaled = src_y * img_height
            dst_x_scaled = dst_x * img_width
            dst_y_scaled = dst_y * img_height
            
            # Calculate arrow properties based on count
            width = 1 + 4 * (count / max_count)  # Width between 1 and 5
            color = transition_cmap(count / max_count)
            
            # Calculate if regions are adjacent
            is_adjacent = dst in name_connections.get(src, [])
            linestyle = '-' if is_adjacent else ':'
            
            # Draw arrow
            ax.arrow(
                src_x_scaled, src_y_scaled,
                (dst_x_scaled - src_x_scaled) * 0.8, (dst_y_scaled - src_y_scaled) * 0.8,
                width=width,
                head_width=width * 3,
                head_length=width * 5,
                length_includes_head=True,
                color=color,
                linestyle=linestyle,
                alpha=0.7,
                zorder=5  # Lower z-index
            )
    
    # Draw regions (higher z-index)
    for _, row in top_regions_data.iterrows():
        region = row['region']
        duration = row['duration']
        
        if region in region_coordinates:
            x, y = region_coordinates[region]
            x_scaled = x * img_width
            y_scaled = y * img_height
            
            # Calculate circle size based on duration
            size = 20 + 40 * (duration / max_duration)  # Size between 20 and 60
            
            # Get color based on duration
            color = region_cmap(duration / max_duration)
            
            # Create circle for region
            circle = plt.Circle(
                (x_scaled, y_scaled),
                radius=size,
                color=color,
                alpha=0.8,
                zorder=10  # Higher z-index
            )
            ax.add_patch(circle)
            
            # Format duration as hh:mm:ss
            duration_formatted = format_seconds_to_hms(duration)
            
            # Add region label and duration
            ax.text(
                x_scaled, y_scaled,
                f"{region}\n{duration_formatted}",
                ha='center',
                va='center',
                color='white',
                fontsize=10,
                fontweight='bold',
                zorder=15  # Highest z-index
            )
    
    # Add title and employee info
    dept = employee_data['department'].iloc[0] if 'department' in employee_data.columns else 'Unknown'
    ax.set_title(f'Region Heatmap: Employee {employee_id} ({dept}) - Top {len(top_regions_data)} Regions', 
                fontsize=16, fontweight='bold')
    
    # Add legend for transition counts
    sm = plt.cm.ScalarMappable(cmap=transition_cmap, norm=plt.Normalize(vmin=min_transitions, vmax=max_count))
    sm.set_array([])
    cbar_trans = plt.colorbar(sm, ax=ax, label='Transition Count', orientation='vertical', 
                           fraction=0.046, pad=0.04, shrink=0.6)
    cbar_trans.ax.set_ylabel('Transition Count', fontsize=12)
    
    # Add legend for region durations
    sm2 = plt.cm.ScalarMappable(cmap=region_cmap, norm=plt.Normalize(vmin=0, vmax=max_duration))
    sm2.set_array([])
    cbar_reg = plt.colorbar(sm2, ax=ax, label='Time Spent (seconds)', orientation='vertical', 
                         fraction=0.046, pad=0.12, shrink=0.6)
    cbar_reg.ax.set_ylabel('Time Spent (seconds)', fontsize=12)
    
    # Add note about line styles
    plt.figtext(0.01, 0.01, "Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions", 
               fontsize=10, ha='left')
    
    # Turn off axis
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig