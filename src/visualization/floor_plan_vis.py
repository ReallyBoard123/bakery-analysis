"""
Floor Plan Visualization Module

Creates visualizations overlaid on the bakery floor plan image.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from ..utils.time_utils import format_seconds_to_hms
from .base import save_figure, set_visualization_style

def create_region_heatmap(floor_plan_data, region_summary, save_path=None, min_percentage=1, figsize=(20, 14)):
    """
    Create a heatmap of region usage overlaid on the floor plan
    
    Parameters:
    -----------
    floor_plan_data : dict
        Dictionary containing floor plan data
    region_summary : pandas.DataFrame
        DataFrame with region usage summary
    save_path : str or Path, optional
        Path to save the visualization
    min_percentage : float, optional
        Minimum percentage of time to include in the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract floor plan and region data
    floor_plan = floor_plan_data['floor_plan']
    region_coordinates = floor_plan_data['region_coordinates']
    region_dimensions = floor_plan_data['region_dimensions']
    
    # Display the floor plan
    img = np.array(floor_plan)
    ax.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Create a dictionary of region durations
    region_durations = {}
    for _, row in region_summary.iterrows():
        region_durations[row['region']] = row['duration']
    
    # Find max duration for relative size comparison
    max_duration = max(region_durations.values()) if region_durations else 1
    
    # Create colormap
    cmap = plt.cm.get_cmap('YlOrRd')
    norm = plt.Normalize(vmin=0, vmax=max_duration)
    
    # Draw regions with color based on duration
    for region, (x, y) in region_coordinates.items():
        # Get region dimensions
        width, height = region_dimensions.get(region, (0.05, 0.05))
        
        # Scale coordinates and dimensions to match image
        x_scaled = x * img_width
        y_scaled = y * img_height
        width_scaled = width * img_width
        height_scaled = height * img_height
        
        # Get duration and percentage for this region
        duration = region_durations.get(region, 0)
        percentage = duration / max_duration * 100
        
        # Skip regions with very small percentages
        if percentage < min_percentage:
            continue
        
        # Format duration as hh:mm:ss
        duration_formatted = format_seconds_to_hms(duration)
        
        # Get color based on duration
        color = cmap(norm(duration))
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_scaled - width_scaled/2, y_scaled - height_scaled/2),
            width_scaled, height_scaled,
            linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add label with region name and duration
        ax.text(x_scaled, y_scaled, f"{region}\n{duration_formatted}",
                fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time Spent')
    
    # Set title and turn off axis labels
    ax.set_title('Heatmap of Time Spent in Different Regions', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def create_movement_flow(floor_plan_data, data, save_path=None, min_transitions=10, figsize=(20, 14)):
    """
    Create a visualization of movement flows between regions
    
    Parameters:
    -----------
    floor_plan_data : dict
        Dictionary containing floor plan data
    data : pandas.DataFrame
        Input dataframe with movement data
    save_path : str or Path, optional
        Path to save the visualization
    min_transitions : int, optional
        Minimum number of transitions to include in the visualization
    figsize : tuple, optional
        Figure size (width, height) in inches
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract floor plan and region data
    floor_plan = floor_plan_data['floor_plan']
    region_coordinates = floor_plan_data['region_coordinates']
    connections = floor_plan_data['connections']
    uuid_to_name = floor_plan_data['uuid_to_name']
    name_to_uuid = floor_plan_data['name_to_uuid']
    
    # Calculate transitions between regions
    transitions = []
    for emp_id, group in data.sort_values(['id', 'startTime']).groupby('id'):
        prev_region = None
        for _, row in group.iterrows():
            if prev_region and prev_region != row['region']:
                transitions.append((prev_region, row['region']))
            prev_region = row['region']
    
    # Count transitions
    from collections import Counter
    transition_counts = Counter(transitions)
    
    # Filter to minimum transitions
    filtered_transitions = {t: count for t, count in transition_counts.items() if count >= min_transitions}
    
    # Display the floor plan
    img = np.array(floor_plan)
    ax.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Draw transitions as arrows
    max_count = max(filtered_transitions.values()) if filtered_transitions else 1
    cmap = plt.cm.plasma
    
    for (source, target), count in filtered_transitions.items():
        # Get coordinates
        if source in region_coordinates and target in region_coordinates:
            source_x, source_y = region_coordinates[source]
            target_x, target_y = region_coordinates[target]
            
            # Scale coordinates
            source_x = source_x * img_width
            source_y = source_y * img_height
            target_x = target_x * img_width
            target_y = target_y * img_height
            
            # Calculate line width and color based on count
            width = 1 + 5 * (count / max_count)
            color = cmap(count / max_count)
            
            # Check if transition is valid based on connections
            source_uuid = name_to_uuid.get(source)
            target_uuid = name_to_uuid.get(target)
            is_valid = target_uuid in connections.get(source_uuid, []) if source_uuid and target_uuid else False
            
            # Draw arrow
            ax.annotate('', 
                      xy=(target_x, target_y), 
                      xytext=(source_x, source_y),
                      arrowprops=dict(
                          arrowstyle='->', 
                          lw=width,
                          color=color if is_valid else 'red',
                          alpha=0.6,
                          connectionstyle='arc3,rad=0.1'
                      ))
            
            # Add count label
            mid_x = (source_x + target_x) / 2
            mid_y = (source_y + target_y) / 2
            ax.text(mid_x, mid_y, str(count),
                  fontsize=9, ha='center', va='center',
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_transitions, vmax=max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Transition Count')
    
    # Set title and turn off axis labels
    ax.set_title('Movement Flows Between Regions', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig