"""
Transition Visualization Module

Functions for creating visualizations of employee transitions between regions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from collections import Counter

from .base import save_figure, set_visualization_style, get_color_palette

def create_employee_transition_visualization(data, floor_plan_data, employee_id, shift=None, 
                                            save_path=None, min_transitions=3, figsize=(16, 12)):
    """
    Create visualization of region transitions for a specific employee
    
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
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter data for the specified employee and shift
    filtered_data = data[(data['id'] == employee_id)]
    if shift is not None:
        filtered_data = filtered_data[filtered_data['shift'] == shift]
    
    # Extract floor plan elements
    floor_plan = floor_plan_data['floor_plan']
    region_coordinates = floor_plan_data['region_coordinates']
    name_connections = floor_plan_data['name_connections']
    
    # Calculate transitions
    transitions = []
    prev_region = None
    for _, row in filtered_data.sort_values('startTime').iterrows():
        if prev_region and prev_region != row['region']:
            transitions.append((prev_region, row['region']))
        prev_region = row['region']
    
    # Count transitions
    transition_counts = Counter(transitions)
    
    # Filter to transitions with count >= min_transitions
    filtered_transitions = {t: count for t, count in transition_counts.items() if count >= min_transitions}
    
    if not filtered_transitions:
        print(f"No transitions with at least {min_transitions} occurrences for {employee_id}" + 
              (f", shift {shift}" if shift is not None else ""))
        return None
    
    # Display the floor plan
    img = np.array(floor_plan)
    ax.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Identify regions involved in transitions
    regions_involved = set()
    for (src, dst) in filtered_transitions.keys():
        regions_involved.add(src)
        regions_involved.add(dst)
    
    # Draw regions with less opacity to highlight transitions
    for region in regions_involved:
        if region in region_coordinates:
            x, y = region_coordinates[region]
            x_scaled = x * img_width
            y_scaled = y * img_height
            
            # Create circle for region
            circle = plt.Circle(
                (x_scaled, y_scaled),
                radius=25,  # Larger radius for better visibility
                color='#2D5F91',
                alpha=0.5  # Semi-transparent
            )
            ax.add_patch(circle)
            
            # Add region label
            ax.text(
                x_scaled, y_scaled,
                region,
                ha='center',
                va='center',
                color='white',
                fontsize=8,
                fontweight='bold'
            )
    
    # Custom colormap for transitions
    transition_cmap = LinearSegmentedColormap.from_list(
        'transition_cmap', 
        ['#87CEEB', '#44A4DF', '#FFD700', '#FF8C00', '#FF4500']
    )
    
    # Create color mapping for transitions based on count
    max_count = max(filtered_transitions.values())
    
    # Draw arrows for transitions
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
            width = 2 + (count / max_count) * 5  # Width between 2 and 7
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
                alpha=0.8,
                zorder=10  # Ensure arrows are on top
            )
            
            # Add count label at midpoint
            mid_x = (src_x_scaled + dst_x_scaled) / 2
            mid_y = (src_y_scaled + dst_y_scaled) / 2
            ax.text(
                mid_x, mid_y,
                str(count),
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                zorder=11  # Ensure text is on top of arrows
            )
    
    # Add title and department info
    dept = filtered_data['department'].iloc[0] if 'department' in filtered_data.columns else 'Unknown'
    title = f'Region Transitions: Employee {employee_id} ({dept})'
    if shift is not None:
        title += f' - Shift {shift}'
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Turn off axis
    ax.axis('off')
    
    # Add legend for transition counts
    sm = plt.cm.ScalarMappable(cmap=transition_cmap, norm=plt.Normalize(vmin=min_transitions, vmax=max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Transition Count', orientation='vertical', shrink=0.8)
    
    # Add note about line styles
    plt.figtext(0.01, 0.01, "Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions", 
               fontsize=10, ha='left')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_movement_transitions_chart(transition_counts, save_path=None, top_n=10, figsize=(15, 10)):
    """
    Plot top movement transitions as horizontal bar charts
    
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
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    set_visualization_style()
    fig, axes = plt.subplots(len(transition_counts), 1, figsize=figsize, sharex=True)
    
    # Ensure axes is always a list
    if len(transition_counts) == 1:
        axes = [axes]
    
    for i, (emp_id, counts) in enumerate(transition_counts.items()):
        # Get top transitions
        top_transitions = counts.sort_values(ascending=False).head(top_n)
        
        # Create transition labels
        labels = [f"{src} → {dst}" for (src, dst) in top_transitions.index]
        
        # Plot horizontal bar chart
        bars = axes[i].barh(labels, top_transitions.values, color=get_color_palette()[i % len(get_color_palette())])
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            axes[i].text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f"{width:.0f}", ha='left', va='center', fontweight='bold')
        
        axes[i].set_title(f'Top {top_n} Movement Transitions - Employee {emp_id}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Number of Transitions', fontsize=12)
        axes[i].set_ylabel('Transition Path', fontsize=12)
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig