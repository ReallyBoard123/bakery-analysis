"""
Floor Plan Visualization Module

Creates visualizations overlaid on the bakery floor plan image, including
movement paths, region heatmaps, and employee-specific visualizations.

Features:
- Configurable display parameters for customizing visualizations
- Option to hide region names and show only time durations
- Minimum duration threshold for displaying region information
- Line thickness for representing transition frequency instead of text labels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
import re
from datetime import timedelta
from matplotlib.lines import Line2D

# Configuration parameters that can be adjusted
CONFIG = {
    # General display settings
    'show_region_names': False,         # Whether to show region names on visualizations
    'min_duration_threshold': 3,        # Minimum duration (in seconds) to display a region's info
    'min_region_relative_size': 0.05,   # Minimum relative region size (compared to max) to display info
    
    # Region heatmap settings
    'heatmap_alpha': 0.7,               # Transparency of region colors (0-1)
    'heatmap_text_size': 8,             # Font size for region text
    
    # Movement visualization settings
    'movement_line_min_width': 1,       # Minimum line width for transitions
    'movement_line_max_width': 6,       # Maximum line width for transitions
    'transition_count_divider': 10,     # Divider for scaling transition counts to line width
    
    # Employee path visualization settings
    'path_circle_base_size': 50,        # Base size for region circles
    'path_circle_scale_factor': 250,    # Scaling factor for region circles based on time spent
    'min_transition_count': 4,          # Minimum number of transitions to display
    'path_circle_alpha': 0.5,           # Transparency of region circles (0-1)
    
    # Color settings
    'region_cmap': 'YlOrRd',            # Colormap for regions
    'transition_cmap': 'plasma',        # Colormap for transitions
    'region_circle_color': '#FFAA00',   # Color for region circles
    'region_circle_edge': '#E8655F',    # Edge color for region circles
    'text_box_color': 'white',          # Background color for text boxes
    'text_box_alpha': 0.7,              # Transparency of text boxes (0-1)
    
    # Path color palette
    'path_colors': [
        '#2D5F91',  # Dark blue
        '#48B7B4',  # Teal
        '#E8655F',  # Coral
        '#FFAA00',  # Amber
        '#96BEE6'   # Light blue
    ]
}

def create_floor_plan_visualizations(data_dict, results, vis_dir, config=None):
    """
    Create visualizations overlaid on the bakery floor plan
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing processed data and metadata
    results : dict
        Dictionary containing analysis results
    vis_dir : Path
        Directory to save visualizations
    config : dict, optional
        Configuration parameters to override defaults
    """
    # Update configuration with any provided parameters
    cfg = CONFIG.copy()
    if config:
        cfg.update(config)
    
    # Get required data
    layout_path = data_dict.get('layout_path')
    region_coordinates = data_dict.get('region_coordinates', {})
    region_dimensions = data_dict.get('region_dimensions', {})
    processed_data = data_dict.get('processed_data')
    
    # Ensure visualization directory exists
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(exist_ok=True)
    
    # Create employee path subdirectory
    emp_path_dir = vis_dir / "employee_paths"
    emp_path_dir.mkdir(exist_ok=True)
    
    # Check if floor plan image exists
    try:
        floor_plan = Image.open(layout_path)
        print(f"  Loading floor plan image: {layout_path}")
    except Exception as e:
        print(f"  Error loading floor plan image: {e}")
        print("  Skipping floor plan visualizations")
        return
    
    # Create movement visualization
    create_movement_floor_plan(floor_plan, region_coordinates, region_dimensions, 
                              results['movement']['valid_transitions'], vis_dir, cfg)
    
    # Create region heatmap
    create_region_heatmap(floor_plan, region_coordinates, region_dimensions,
                         results['activity']['region_summary'], vis_dir, cfg)
    
    # Create bottleneck visualization
    create_bottleneck_floor_plan(floor_plan, region_coordinates, region_dimensions,
                                results['bottlenecks']['bottlenecks'], vis_dir, cfg)
    
    # Create employee path visualizations by source file
    create_employee_path_by_source(floor_plan, region_coordinates, 
                                 processed_data, emp_path_dir, cfg)
    
    # Export invalid transitions to CSV
    export_invalid_transitions(results['movement']['invalid_transitions'], vis_dir)

def create_movement_floor_plan(floor_plan, region_coordinates, region_dimensions, transitions, vis_dir, cfg):
    """
    Create a visualization of movement patterns overlaid on the floor plan
    """
    print("  Creating movement floor plan visualization...")
    
    # Create figure
    plt.figure(figsize=(20, 14))
    
    # Display the floor plan
    img = np.array(floor_plan)
    plt.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    normalized_coords = {}
    normalized_dims = {}
    
    for region, (x, y) in region_coordinates.items():
        # Scale coordinates to match image dimensions
        normalized_coords[region] = (x * img_width, y * img_height)
        
        width, height = region_dimensions.get(region, (0.05, 0.05))
        normalized_dims[region] = (width * img_width, height * img_height)
    
    # Draw movement paths with width based on frequency
    lines = []
    colors = []
    widths = []
    
    for (from_region, to_region), count in transitions.items():
        if from_region in normalized_coords and to_region in normalized_coords:
            start = normalized_coords[from_region]
            end = normalized_coords[to_region]
            
            lines.append([start, end])
            colors.append(count)
            widths.append(max(cfg['movement_line_min_width'], 
                             min(cfg['movement_line_max_width'], 
                                count / cfg['transition_count_divider'])))
    
    # Create a LineCollection for efficient line plotting
    if lines:
        lc = LineCollection(lines, linewidths=widths, 
                          cmap=cfg['transition_cmap'], 
                          norm=plt.Normalize(min(colors), max(colors)))
        lc.set_array(np.array(colors))
        plt.gca().add_collection(lc)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=plt.gca())
        cbar.set_label('Transition Frequency')
    
    # Add region labels if configured to show them
    if cfg['show_region_names']:
        for region, (x, y) in normalized_coords.items():
            width, height = normalized_dims.get(region, (10, 10))
            
            # Only label sufficiently large regions
            if width > img_width * 0.03 and height > img_height * 0.03:
                plt.text(x, y, region, fontsize=cfg['heatmap_text_size'], ha='center', va='center',
                       bbox=dict(facecolor=cfg['text_box_color'], alpha=cfg['text_box_alpha'], edgecolor='none'))
    
    # Set title and labels
    plt.title('Bakery Employee Movement Patterns', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(vis_dir / 'movement_floor_plan.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_region_heatmap(floor_plan, region_coordinates, region_dimensions, region_summary, vis_dir, cfg):
    """
    Create a heatmap of region usage overlaid on the floor plan
    """
    print("  Creating region heatmap visualization...")
    
    # Create figure
    plt.figure(figsize=(20, 14))
    
    # Display the floor plan
    img = np.array(floor_plan)
    plt.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Create a dictionary of region durations
    region_durations = {}
    for _, row in region_summary.iterrows():
        region_durations[row['region']] = row['sum']
    
    # Find max duration for relative size comparison
    max_duration = max(region_durations.values()) if region_durations else 1
    
    # Create colormap
    cmap = plt.cm.get_cmap(cfg['region_cmap'])
    norm = plt.Normalize(vmin=min(region_durations.values()) if region_durations else 0,
                       vmax=max(region_durations.values()) if region_durations else 1)
    
    # Draw regions with color based on duration
    for region, (x, y) in region_coordinates.items():
        # Get region dimensions
        width, height = region_dimensions.get(region, (0.05, 0.05))
        
        # Scale coordinates and dimensions to match image
        x_scaled = x * img_width
        y_scaled = y * img_height
        width_scaled = width * img_width
        height_scaled = height * img_height
        
        # Get duration for this region
        duration = region_durations.get(region, 0)
        
        # Skip regions with very short durations if configured
        if duration < cfg['min_duration_threshold']:
            continue
        
        # Format duration as hh:mm:ss
        duration_formatted = str(timedelta(seconds=int(duration)))
        
        # Get color based on duration
        color = cmap(norm(duration))
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_scaled - width_scaled/2, y_scaled - height_scaled/2),
            width_scaled, height_scaled,
            linewidth=1, edgecolor='black', facecolor=color, alpha=cfg['heatmap_alpha']
        )
        plt.gca().add_patch(rect)
        
        # Add label if region is large enough and duration is significant
        if (width > 0.03 and height > 0.03 and 
            duration/max_duration > cfg['min_region_relative_size']):
            
            # Format text based on configuration
            if cfg['show_region_names']:
                label_text = f"{region}\n{duration_formatted}"
            else:
                label_text = f"{duration_formatted}"
            
            plt.text(x_scaled, y_scaled, label_text, 
                   fontsize=cfg['heatmap_text_size'], ha='center', va='center',
                   bbox=dict(facecolor=cfg['text_box_color'], alpha=cfg['text_box_alpha'], edgecolor='none'))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Time Spent')
    
    # Set title and labels
    plt.title('Heatmap of Time Spent in Different Regions', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(vis_dir / 'region_heatmap_floor_plan.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_bottleneck_floor_plan(floor_plan, region_coordinates, region_dimensions, bottlenecks, vis_dir, cfg):
    """
    Visualize bottlenecks on the floor plan
    """
    print("  Creating bottleneck visualization...")
    
    # Create figure
    plt.figure(figsize=(20, 14))
    
    # Display the floor plan
    img = np.array(floor_plan)
    plt.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Create a dictionary of bottleneck scores
    bottleneck_scores = {}
    for region, metrics in bottlenecks:
        bottleneck_scores[region] = metrics['bottleneck_score']
    
    # Find min and max scores for colormap normalization
    min_score = min(bottleneck_scores.values()) if bottleneck_scores else 0
    max_score = max(bottleneck_scores.values()) if bottleneck_scores else 1
    
    # Create colormap
    cmap = plt.cm.get_cmap(cfg['region_cmap'])
    norm = plt.Normalize(vmin=min_score, vmax=max_score)
    
    # Draw regions with color based on bottleneck score
    for region, (x, y) in region_coordinates.items():
        # Get region dimensions
        width, height = region_dimensions.get(region, (0.05, 0.05))
        
        # Scale coordinates and dimensions to match image
        x_scaled = x * img_width
        y_scaled = y * img_height
        width_scaled = width * img_width
        height_scaled = height * img_height
        
        # Get bottleneck score for this region
        score = bottleneck_scores.get(region, 0)
        
        # Skip regions with very low scores (to focus on bottlenecks)
        if score < min_score + (max_score - min_score) * 0.2:
            continue
        
        # Get color based on score
        color = cmap(norm(score))
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_scaled - width_scaled/2, y_scaled - height_scaled/2),
            width_scaled, height_scaled,
            linewidth=1, edgecolor='black', facecolor=color, alpha=cfg['heatmap_alpha']
        )
        plt.gca().add_patch(rect)
        
        # Add label if region is large enough
        if width > 0.03 and height > 0.03:
            # Format text based on configuration
            if cfg['show_region_names']:
                label_text = f"{region}\nScore: {score:.1f}"
            else:
                label_text = f"Score: {score:.1f}"
                
            plt.text(x_scaled, y_scaled, label_text, 
                   fontsize=cfg['heatmap_text_size'], ha='center', va='center',
                   bbox=dict(facecolor=cfg['text_box_color'], alpha=cfg['text_box_alpha'], edgecolor='none'))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Bottleneck Score')
    
    # Set title and labels
    plt.title('Bottleneck Regions in the Bakery', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(vis_dir / 'bottleneck_floor_plan.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_employee_path_by_source(floor_plan, region_coordinates, data, vis_dir, cfg):
    """
    Create visualizations of employee paths by source file, with line thickness
    representing transition frequency. Additionally, create combined visualizations
    for each employee across all their shifts, focusing on the top 6 regions.
    """
    print("  Creating employee path visualizations by source file...")

    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    normalized_coords = {}

    for region, (x, y) in region_coordinates.items():
        normalized_coords[region] = (x * img_width, y * img_height)

    # Group data by source file
    source_files = data['source_file'].unique()
    print(f"  Found {len(source_files)} unique source files")

    # Extract shift information
    shift_info = get_shift_info(data)

    # Create a visualization for each source file
    for source_file in source_files:
        # Extract employee ID from source file
        employee_id_match = re.match(r'(\d+\-[A-Z])', source_file)
        if not employee_id_match:
            continue

        employee_id = employee_id_match.group(1)

        # Extract date from source file
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', source_file)
        date_str = date_match.group(1) if date_match else "Unknown"

        # Get shift info
        shift_start, shift_end = shift_info.get(source_file, ("Unknown", "Unknown"))

        print(f"    Creating path visualization for {source_file}")

        plt.figure(figsize=(16, 10))

        # Display the floor plan
        img = np.array(floor_plan)
        plt.imshow(img)

        # Get source file data
        source_data = data[data['source_file'] == source_file].sort_values('startTime')

        # Skip if no data
        if len(source_data) == 0:
            plt.close()
            continue

        # Calculate session duration using actual data
        if len(source_data) >= 2:
            # Use the actual first and last entries from the source file data
            start_time = source_data.iloc[0]['startTimeClock']
            end_time = source_data.iloc[-1]['endTimeClock']

            # Calculate total duration in a readable format
            total_duration_seconds = source_data['duration'].sum()
            total_duration = str(timedelta(seconds=int(total_duration_seconds)))

            session_info = f"Session: {start_time} to {end_time} | Total: {total_duration}"
        else:
            session_info = "Session: Unknown"

        # Count time spent in each region
        region_time = source_data.groupby('region')['duration'].sum()

        # Find max time for relative size comparison
        max_time = region_time.max() if not region_time.empty else 1

        # Track transitions between regions and count them
        transitions = []
        transition_counts = Counter()
        prev_region = None

        for _, row in source_data.iterrows():
            if prev_region and prev_region != row['region']:
                transitions.append((prev_region, row['region']))
                transition_counts[(prev_region, row['region'])] += 1
            prev_region = row['region']

        # Calculate max transition count for line width scaling
        max_count = max(transition_counts.values()) if transition_counts else 1

        # Sort transitions by count in descending order to ensure thicker lines are drawn on top
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)

        # Plot transitions with line thickness representing count
        for (from_region, to_region), count in sorted_transitions:
            # Skip transitions with counts below threshold
            if count < cfg['min_transition_count']:
                continue

            if from_region in normalized_coords and to_region in normalized_coords:
                start = normalized_coords[from_region]
                end = normalized_coords[to_region]

                # Calculate normalized line width (between 1 and 5)
                line_width = cfg['movement_line_min_width'] + (cfg['movement_line_max_width'] - cfg['movement_line_min_width']) * (count / max_count)

                # Calculate color index based on count
                color_idx = min(int(count / max_count * (len(cfg['path_colors']) - 1)), len(cfg['path_colors']) - 1)
                line_color = cfg['path_colors'][color_idx]

                # Assign z-order based on count (higher count gets higher z-order)
                z_order = 1 + count / max_count

                # Draw arrow with higher z-order for more frequent transitions
                plt.annotate('', 
                          xy=end, 
                          xytext=start,
                          arrowprops=dict(
                              arrowstyle='->', 
                              lw=line_width,
                              color=line_color,
                              alpha=0.8,
                              zorder=z_order
                          ))

        # Draw circles for regions, with size based on time spent
        for region, (x, y) in normalized_coords.items():
            if region in region_time:
                time = region_time[region]

                # Skip regions with durations below threshold
                if time < cfg['min_duration_threshold'] or time/max_time < cfg['min_region_relative_size']:
                    continue

                # Scale circle size
                size = cfg['path_circle_base_size'] + cfg['path_circle_scale_factor'] * (time / max_time)

                # Format time as hh:mm:ss
                time_formatted = str(timedelta(seconds=int(time)))

                # Use a slightly transparent circle
                plt.scatter(x, y, s=size, alpha=cfg['path_circle_alpha'], 
                          color=cfg['region_circle_color'], 
                          edgecolor=cfg['region_circle_edge'], linewidth=1)

                # Format label based on configuration
                if cfg['show_region_names']:
                    label_text = f"{region}\n{time_formatted}"
                else:
                    label_text = f"{time_formatted}"

                # Add label
                plt.text(x, y, label_text, fontsize=cfg['heatmap_text_size'], ha='center', va='center',
                       bbox=dict(facecolor=cfg['text_box_color'], alpha=cfg['text_box_alpha'], edgecolor='none'))

        # Create legend for line thickness
        legend_elements = [
            Line2D([0], [0], color='gray', lw=cfg['movement_line_min_width'], label='Low frequency'),
            Line2D([0], [0], color='red', lw=cfg['movement_line_max_width'], label='High frequency')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Set title and labels
        plt.title(f'Movement Path for {employee_id} - {date_str}\n'
                 f'Shift: {shift_start} to {shift_end}\n{session_info}', fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        # Create a safe filename
        safe_filename = source_file.replace(':', '_').replace('/', '_').replace('\\', '_')

        # Save visualization
        plt.savefig(vis_dir / f'{safe_filename}_path.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Create combined visualizations for each employee
    print("  Creating combined visualizations for each employee...")
    employees = data['id'].unique()

    for employee_id in employees:
        print(f"    Creating combined path visualization for {employee_id}")

        plt.figure(figsize=(16, 10))

        # Display the floor plan
        img = np.array(floor_plan)
        plt.imshow(img)

        # Get all data for the employee
        employee_data = data[data['id'] == employee_id].sort_values('startTime')

        # Skip if no data
        if len(employee_data) == 0:
            plt.close()
            continue

        # Count time spent in each region
        region_time = employee_data.groupby('region')['duration'].sum()

        # Select the top 6 regions by time spent
        top_regions = region_time.nlargest(6).index

        # Filter data to include only the top regions
        employee_data = employee_data[employee_data['region'].isin(top_regions)]

        # Find max time for relative size comparison
        max_time = region_time[top_regions].max() if not region_time.empty else 1

        # Track transitions between regions and count them
        transitions = []
        transition_counts = Counter()
        prev_region = None

        for _, row in employee_data.iterrows():
            if prev_region and prev_region != row['region']:
                transitions.append((prev_region, row['region']))
                transition_counts[(prev_region, row['region'])] += 1
            prev_region = row['region']

        # Calculate max transition count for line width scaling
        max_count = max(transition_counts.values()) if transition_counts else 1

        # Plot transitions with line thickness representing count
        for (from_region, to_region), count in transition_counts.items():
            # Skip transitions with counts below threshold
            if count < cfg['min_transition_count']:
                continue

            if from_region in normalized_coords and to_region in normalized_coords:
                start = normalized_coords[from_region]
                end = normalized_coords[to_region]

                # Calculate normalized line width (between 1 and 5)
                line_width = cfg['movement_line_min_width'] + (cfg['movement_line_max_width'] - cfg['movement_line_min_width']) * (count / max_count)

                # Calculate color index based on count
                color_idx = min(int(count / max_count * (len(cfg['path_colors']) - 1)), len(cfg['path_colors']) - 1)
                line_color = cfg['path_colors'][color_idx]

                # Draw arrow with higher z-order for more frequent transitions
                plt.annotate('', 
                          xy=end, 
                          xytext=start,
                          arrowprops=dict(
                              arrowstyle='->', 
                              lw=line_width,
                              color=line_color,
                              alpha=0.8,
                              zorder=2 if count == max_count else 1
                          ))

        # Draw circles for regions, with size based on time spent
        for region, (x, y) in normalized_coords.items():
            if region in region_time:
                time = region_time[region]

                # Skip regions with durations below threshold
                if time < cfg['min_duration_threshold'] or time/max_time < cfg['min_region_relative_size']:
                    continue

                # Scale circle size
                size = cfg['path_circle_base_size'] + cfg['path_circle_scale_factor'] * (time / max_time)

                # Format time as hh:mm:ss
                time_formatted = str(timedelta(seconds=int(time)))

                # Use a slightly transparent circle
                plt.scatter(x, y, s=size, alpha=cfg['path_circle_alpha'], 
                          color=cfg['region_circle_color'], 
                          edgecolor=cfg['region_circle_edge'], linewidth=1)

                # Format label based on configuration
                if cfg['show_region_names']:
                    label_text = f"{region}\n{time_formatted}"
                else:
                    label_text = f"{time_formatted}"

                # Add label
                plt.text(x, y, label_text, fontsize=cfg['heatmap_text_size'], ha='center', va='center',
                       bbox=dict(facecolor=cfg['text_box_color'], alpha=cfg['text_box_alpha'], edgecolor='none'))

        # Set title and labels
        plt.title(f'Combined Movement Path for {employee_id} (Top 6 Regions)', fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        # Save visualization
        plt.savefig(vis_dir / f'{employee_id}_combined_path_top6.png', dpi=300, bbox_inches='tight')
        plt.close()

def get_shift_info(data):
    """
    Extract shift information from data
    
    Returns:
    --------
    dict
        Dictionary mapping source files to (shift_start, shift_end) tuples
    """
    shift_info = {}
    
    # Default shift time (from the metadata - night shift)
    default_start = "22:00:00"
    default_end = "13:00:00"
    
    # Group data by source file
    for source_file in data['source_file'].unique():
        shift_info[source_file] = (default_start, default_end)
    
    return shift_info

def export_invalid_transitions(invalid_transitions, vis_dir):
    """
    Export invalid transitions to CSV
    
    Parameters:
    -----------
    invalid_transitions : dict
        Dictionary of invalid transitions with counts
    vis_dir : Path
        Directory to save the CSV file
    """
    print("  Exporting invalid transitions to CSV...")
    
    # Create data for CSV
    invalid_data = []
    for (from_region, to_region), count in invalid_transitions.items():
        invalid_data.append({
            'from_region': from_region,
            'to_region': to_region,
            'count': count
        })
    
    # Convert to DataFrame
    invalid_df = pd.DataFrame(invalid_data)
    
    # Sort by count (highest first)
    invalid_df = invalid_df.sort_values('count', ascending=False)
    
    # Save to CSV
    csv_path = vis_dir / 'invalid_transitions.csv'
    invalid_df.to_csv(csv_path, index=False)
    
    print(f"  Saved {len(invalid_transitions)} invalid transitions to {csv_path}")

# Example of how to use the configuration to customize visualizations
def create_custom_visualizations(data_dict, results, vis_dir):
    """
    Example of creating visualizations with custom configuration
    """
    custom_config = {
        'show_region_names': False,
        'min_duration_threshold': 5,
        'min_region_relative_size': 0.1,
        'heatmap_alpha': 0.8,
        'path_circle_base_size': 30,
        'min_transition_count': 50
    }
    
    create_floor_plan_visualizations(data_dict, results, vis_dir, custom_config)