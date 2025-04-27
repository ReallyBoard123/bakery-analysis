"""
Floor Plan Visualization Module - Streamlined for Employee Path Analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path
import pandas as pd
import re
from datetime import timedelta
from matplotlib.lines import Line2D

# Configuration parameters
CONFIG = {
    # Display settings
    'show_region_names': True,
    'min_duration_threshold': 3,
    'min_region_relative_size': 0.05,
    'heatmap_text_size': 8,
    
    # Movement visualization settings
    'movement_line_min_width': 1,
    'movement_line_max_width': 6,
    
    # Employee path visualization settings
    'path_circle_base_size': 50,
    'path_circle_scale_factor': 250,
    'min_transition_count': 4,
    'path_circle_alpha': 0.5,
    
    # Color settings
    'region_cmap': 'YlOrRd',
    'text_box_color': 'white',
    'text_box_alpha': 0.7,
    
    # Path color palette
    'path_colors': [
        '#2D5F91',  # Dark blue
        '#48B7B4',  # Teal
        '#E8655F',  # Coral
        '#FFAA00',  # Amber
        '#96BEE6'   # Light blue
    ],
    
    # Combined visualization settings
    'top_regions_count': 6,
    'show_transition_table': True,
    'combine_bidirectional': True
}

def create_combined_path_visualization(employee_id, employee_data, floor_plan, 
                                     normalized_coords, name_connections, 
                                     top_regions, output_dir, cfg):
    """Create a combined path visualization focusing on paths between top regions."""
    plt.figure(figsize=(16, 10))
    
    # Display the floor plan
    img = np.array(floor_plan)
    plt.imshow(img)
    
    # Filter data to include only the top regions
    filtered_data = employee_data[employee_data['region'].isin(top_regions)]
    
    # Calculate time spent in each region for scaling node sizes
    region_time = filtered_data.groupby('region')['duration'].sum()
    max_time = region_time.max() if not region_time.empty else 1
    
    # Track transitions between regions and count them
    transitions = []
    bidirectional_transitions = {}
    prev_region = None
    
    for _, row in filtered_data.sort_values('startTime').iterrows():
        if prev_region and prev_region != row['region']:
            # Create a sorted tuple to handle bidirectionality
            region_pair = tuple(sorted([prev_region, row['region']]))
            
            if region_pair not in bidirectional_transitions:
                bidirectional_transitions[region_pair] = {
                    'count': 0,
                    'time': 0,
                    'regions': (prev_region, row['region']),
                    'forward_count': 0,
                    'backward_count': 0
                }
            
            bidirectional_transitions[region_pair]['count'] += 1
            
            # Track directional counts
            if (prev_region, row['region']) == bidirectional_transitions[region_pair]['regions']:
                bidirectional_transitions[region_pair]['forward_count'] += 1
            else:
                bidirectional_transitions[region_pair]['backward_count'] += 1
            
            # If this is a walking activity, add its duration to the transition time
            if row['activity'] == 'Walk':
                bidirectional_transitions[region_pair]['time'] += row['duration']
            
            transitions.append((prev_region, row['region']))
        
        prev_region = row['region']
    
    # Calculate transition counts for line thickness
    transition_counts = Counter(transitions)
    max_count = max(transition_counts.values()) if transition_counts else 1
    
    # Calculate total time for percentages
    total_time = filtered_data['duration'].sum()
    
    # Draw transitions with thickness based on frequency
    drawn_transitions = set()
    
    # Sort transitions by count for proper z-ordering (ascending so higher counts are drawn last)
    sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1])
    
    for (from_region, to_region), count in sorted_transitions:
        # Skip if not both in top regions or already drawn
        if not (from_region in top_regions and to_region in top_regions):
            continue
            
        if (from_region, to_region) in drawn_transitions:
            continue
        
        drawn_transitions.add((from_region, to_region))
        
        # Get coordinates
        if from_region in normalized_coords and to_region in normalized_coords:
            start = normalized_coords[from_region]
            end = normalized_coords[to_region]
            
            # Calculate line width (between min and max width)
            line_width = cfg['movement_line_min_width'] + (
                (cfg['movement_line_max_width'] - cfg['movement_line_min_width']) * 
                (count / max_count)
            )
            
            # Calculate color based on frequency
            color_idx = min(int(count / max_count * (len(cfg['path_colors']) - 1)), len(cfg['path_colors']) - 1)
            color = cfg['path_colors'][color_idx]
            
            # Check if connection is valid in floor plan
            is_adjacent = to_region in name_connections.get(from_region, [])
            line_style = '-' if is_adjacent else ':'
            
            # Calculate z-order based on count (higher count = higher z-order)
            z_order = count
            
            # Draw arrow with z-ordering
            plt.annotate('', 
                      xy=end, 
                      xytext=start,
                      arrowprops=dict(
                          arrowstyle='->', 
                          lw=line_width,
                          color=color,
                          alpha=0.8,
                          linestyle=line_style,
                          zorder=z_order
                      ))
            
            # Add count and percentage label
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Get transition time and percentage
            region_pair = tuple(sorted([from_region, to_region]))
            transition_time = bidirectional_transitions.get(region_pair, {}).get('time', 0)
            transition_pct = (transition_time / total_time * 100) if total_time > 0 else 0
            
            label = f"{count} times"
            if transition_time > 0:
                label += f"\n{transition_pct:.1f}%"
            
            plt.text(mid_x, mid_y, label, 
                   ha='center', va='center', fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                   zorder=z_order+1)
    
    # Draw circles for regions, with size based on time spent
    for region in top_regions:
        if region in normalized_coords and region in region_time.index:
            x, y = normalized_coords[region]
            time = region_time[region]
            
            # Skip regions with very small durations
            if time < cfg['min_duration_threshold']:
                continue
            
            # Scale circle size
            size = cfg['path_circle_base_size'] + cfg['path_circle_scale_factor'] * (time / max_time)
            
            # Use a heat map color based on time spent
            intensity = time / max_time
            color = plt.cm.get_cmap(cfg['region_cmap'])(intensity)
            
            # Draw circle
            circle = plt.Circle(
                (x, y),
                np.sqrt(size),  # Use sqrt for better visual scaling
                color=color,
                alpha=cfg['path_circle_alpha'],
                zorder=100  # Ensure circles are on top
            )
            plt.gca().add_patch(circle)
            
            # Format time as hh:mm:ss
            time_formatted = str(timedelta(seconds=int(time)))
            time_pct = (time / total_time * 100) if total_time > 0 else 0
            
            # Add region label
            plt.text(x, y, f"{region}\n{time_formatted}\n{time_pct:.1f}%", 
                   ha='center', va='center', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                   zorder=101)  # Ensure text is on top of circles
    
    # Add title
    plt.title(f"Combined Movement Path for {employee_id} (Top {len(top_regions)} Regions)", fontsize=16)
    
    # Add legend for line styles
    plt.figtext(0.01, 0.01, "Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions", fontsize=10)
    
    # Turn off axis
    plt.axis('off')
    
    # Add transition table if configured
    if cfg.get('show_transition_table', True):
        # Create a table of transitions
        table_data = []
        for pair, info in bidirectional_transitions.items():
            if pair[0] in top_regions and pair[1] in top_regions:
                from_region, to_region = info['regions']
                count = info['count']
                forward_count = info['forward_count']
                backward_count = info['backward_count']
                time = info['time']
                time_pct = (time / total_time * 100) if total_time > 0 else 0
                
                if cfg.get('combine_bidirectional', True):
                    # Show as bidirectional with counts in both directions
                    table_data.append([
                        f"{from_region} ↔ {to_region}",
                        f"{forward_count}→ {backward_count}←",
                        f"{str(timedelta(seconds=int(time)))}",
                        f"{time_pct:.1f}%"
                    ])
                else:
                    # Show as separate directional transitions
                    if forward_count > 0:
                        table_data.append([
                            f"{from_region} → {to_region}",
                            f"{forward_count}",
                            f"{str(timedelta(seconds=int(time * forward_count/count)))}",
                            f"{time_pct * forward_count/count:.1f}%"
                        ])
                    if backward_count > 0:
                        table_data.append([
                            f"{to_region} → {from_region}",
                            f"{backward_count}",
                            f"{str(timedelta(seconds=int(time * backward_count/count)))}",
                            f"{time_pct * backward_count/count:.1f}%"
                        ])
        
        if table_data:
            # Sort by percentage (descending)
            table_data.sort(key=lambda x: float(x[3].rstrip('%')), reverse=True)
            
            # Limit to top 10 for readability
            show_rows = min(10, len(table_data))
            
            # Create table
            table_ax = plt.axes([0.7, 0.05, 0.25, 0.3])
            table_ax.axis('off')
            
            table = table_ax.table(
                cellText=table_data[:show_rows],
                colLabels=["Transition", "Count", "Time", "%"],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.15, 0.25, 0.2]
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            
            # Add heading
            heading = "Top Transitions"
            if cfg.get('combine_bidirectional', True):
                heading += " (Bidirectional)"
            plt.figtext(0.7, 0.36, heading, ha='left', fontsize=10, weight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(output_dir / f"{employee_id}_combined_path.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_within_region_walking_visualization(employee_id, employee_data, floor_plan, 
                                             normalized_coords, top_regions, output_dir, cfg):
    """Create a visualization showing walking within regions (no transitions)."""
    plt.figure(figsize=(16, 10))
    
    # Display the floor plan
    img = np.array(floor_plan)
    plt.imshow(img)
    
    # Filter data to include only walking activities in top regions
    walk_data = employee_data[
        (employee_data['activity'] == 'Walk') & 
        (employee_data['region'].isin(top_regions))
    ]
    
    # Group walking activities by region
    within_region_walks = {}
    prev_region = None
    
    for _, row in walk_data.sort_values('startTime').iterrows():
        region = row['region']
        
        # Check if this is a within-region walk (not transitioning)
        is_within_region = (prev_region == region) or (prev_region is None)
        
        if is_within_region:
            if region not in within_region_walks:
                within_region_walks[region] = 0
            within_region_walks[region] += row['duration']
        
        prev_region = region
    
    # Calculate total walking time for percentages
    total_walking_time = sum(within_region_walks.values())
    
    # Find max time for relative sizing
    max_time = max(within_region_walks.values()) if within_region_walks else 1
    
    # Draw circles for regions, with size based on walking time
    for region, time in within_region_walks.items():
        if region in normalized_coords:
            x, y = normalized_coords[region]
            
            # Skip regions with very small durations
            if time < cfg['min_duration_threshold']:
                continue
            
            # Scale circle size
            size = cfg['path_circle_base_size'] + cfg['path_circle_scale_factor'] * (time / max_time)
            
            # Use a different color scheme for within-region walking
            intensity = time / max_time
            color = plt.cm.get_cmap('Blues')(intensity)
            
            # Draw circle
            circle = plt.Circle(
                (x, y),
                np.sqrt(size),  # Use sqrt for better visual scaling
                color=color,
                alpha=cfg['path_circle_alpha']
            )
            plt.gca().add_patch(circle)
            
            # Format time as hh:mm:ss
            time_formatted = str(timedelta(seconds=int(time)))
            time_pct = (time / total_walking_time * 100) if total_walking_time > 0 else 0
            
            # Add region label
            plt.text(x, y, f"{region}\n{time_formatted}\n{time_pct:.1f}% of walking", 
                   ha='center', va='center', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add title
    plt.title(f"Within-Region Walking for {employee_id} (Top {len(top_regions)} Regions)", fontsize=16)
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
              "Circle size represents time spent walking within each region (no transitions)", 
              ha='center', fontsize=10)
    
    # Turn off axis
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(output_dir / f"{employee_id}_within_region_walking.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_employee_path_by_source(floor_plan, region_coordinates, data, vis_dir, cfg=None):
    """
    Create visualizations of employee paths by source file, with individual and 
    combined visualizations showing paths between top regions.
    
    Parameters:
    -----------
    floor_plan : PIL.Image
        Floor plan image
    region_coordinates : dict
        Dictionary mapping region names to (x, y) coordinates
    data : pandas.DataFrame
        Input dataframe with tracking data
    vis_dir : str or Path
        Directory to save visualizations
    cfg : dict, optional
        Configuration parameters to override defaults
    """
    # Update configuration with any provided parameters
    config = CONFIG.copy()
    if cfg:
        config.update(cfg)
    
    # Create output directories
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(exist_ok=True, parents=True)
    combined_dir = vis_dir.parent / "combined_paths"
    combined_dir.mkdir(exist_ok=True, parents=True)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    normalized_coords = {}
    for region, (x, y) in region_coordinates.items():
        normalized_coords[region] = (x * img_width, y * img_height)

    # Create simple adjacency if connections file is missing
    region_list = list(region_coordinates.keys())
    name_connections = {}
    for region1 in region_list:
        name_connections[region1] = []
        x1, y1 = region_coordinates[region1]
        
        for region2 in region_list:
            if region1 != region2:
                x2, y2 = region_coordinates[region2]
                # Calculate distance
                distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                # If close enough, consider them adjacent
                if distance < 0.1:  # Adjust this threshold as needed
                    name_connections[region1].append(region2)

    # Extract shift information (default to night shift times)
    shift_info = {}
    for source_file in data['source_file'].unique():
        shift_info[source_file] = ("22:00:00", "13:00:00")  # Default night shift

    # PART 1: Create individual employee-shift visualizations
    print(f"Creating employee path visualizations by source file...")
    source_files = data['source_file'].unique()
    
    for source_file in source_files:
        # Extract employee ID from source file
        employee_id_match = re.match(r'(\d+\-[A-Z])', source_file)
        if not employee_id_match:
            continue

        employee_id = employee_id_match.group(1)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', source_file)
        date_str = date_match.group(1) if date_match else "Unknown"
        shift_start, shift_end = shift_info.get(source_file, ("Unknown", "Unknown"))

        # Get source file data
        source_data = data[data['source_file'] == source_file].sort_values('startTime')
        if len(source_data) == 0:
            continue

        plt.figure(figsize=(16, 10))
        img = np.array(floor_plan)
        plt.imshow(img)

        # Calculate session info
        if len(source_data) >= 2:
            start_time = source_data.iloc[0]['startTimeClock']
            end_time = source_data.iloc[-1]['endTimeClock']
            total_duration_seconds = source_data['duration'].sum()
            total_duration = str(timedelta(seconds=int(total_duration_seconds)))
            session_info = f"Session: {start_time} to {end_time} | Total: {total_duration}"
        else:
            session_info = "Session: Unknown"

        # Count time spent in each region
        region_time = source_data.groupby('region')['duration'].sum()
        max_time = region_time.max() if not region_time.empty else 1

        # Track transitions
        transitions = []
        transition_counts = Counter()
        prev_region = None

        for _, row in source_data.iterrows():
            if prev_region and prev_region != row['region']:
                transitions.append((prev_region, row['region']))
                transition_counts[(prev_region, row['region'])] += 1
            prev_region = row['region']

        # Sort transitions by count and get max count
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1])
        max_count = max(transition_counts.values()) if transition_counts else 1

        # Plot transitions
        for (from_region, to_region), count in sorted_transitions:
            if count < config['min_transition_count']:
                continue

            if from_region in normalized_coords and to_region in normalized_coords:
                start = normalized_coords[from_region]
                end = normalized_coords[to_region]

                # Calculate line width and color
                line_width = config['movement_line_min_width'] + (config['movement_line_max_width'] - config['movement_line_min_width']) * (count / max_count)
                color_idx = min(int(count / max_count * (len(config['path_colors']) - 1)), len(config['path_colors']) - 1)
                line_color = config['path_colors'][color_idx]
                
                # Check adjacency
                is_adjacent = to_region in name_connections.get(from_region, [])
                line_style = '-' if is_adjacent else ':'

                # Draw arrow with z-ordering based on count
                plt.annotate('', 
                          xy=end, 
                          xytext=start,
                          arrowprops=dict(
                              arrowstyle='->', 
                              lw=line_width,
                              color=line_color,
                              alpha=0.8,
                              linestyle=line_style,
                              zorder=count
                          ))

        # Draw circles for regions
        for region, (x, y) in normalized_coords.items():
            if region in region_time:
                time = region_time[region]
                if time < config['min_duration_threshold'] or time/max_time < config['min_region_relative_size']:
                    continue

                # Scale circle size
                size = config['path_circle_base_size'] + config['path_circle_scale_factor'] * (time / max_time)
                time_formatted = str(timedelta(seconds=int(time)))

                # Draw circle
                plt.scatter(x, y, s=size, alpha=config['path_circle_alpha'], 
                          color='#FFAA00', edgecolor='#E8655F', linewidth=1)

                # Add label
                label_text = f"{region}\n{time_formatted}" if config['show_region_names'] else f"{time_formatted}"
                plt.text(x, y, label_text, fontsize=config['heatmap_text_size'], ha='center', va='center',
                       bbox=dict(facecolor=config['text_box_color'], alpha=config['text_box_alpha'], edgecolor='none'))

        # Add legend
        plt.legend([
            Line2D([0], [0], color='gray', lw=config['movement_line_min_width'], label='Low frequency'),
            Line2D([0], [0], color='red', lw=config['movement_line_max_width'], label='High frequency')
        ], ['Low frequency', 'High frequency'], loc='upper right')

        # Set title and save
        plt.title(f'Movement Path for {employee_id} - {date_str}\n'
                 f'Shift: {shift_start} to {shift_end}\n{session_info}', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        safe_filename = source_file.replace(':', '_').replace('/', '_').replace('\\', '_')
        plt.savefig(vis_dir / f'{safe_filename}_path.png', dpi=300, bbox_inches='tight')
        plt.close()

    # PART 2: Create combined visualizations for each employee
    print("Creating combined visualizations for each employee...")
    employees = data['id'].unique()

    for employee_id in employees:
        # Get all data for this employee
        employee_data = data[data['id'] == employee_id]
        if len(employee_data) == 0:
            continue
        
        # Calculate total time spent in each region
        region_time = employee_data.groupby('region')['duration'].sum()
        
        # Get top N regions by time spent
        top_n = config.get('top_regions_count', 6)
        top_regions = region_time.nlargest(top_n).index.tolist()
        
        # Create combined path visualization
        create_combined_path_visualization(
            employee_id,
            employee_data,
            floor_plan,
            normalized_coords,
            name_connections,
            top_regions,
            combined_dir,
            config
        )
        
        # Create within-region walking visualization
        create_within_region_walking_visualization(
            employee_id,
            employee_data,
            floor_plan,
            normalized_coords,
            top_regions,
            combined_dir,
            config
        )
        