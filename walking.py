#!/usr/bin/env python3
"""
Walking Path Analysis Module

Analyzes employee movement patterns to distinguish between meaningful presence in regions
versus transit paths. Identifies and visualizes key regions where employees spend time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import json
import os
from PIL import Image

# Constants
MEANINGFUL_PRESENCE_THRESHOLD = 15  # seconds
TOP_REGIONS_COUNT = 8

def load_data(data_path):
    """Load and preprocess the bakery tracking data"""
    print(f"Loading data from {data_path}...")
    
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} records")
        
        # Remove pause data if the column exists
        if 'isPauseData' in data.columns:
            data = data[data['isPauseData'] == False]
            print(f"After removing pause data: {len(data)} records")
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_floor_plan_data(layout_path, metadata_path, connections_path):
    """Load floor plan data from files"""
    print("Loading floor plan data...")
    
    try:
        # Load floor plan image
        floor_plan = Image.open(layout_path)
        print(f"Loaded floor plan image: {layout_path}")
    except Exception as e:
        print(f"Error loading floor plan image: {e}")
        return None
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata_path}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None
    
    # Load connections
    try:
        with open(connections_path, 'r') as f:
            connections = json.load(f)
        print(f"Loaded connections: {connections_path}")
    except Exception as e:
        print(f"Error loading connections: {e}")
        return None
    
    # Create uuid to name mapping
    uuid_to_name = {}
    name_to_uuid = {}
    region_coordinates = {}
    region_dimensions = {}
    
    for region in metadata['layout']['regions']:
        uuid_to_name[region['uuid']] = region['name']
        name_to_uuid[region['name']] = region['uuid']
        
        # Calculate coordinates and dimensions for visualization
        x_center = (region['position_top_left_x'] + region['position_bottom_right_x']) / 2
        y_center = (region['position_top_left_y'] + region['position_bottom_right_y']) / 2
        width = region['position_bottom_right_x'] - region['position_top_left_x']
        height = region['position_bottom_right_y'] - region['position_top_left_y']
        
        region_coordinates[region['name']] = (x_center, y_center)
        region_dimensions[region['name']] = (width, height)
    
    # Create connection mapping by region name
    name_connections = {}
    for source_uuid, target_uuids in connections.items():
        if source_uuid in uuid_to_name:
            source_name = uuid_to_name[source_uuid]
            name_connections[source_name] = [uuid_to_name.get(uuid, '') for uuid in target_uuids if uuid in uuid_to_name]
    
    return {
        'floor_plan': floor_plan,
        'metadata': metadata,
        'connections': connections,
        'name_connections': name_connections,
        'uuid_to_name': uuid_to_name,
        'name_to_uuid': name_to_uuid,
        'region_coordinates': region_coordinates,
        'region_dimensions': region_dimensions
    }

def analyze_region_visits(employee_data):
    """
    Analyze an employee's visits to regions, distinguishing between meaningful presence and transitions
    
    Parameters:
    -----------
    employee_data : pandas.DataFrame
        Data for a single employee, sorted by time
    
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
    
    current_region = None
    visit_start_idx = None
    visit_rows = []
    
    # Process each record sequentially
    for idx, row in employee_data.iterrows():
        if current_region is None:
            # First record - start tracking this region
            current_region = row['region']
            visit_start_idx = idx
            visit_rows = [row]
        elif row['region'] != current_region:
            # Region changed - analyze the completed visit
            visit_end_idx = idx - 1
            visit_end_row = employee_data.loc[visit_end_idx]
            
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
                'start_time': visit_rows[0]['startTime'],
                'end_time': visit_rows[-1]['endTime'],
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
            visit_start_idx = idx
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
            'start_time': visit_rows[0]['startTime'],
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

def calculate_walking_paths(employee_analysis):
    """
    Calculate complete walking paths from region visits
    
    Parameters:
    -----------
    employee_analysis : dict
        Output from analyze_region_visits function
    
    Returns:
    --------
    list
        List of walking paths
    """
    all_visits = employee_analysis['all_visits']
    region_transitions = employee_analysis['region_transitions']
    meaningful_visits = employee_analysis['meaningful_visits']
    
    paths = []
    current_path = []
    
    # Identify meaningful regions (for highlighting in paths)
    meaningful_regions = set(visit['region'] for visit in meaningful_visits)
    
    # Build complete paths including transitions
    for i, visit in enumerate(all_visits):
        region = visit['region']
        is_meaningful = visit['duration'] >= MEANINGFUL_PRESENCE_THRESHOLD
        
        current_path.append({
            'region': region,
            'duration': visit['duration'],
            'is_meaningful': is_meaningful,
            'activities': visit['activity_durations']
        })
        
        # End path if this is the last visit or next visit is not directly connected
        if i == len(all_visits) - 1:
            if len(current_path) > 1:
                paths.append(current_path)
            current_path = []
    
    return paths

def format_seconds(seconds):
    """Format seconds as minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    else:
        return f"{remaining_seconds}s"

def create_employee_path_visualization(employee_id, shift, paths, floor_plan_data, top_regions, output_dir):
    """
    Create visualization of employee walking paths
    
    Parameters:
    -----------
    employee_id : str
        Employee ID
    shift : int
        Shift number
    paths : list
        List of walking paths
    floor_plan_data : dict
        Floor plan data
    top_regions : list
        List of top regions by time spent
    output_dir : Path
        Output directory
    
    Returns:
    --------
    str
        Path to saved visualization
    """
    floor_plan = floor_plan_data['floor_plan']
    region_coordinates = floor_plan_data['region_coordinates']
    region_dimensions = floor_plan_data['region_dimensions']
    name_connections = floor_plan_data['name_connections']
    
    # Create figure
    plt.figure(figsize=(20, 14))
    
    # Display the floor plan
    img = np.array(floor_plan)
    plt.imshow(img)
    
    # Normalize coordinates to match the image dimensions
    img_width, img_height = floor_plan.size
    
    # Create region circles for top regions
    top_region_names = [r['region'] for r in top_regions]
    
    for region in top_region_names:
        if region in region_coordinates:
            x, y = region_coordinates[region]
            
            # Scale coordinates to match image
            x_scaled = x * img_width
            y_scaled = y * img_height
            
            # Draw circle for region
            circle = plt.Circle(
                (x_scaled, y_scaled),
                radius=40,  # Adjust as needed
                color='#FFAA00',
                alpha=0.6
            )
            plt.gca().add_patch(circle)
            
            # Find region stats in top_regions
            region_info = next((r for r in top_regions if r['region'] == region), None)
            if region_info:
                avg_duration = region_info['avg_meaningful_duration']
                total_duration = region_info['total_duration']
                visit_count = region_info['meaningful_visits']
                
                # Add label with duration info
                plt.text(
                    x_scaled, y_scaled,
                    f"{region}\n{format_seconds(avg_duration)}/visit\n{format_seconds(total_duration)} total\n({visit_count} visits)",
                    ha='center',
                    va='center',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
    
    # Draw transition paths
    drawn_transitions = set()
    
    # Custom colormap for transitions
    transition_cmap = LinearSegmentedColormap.from_list(
        'transition_cmap', 
        ['#87CEEB', '#44A4DF', '#FFD700', '#FF8C00', '#FF4500']
    )
    
    # Find all transitions
    transition_counts = Counter()
    for path in paths:
        for i in range(len(path) - 1):
            from_region = path[i]['region']
            to_region = path[i+1]['region']
            transition_counts[(from_region, to_region)] += 1
    
    max_count = max(transition_counts.values()) if transition_counts else 1
    
    # Draw transition arrows
    for (from_region, to_region), count in transition_counts.items():
        if from_region in region_coordinates and to_region in region_coordinates:
            # Skip if already drawn
            if (from_region, to_region) in drawn_transitions:
                continue
            
            drawn_transitions.add((from_region, to_region))
            
            # Get coordinates
            from_x, from_y = region_coordinates[from_region]
            to_x, to_y = region_coordinates[to_region]
            
            # Scale coordinates
            from_x_scaled = from_x * img_width
            from_y_scaled = from_y * img_height
            to_x_scaled = to_x * img_width
            to_y_scaled = to_y * img_height
            
            # Calculate line width based on count (1-5 range)
            line_width = 1 + 4 * (count / max_count)
            
            # Calculate color intensity based on count
            color = transition_cmap(count / max_count)
            
            # Check if connection is valid in floor plan
            is_adjacent = to_region in name_connections.get(from_region, [])
            line_style = '-' if is_adjacent else ':'
            
            # Draw arrow
            plt.arrow(
                from_x_scaled, from_y_scaled,
                (to_x_scaled - from_x_scaled) * 0.8,  # Scale to not overlap with circles
                (to_y_scaled - from_y_scaled) * 0.8,
                head_width=15,
                head_length=15,
                fc=color,
                ec=color,
                linewidth=line_width,
                linestyle=line_style,
                length_includes_head=True,
                alpha=0.7,
                zorder=10
            )
            
            # Add count label near middle of arrow
            mid_x = (from_x_scaled + to_x_scaled) / 2
            mid_y = (from_y_scaled + to_y_scaled) / 2
            plt.text(
                mid_x, mid_y,
                str(count),
                fontsize=10,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                zorder=11
            )
    
    # Set title
    plt.title(f"Walking Paths for Employee {employee_id} - Shift {shift}", fontsize=16)
    
    # Add legend for line types
    plt.figtext(0.01, 0.01, "Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions", fontsize=10)
    
    # Turn off axis
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(output_dir) / f"employee_{employee_id}_shift_{shift}_walking_paths.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def analyze_employee_data(data, floor_plan_data, output_dir):
    """
    Analyze walking patterns for all employees and shifts
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    floor_plan_data : dict
        Floor plan data
    output_dir : Path
        Output directory
    
    Returns:
    --------
    dict
        Analysis results
    """
    results = {}
    
    # Create output directories
    vis_dir = Path(output_dir) / "visualizations"
    stats_dir = Path(output_dir) / "statistics"
    
    vis_dir.mkdir(exist_ok=True, parents=True)
    stats_dir.mkdir(exist_ok=True, parents=True)
    
    # Group data by employee and shift
    grouped = data.groupby(['id', 'shift'])
    
    print(f"Analyzing {len(grouped)} employee-shift combinations...")
    
    for (employee_id, shift), group in grouped:
        print(f"Processing employee {employee_id}, shift {shift}...")
        
        # Sort by time
        employee_data = group.sort_values('startTime')
        
        # Analyze region visits
        visit_analysis = analyze_region_visits(employee_data)
        
        # Calculate walking paths
        paths = calculate_walking_paths(visit_analysis)
        
        # Get top regions by total duration
        region_stats = visit_analysis['region_stats']
        top_regions = [
            {'region': region, **stats}
            for region, stats in region_stats.items()
            if stats['meaningful_visits'] > 0  # Only include regions with meaningful visits
        ]
        top_regions.sort(key=lambda x: x['total_duration'], reverse=True)
        top_regions = top_regions[:TOP_REGIONS_COUNT]
        
        # Generate visualization
        vis_path = create_employee_path_visualization(
            employee_id, shift, paths, floor_plan_data, top_regions, vis_dir
        )
        
        # Save statistics
        stats_output = {
            'employee_id': employee_id,
            'shift': shift,
            'total_records': len(employee_data),
            'total_duration': employee_data['duration'].sum(),
            'region_visits': {
                'meaningful': len(visit_analysis['meaningful_visits']),
                'transitions': len(visit_analysis['transitions']),
                'total': len(visit_analysis['all_visits'])
            },
            'top_regions': [
                {
                    'region': region_info['region'],
                    'total_duration': region_info['total_duration'],
                    'meaningful_visits': region_info['meaningful_visits'],
                    'avg_meaningful_duration': region_info['avg_meaningful_duration'],
                    'formatted_avg_duration': format_seconds(region_info['avg_meaningful_duration']),
                    'formatted_total_duration': format_seconds(region_info['total_duration'])
                }
                for region_info in top_regions
            ],
            'most_common_transitions': [
                {
                    'from_region': from_region,
                    'to_region': to_region,
                    'count': count
                }
                for (from_region, to_region), count in visit_analysis['transition_counts'].most_common(10)
            ],
            'visualization_path': vis_path
        }
        
        # Save to JSON
        stats_path = stats_dir / f"employee_{employee_id}_shift_{shift}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_output, f, indent=2)
        
        # Store in results
        results[(employee_id, shift)] = {
            'visit_analysis': visit_analysis,
            'paths': paths,
            'top_regions': top_regions,
            'stats': stats_output
        }
    
    # Generate summary CSV of top regions for all employees
    summary_rows = []
    
    for (employee_id, shift), result in results.items():
        for region_info in result['top_regions']:
            summary_rows.append({
                'employee_id': employee_id,
                'shift': shift,
                'region': region_info['region'],
                'total_duration': region_info['total_duration'],
                'meaningful_visits': region_info['meaningful_visits'],
                'avg_meaningful_duration': region_info['avg_meaningful_duration'],
                'formatted_avg_duration': format_seconds(region_info['avg_meaningful_duration']),
                'formatted_total_duration': format_seconds(region_info['total_duration'])
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(stats_dir / "top_regions_summary.csv", index=False)
        print(f"Saved summary to {stats_dir}/top_regions_summary.csv")
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze bakery employee walking patterns')
    
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                      help='Path to the data file')
    parser.add_argument('--layout', type=str, default='data/assets/layout.png',
                      help='Path to the floor plan image')
    parser.add_argument('--metadata', type=str, default='data/assets/process_metadata.json',
                      help='Path to the process metadata file')
    parser.add_argument('--connections', type=str, default='data/assets/floor-plan-connections-2025-04-09.json',
                      help='Path to the floor plan connections file')
    parser.add_argument('--output', type=str, default='output/walking_analysis',
                      help='Directory to save output files')
    parser.add_argument('--threshold', type=int, default=15,
                      help='Threshold in seconds for meaningful region presence')
    parser.add_argument('--employee', type=str,
                      help='Analyze specific employee (e.g., 32-A)')
    parser.add_argument('--shift', type=int,
                      help='Analyze specific shift')
    
    return parser.parse_args()

def main():
    """Main function to run the analysis"""
    # Parse command line arguments
    args = parse_args()
    
    # Set threshold from arguments
    global MEANINGFUL_PRESENCE_THRESHOLD
    MEANINGFUL_PRESENCE_THRESHOLD = args.threshold
    
    # Load data
    data = load_data(args.data)
    if data is None:
        return 1
    
    # Load floor plan data
    floor_plan_data = load_floor_plan_data(args.layout, args.metadata, args.connections)
    if floor_plan_data is None:
        return 1
    
    # Filter data if employee or shift specified
    if args.employee:
        data = data[data['id'] == args.employee]
        print(f"Filtered to employee {args.employee}: {len(data)} records")
    
    if args.shift:
        data = data[data['shift'] == args.shift]
        print(f"Filtered to shift {args.shift}: {len(data)} records")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run analysis
    results = analyze_employee_data(data, floor_plan_data, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    main()