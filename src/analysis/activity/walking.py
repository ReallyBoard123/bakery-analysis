"""
Functions to modify for the improved walking pattern analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from collections import defaultdict

from ...utils.time_utils import format_seconds_to_hms
from ...visualization.base import set_visualization_style, get_employee_colors, save_figure, get_text

# Define consistent colors for shifts
SHIFT_COLORS = {
    1: '#8A2BE2',  # Purple
    2: '#228B22',  # Green
    3: '#20B2AA',  # Light Sea Green
    4: '#32CD32',  # Lime Green
    5: '#1E90FF',  # Dodger Blue
    6: '#FF8C00',  # Dark Orange
    7: '#4169E1',  # Royal Blue
    8: '#FF4500',  # Orange Red
    9: '#9932CC',  # Dark Orchid
}

def get_shift_color(shift, base_color=None):
    """Get consistent color for shifts"""
    if shift in SHIFT_COLORS:
        return SHIFT_COLORS[shift]
    
    # Fallback to base color with hue variation
    if base_color:
        from matplotlib.colors import to_rgba
        import colorsys
        
        # Convert to HSV, adjust hue, convert back to RGB
        base_rgba = to_rgba(base_color)
        h, s, v = colorsys.rgb_to_hsv(*base_rgba[:3])
        shift_val = (int(shift) % 10) / 10  # Use shift value to adjust hue
        new_h = (h + shift_val) % 1.0
        r, g, b = colorsys.hsv_to_rgb(new_h, s, v)
        return (r, g, b, base_rgba[3])
    
    # Final fallback
    return plt.cm.tab10(int(shift) % 10)

def create_shift_comparison_plot(emp_data, emp_id, shifts, slots_per_day, 
                              time_slot_labels, output_dir, time_slot_minutes, language='en'):
    """
    Create an improved visualization comparing walking patterns across different shifts
    with consistent colors, time ranges, and region information
    """
    set_visualization_style()
    
    # Create figure with gridspec for main plot and region panel
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Main plot (top)
    ax = fig.add_subplot(gs[0])
    
    # Get employee color
    employee_colors = get_employee_colors()
    base_color = employee_colors.get(emp_id, '#1f77b4')
    
    # Track active time periods
    all_active_slots = []
    
    # Calculate walking stats by shift and time slot
    all_shift_data = []
    max_minutes = 0
    
    for shift_idx, shift in enumerate(shifts):
        # Filter data for this shift
        shift_data = emp_data[emp_data['shift'] == shift]
        
        # Calculate time slot statistics 
        # Also collect region information per time slot
        slot_stats = defaultdict(lambda: {'duration': 0, 'count': 0, 'regions': []})
        
        for _, row in shift_data.iterrows():
            slot_idx = int(row['time_slot_idx'])
            slot_stats[slot_idx]['duration'] += row['duration']
            slot_stats[slot_idx]['count'] += 1
            slot_stats[slot_idx]['regions'].append(row['region'])
        
        # Convert to DataFrame format
        slot_data = []
        for slot_idx, stats in slot_stats.items():
            # Find most common region
            regions = stats['regions']
            most_common = max(set(regions), key=regions.count) if regions else None
            
            slot_data.append({
                'time_slot_idx': slot_idx,
                'total_duration': stats['duration'],
                'walk_count': stats['count'],
                'most_common_region': most_common
            })
        
        slot_df = pd.DataFrame(slot_data)
        
        # Track which slots have activity (for focusing the x-axis)
        active_slots = slot_df[slot_df['total_duration'] > 0]['time_slot_idx'].tolist()
        all_active_slots.extend(active_slots)
        
        # Create full time series with all slots
        full_slots = pd.DataFrame({
            'time_slot_idx': range(slots_per_day),
            'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
        })
        
        # Merge with slot data to ensure all slots exist
        if not slot_df.empty:
            full_data = pd.merge(full_slots, slot_df, on=['time_slot_idx'], how='left')
        else:
            full_data = full_slots.copy()
            full_data['total_duration'] = 0
            full_data['walk_count'] = 0
            full_data['most_common_region'] = None
            
        full_data = full_data.fillna(0).sort_values('time_slot_idx')
        full_data['duration_minutes'] = full_data['total_duration'] / 60
        
        all_shift_data.append((shift, full_data))
        max_minutes = max(max_minutes, full_data['duration_minutes'].max())
    
    # Store region data by time slot for the bottom panel
    time_slot_regions = defaultdict(list)
    
    # Plot each shift's data with enhanced smoothing
    for shift, shift_data in all_shift_data:
        # Get consistent color for this shift
        color = get_shift_color(shift, base_color)
        
        # Apply Gaussian smoothing
        y_values = shift_data['duration_minutes'].values
        smoothed_y = gaussian_filter1d(y_values, sigma=2.0)
        
        # Plot with thicker, smoother lines
        ax.plot(
            shift_data['time_slot_idx'],
            smoothed_y,
            '-', 
            label=f"Shift {shift}",
            linewidth=3.0,
            color=color,
            alpha=0.9
        )
        
        # Add subtle fill below the line for density-like appearance
        ax.fill_between(
            shift_data['time_slot_idx'],
            smoothed_y,
            alpha=0.2,
            color=color
        )
        
        # Find peaks for region labeling
        peaks = []
        for i in range(1, len(smoothed_y)-1):
            if smoothed_y[i] > max(smoothed_y[i-1], smoothed_y[i+1]) and smoothed_y[i] > 1.0:
                peaks.append((i, smoothed_y[i]))
        
        # Sort by height and take top peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        for idx, height in peaks[:2]:  # Limit to top 2 peaks
            if height > 0:
                # Get region info if available
                region = None
                if idx < len(shift_data) and 'most_common_region' in shift_data.columns:
                    region = shift_data.iloc[idx]['most_common_region']
                    
                # Collect region data for bottom panel
                if region:
                    time_slot_regions[idx].append((shift, region, height))
                
                # Add marker at peak
                ax.scatter(
                    idx,
                    height,
                    s=80,
                    color=color,
                    edgecolor='white',
                    zorder=10
                )
                
                # Add label with minutes and region
                region_text = f"\n({region})" if region else ""
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
    
    # Create bottom panel for region visualization
    ax_regions = fig.add_subplot(gs[1])
    
    # Determine global time range focusing on activity periods
    if all_active_slots:
        global_min = max(0, min(all_active_slots) - 4)  # Add padding
        global_max = min(slots_per_day - 1, max(all_active_slots) + 4)
    else:
        # Default to showing roughly 7:30PM to 2PM (next day)
        evening_slot = 19 * 60 // time_slot_minutes  # 7:00 PM
        afternoon_slot = 14 * 60 // time_slot_minutes  # 2:00 PM
        global_min = evening_slot
        global_max = afternoon_slot
    
    # Set both axes to use same x range
    ax.set_xlim(global_min, global_max)
    ax_regions.set_xlim(global_min, global_max)
    
    # Set y-axis limit with padding
    ax.set_ylim(0, max_minutes * 1.2)
    
    # Extract unique regions from time slots
    all_regions = set()
    for slot_regions in time_slot_regions.values():
        for _, region, _ in slot_regions:
            if region:
                all_regions.add(region)
    
    # Create colors for regions
    region_colors = {}
    color_cycle = plt.cm.tab20(np.linspace(0, 1, len(all_regions)))
    for i, region in enumerate(sorted(all_regions)):
        region_colors[region] = color_cycle[i]
    
    # Create region visualization in bottom panel
    for slot_idx in range(global_min, global_max + 1):
        if slot_idx in time_slot_regions:
            # Get most common region at this time slot
            slot_regions = time_slot_regions[slot_idx]
            
            for shift, region, _ in slot_regions:
                if region:
                    color = region_colors.get(region, 'gray')
                    ax_regions.bar(
                        slot_idx, 
                        1, 
                        width=0.8, 
                        color=color,
                        alpha=0.8
                    )
    
    # Add region legend if we have regions
    if region_colors:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=region) 
                           for region, color in region_colors.items()]
        ax_regions.legend(
            handles=legend_elements, 
            loc='upper right',
            title="Most Visited Regions",
            bbox_to_anchor=(1.01, 1),
            ncol=min(3, len(region_colors))
        )
    
    # Configure x-axis ticks with appropriate spacing
    span = global_max - global_min
    if span <= 24:
        tick_every = 1
    else:
        tick_every = max(1, span // 12)
    
    tick_positions = range(global_min, global_max + 1, tick_every)
    
    # Set labels
    ax.set_title(f"Walking Patterns Across Shifts - Employee {emp_id}", 
                fontsize=18, fontweight='bold', color='green')
    ax.set_ylabel("Minutes Spent Walking", fontsize=14, fontweight='bold')
    
    # Hide x tick labels on top plot
    ax.set_xticklabels([])
    
    # Configure region plot
    ax_regions.set_ylabel("Region", fontsize=12)
    ax_regions.set_xlabel("Time of Day", fontsize=14, fontweight='bold')
    ax_regions.set_yticks([])
    
    # Set x ticks on bottom plot
    ax_regions.set_xticks(tick_positions)
    ax_regions.set_xticklabels(
        [time_slot_labels[i % len(time_slot_labels)] for i in tick_positions], 
        rotation=45
    )
    
    # Enhanced grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # More attractive legend for shifts
    legend = ax.legend(title="Shift", loc="upper right", frameon=True, fancybox=True, 
                      framealpha=0.9, shadow=True, fontsize=12)
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, output_dir / f"employee_{emp_id}_shift_comparison_{time_slot_minutes}min.png")
    
    plt.close()
    return fig


def create_shift_employee_comparison(shift_data, shift, slots_per_day, 
                                  time_slot_labels, output_dir, time_slot_minutes, language='en'):
    """
    Create a visualization comparing all employees within a single shift
    with region information
    """
    set_visualization_style()
    
    # Create figure with gridspec for main plot and region panel
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Main plot (top)
    ax = fig.add_subplot(gs[0])
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Track all active slots
    all_active_slots = []
    
    # Store employee time slot data and region information
    all_emp_data = []
    time_slot_regions = defaultdict(list)
    
    for emp_id in shift_data['id'].unique():
        # Filter data for this employee
        emp_shift_data = shift_data[shift_data['id'] == emp_id]
        
        # Calculate time slot statistics and regions
        slot_stats = defaultdict(lambda: {'duration': 0, 'count': 0, 'regions': []})
        
        for _, row in emp_shift_data.iterrows():
            slot_idx = int(row['time_slot_idx'])
            slot_stats[slot_idx]['duration'] += row['duration']
            slot_stats[slot_idx]['count'] += 1
            slot_stats[slot_idx]['regions'].append(row['region'])
        
        # Convert to DataFrame
        slot_data = []
        for slot_idx, stats in slot_stats.items():
            # Find most common region
            regions = stats['regions']
            most_common = max(set(regions), key=regions.count) if regions else None
            
            slot_data.append({
                'time_slot_idx': slot_idx,
                'total_duration': stats['duration'],
                'walk_count': stats['count'],
                'most_common_region': most_common
            })
        
        slot_df = pd.DataFrame(slot_data)
        
        # Track active slots
        active_slots = slot_df[slot_df['total_duration'] > 0]['time_slot_idx'].tolist()
        all_active_slots.extend(active_slots)
        
        # Create full time series
        full_slots = pd.DataFrame({
            'time_slot_idx': range(slots_per_day),
            'time_slot': [time_slot_labels[i] for i in range(slots_per_day)]
        })
        
        # Merge with slot data
        if not slot_df.empty:
            full_data = pd.merge(full_slots, slot_df, on=['time_slot_idx'], how='left')
        else:
            full_data = full_slots.copy()
            full_data['total_duration'] = 0
            full_data['walk_count'] = 0
            full_data['most_common_region'] = None
            
        full_data = full_data.fillna(0).sort_values('time_slot_idx')
        full_data['duration_minutes'] = full_data['total_duration'] / 60
        
        all_emp_data.append((emp_id, full_data))
    
    # Extract regions across employees
    all_regions = set()
    
    # Plot each employee's data
    for emp_id, emp_data in all_emp_data:
        # Apply smoothing for nicer curves
        y_values = emp_data['duration_minutes'].values
        smoothed_y = gaussian_filter1d(y_values, sigma=1.5)
        
        # Get employee color
        color = employee_colors.get(emp_id, None)
        
        # Plot the line
        ax.plot(
            emp_data['time_slot_idx'],
            smoothed_y,
            '-',
            label=f"Employee {emp_id}",
            linewidth=2.5,
            color=color
        )
        
        # Find peaks
        peaks = []
        for i in range(1, len(smoothed_y)-1):
            if smoothed_y[i] > max(smoothed_y[i-1], smoothed_y[i+1]) and smoothed_y[i] > 1.0:
                peaks.append((i, smoothed_y[i]))
        
        # Sort and take top peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        for idx, height in peaks[:2]:  # Limit to 2 peaks per employee
            if height > 0:
                # Get region info
                region = None
                if idx < len(emp_data) and 'most_common_region' in emp_data.columns:
                    region = emp_data.iloc[idx]['most_common_region']
                
                # Store region info for bottom panel
                if region:
                    time_slot_regions[idx].append((emp_id, region, height))
                    all_regions.add(region)
                
                # Add marker
                ax.scatter(
                    idx,
                    height,
                    s=70,
                    color=color,
                    edgecolor='white',
                    zorder=10
                )
                
                # Add label with region
                region_text = f"\n({region})" if region else ""
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
    
    # Region visualization in bottom panel
    ax_regions = fig.add_subplot(gs[1])
    
    # Determine global time range
    if all_active_slots:
        global_min = max(0, min(all_active_slots) - 4)
        global_max = min(slots_per_day - 1, max(all_active_slots) + 4)
    else:
        # Default to showing roughly 7:30PM to 2PM (next day)
        evening_slot = 19 * 60 // time_slot_minutes  # 7:00 PM
        afternoon_slot = 14 * 60 // time_slot_minutes  # 2:00 PM  
        global_min = evening_slot
        global_max = afternoon_slot
    
    # Set both axes to use same x range
    ax.set_xlim(global_min, global_max)
    ax_regions.set_xlim(global_min, global_max)
    
    # Colors for regions
    region_colors = {}
    color_cycle = plt.cm.tab20(np.linspace(0, 1, len(all_regions)))
    for i, region in enumerate(sorted(all_regions)):
        region_colors[region] = color_cycle[i]
    
    # Create region visualization
    for slot_idx in range(global_min, global_max + 1):
        if slot_idx in time_slot_regions:
            slot_regions = time_slot_regions[slot_idx]
            
            for emp_id, region, _ in slot_regions:
                if region:
                    color = region_colors.get(region, 'gray')
                    ax_regions.bar(
                        slot_idx,
                        1,
                        width=0.8,
                        color=color,
                        alpha=0.8
                    )
    
    # Add region legend
    if region_colors:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=region)
                           for region, color in region_colors.items()]
        ax_regions.legend(
            handles=legend_elements,
            loc='upper right',
            title="Most Visited Regions",
            bbox_to_anchor=(1.01, 1),
            ncol=min(3, len(region_colors))
        )
    
    # Configure x-axis ticks
    span = global_max - global_min
    if span <= 24:
        tick_every = 1
    else:
        tick_every = max(1, span // 12)
    
    tick_positions = range(global_min, global_max + 1, tick_every)
    
    # Set labels
    ax.set_title(f"Walking Patterns for Shift {shift}", fontsize=16, fontweight='bold', color='brown')
    ax.set_ylabel("Minutes Spent Walking", fontsize=14)
    
    # Hide x tick labels on top plot
    ax.set_xticklabels([])
    
    # Configure region plot
    ax_regions.set_ylabel("Region", fontsize=12)
    ax_regions.set_xlabel("Time of Day", fontsize=14)
    ax_regions.set_yticks([])
    
    # Set x ticks on bottom plot
    ax_regions.set_xticks(tick_positions)
    ax_regions.set_xticklabels(
        [time_slot_labels[i % len(time_slot_labels)] for i in tick_positions],
        rotation=45
    )
    
    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(title="Employee", loc="upper right")
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, output_dir / f"shift_{shift}_employee_comparison_{time_slot_minutes}min.png")
    
    plt.close()
    return fig


def analyze_walking_by_shift(data, output_dir=None, time_slot_minutes=30, language='en'):
    """
    Analyze walking patterns by shift for each employee with improved visualizations
    """
    # Filter walking data
    walking_data = data[data['activity'] == 'Walk'].copy()
    
    if walking_data.empty:
        print(f"No walking data found in the dataset.")
        return {}
    
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