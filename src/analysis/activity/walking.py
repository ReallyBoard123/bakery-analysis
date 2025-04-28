"""
Functions for walking pattern analysis
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
from datetime import datetime

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

# Define colors for specific regions
REGION_COLORS = {
    'Fettbacken': '#6BAED6',            # Light blue
    'Konditorei_station': '#3182BD',    # Blue
    'konditorei_deco': '#08519C',       # Dark blue
    'Brotstation': '#31A354',           # Green
    '1_Brotstation': '#31A354',         # Same green for variants
    '2_Brotstation': '#31A354',
    '3_Brotstation': '#31A354',
    '4_Brotstation': '#31A354',
    'Mehlmaschine': '#74C476',          # Light green
    'brotkisten_regal': '#FFFF00',      # Yellow
    '1_Konditorei_station': '#3182BD',  # Blue variants
    '2_Konditorei_station': '#3182BD',
    '3_Konditorei_station': '#3182BD',
    'bereitstellen_prepared_goods': '#FF7F0E',  # Orange
    '1_corridor': '#D62728',            # Red
    'leere_Kisten': '#9467BD',          # Purple
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

def get_region_color(region, default_colors=None):
    """Get color for a specific region"""
    # First check predefined region colors
    if region in REGION_COLORS:
        return REGION_COLORS[region]
    
    # Next check if we have a custom color dictionary
    if default_colors and region in default_colors:
        return default_colors[region]
    
    # Fallback to a standard color scheme
    return None  # This will use the default coloring system

def prepare_continuous_timeline(shift_data, time_slot_minutes=30):
    """
    Create a continuous timeline that respects date boundaries
    """
    # Sort by date and startTime for proper order
    if 'date' in shift_data.columns:
        shift_data = shift_data.sort_values(['date', 'startTime'])
    else:
        shift_data = shift_data.sort_values('startTime')
    
    # Extract hour of day
    shift_data['hour_of_day'] = shift_data['startTime'] / 3600  # Convert to hours
    
    # Number of slots in a day
    slots_per_day = 24 * 60 // time_slot_minutes
    
    # Create basic time slot index (within a day)
    shift_data['time_slot_idx'] = (shift_data['hour_of_day'] * 60 / time_slot_minutes).astype(int) % slots_per_day
    
    # Create continuous slot index that respects date boundaries
    if 'date' in shift_data.columns:
        # Convert date strings to numeric days since first date
        base_date = shift_data['date'].min()
        shift_data['day_offset'] = shift_data['date'].apply(lambda d: 
            (datetime.strptime(d, '%d-%m-%Y') - datetime.strptime(base_date, '%d-%m-%Y')).days)
        
        # Continuous slot = day_offset * slots_per_day + time_slot_idx
        shift_data['continuous_slot'] = shift_data['day_offset'] * slots_per_day + shift_data['time_slot_idx']
    else:
        # If no date column, just use the time_slot_idx
        shift_data['continuous_slot'] = shift_data['time_slot_idx']
    
    # Create time slot labels
    time_slot_labels = []
    for i in range(slots_per_day):
        minutes_from_midnight = i * time_slot_minutes
        hours = minutes_from_midnight // 60
        minutes = minutes_from_midnight % 60
        time_slot_labels.append(f"{hours:02d}:{minutes:02d}")
    
    shift_data['time_slot'] = shift_data['time_slot_idx'].apply(lambda x: time_slot_labels[x])
    
    return shift_data, slots_per_day, time_slot_labels

def create_shift_comparison_plot(emp_data, emp_id, shifts, time_slot_minutes, output_dir, 
                               language='en', show_region_labels=False, custom_region_colors=None):
    """
    Create an improved visualization comparing walking patterns across different shifts
    with continuous timeline and no gaps
    
    Parameters:
    -----------
    emp_data : pandas.DataFrame
        Input dataframe with employee data
    emp_id : str
        Employee ID to analyze
    shifts : list
        List of shifts to compare
    time_slot_minutes : int
        Size of time slots in minutes
    output_dir : Path
        Output directory for saving files
    language : str, optional
        Language code ('en' or 'de')
    show_region_labels : bool, optional
        Whether to show region names in annotations
    custom_region_colors : dict, optional
        Custom color mapping for regions
    """
    set_visualization_style()
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    
    # Create main graph and regions panel
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax_regions = fig.add_subplot(gs[1])
    
    # Get employee color
    employee_colors = get_employee_colors()
    base_color = employee_colors.get(emp_id, '#1f77b4')
    
    # Define our standard x-axis range (7:30PM to 3:00PM next day)
    slots_per_day = 24 * 60 // time_slot_minutes
    evening_slot = 19 * 60 // time_slot_minutes + 2  # 7:30 PM (19:30)
    afternoon_slot = 15 * 60 // time_slot_minutes  # 3:00 PM (15:00)
    
    # Calculate total range for plotting (for next day, add slots_per_day)
    x_min = evening_slot
    x_max = afternoon_slot + slots_per_day if afternoon_slot < evening_slot else afternoon_slot
    
    # Dictionary to track region data for bottom panel
    region_data_by_slot = {}
    max_minutes = 0
    
    # For saving data
    all_plot_data = []
    
    # Process each shift
    all_shift_curves = []
    
    for shift in shifts:
        # Filter data for this shift
        shift_data = emp_data[emp_data['shift'] == shift].copy()
        if shift_data.empty:
            continue
        
        # Process date and time to create continuous timeline
        shift_data, slots_per_day, time_slot_labels = prepare_continuous_timeline(shift_data, time_slot_minutes)
        
        # Group data by continuous slot
        slot_stats = defaultdict(lambda: {'duration': 0, 'count': 0, 'regions': [], 'region_durations': defaultdict(float)})
        
        for _, row in shift_data.iterrows():
            slot = int(row['continuous_slot'])
            time_slot = int(row['time_slot_idx'])
            
            # Map continuous slot to our standard viewing window
            # For slots after midnight, we need to map them to the "next day" in our viewing window
            if time_slot < evening_slot:
                # This is a next-morning slot, shift it for continuity in plotting
                display_slot = time_slot + slots_per_day
            else:
                display_slot = time_slot
            
            # Update stats
            slot_stats[display_slot]['duration'] += row['duration']
            slot_stats[display_slot]['count'] += 1
            slot_stats[display_slot]['regions'].append(row['region'])
            slot_stats[display_slot]['region_durations'][row['region']] += row['duration']
        
        # Convert to dataframe
        slot_data = []
        for slot, stats in slot_stats.items():
            # Find most common region by duration
            if stats['region_durations']:
                most_common = max(stats['region_durations'].items(), key=lambda x: x[1])
                region, region_duration = most_common
            else:
                region, region_duration = None, 0
            
            # Store for regions panel
            if slot not in region_data_by_slot or region_duration > region_data_by_slot.get(slot, {}).get('duration', 0):
                region_data_by_slot[slot] = {
                    'region': region,
                    'duration': region_duration,
                    'shift': shift
                }
            
            slot_data.append({
                'slot': slot,
                'total_duration': stats['duration'],
                'walk_count': stats['count'],
                'most_common_region': region,
                'region_duration': region_duration
            })
        
        if not slot_data:
            continue
            
        df = pd.DataFrame(slot_data)
        df['duration_minutes'] = df['total_duration'] / 60
        df['region_duration_minutes'] = df['region_duration'] / 60
        df['shift'] = shift
        
        # Add to plot data for saving
        all_plot_data.append(df)
        
        # Update maximum minutes for y-axis scaling
        max_minutes = max(max_minutes, df['duration_minutes'].max())
        
        # Get shift color
        color = get_shift_color(shift, base_color)
        
        # Create smoothed curve - sort first to ensure correct order
        df = df.sort_values('slot')
        
        # Create continuous series for smoother curves
        slots = np.array(df['slot'])
        minutes = np.array(df['duration_minutes'])
        
        # Interpolate for smoother curve
        from scipy.interpolate import interp1d
        
        if len(slots) > 1:
            # Create more points for smoother curve
            x_smooth = np.linspace(slots.min(), slots.max(), 300)
            
            # Cubic interpolation for smoothness
            try:
                f = interp1d(slots, minutes, kind='cubic', bounds_error=False, fill_value=0)
                y_smooth = f(x_smooth)
                # Remove negative values from interpolation
                y_smooth = np.maximum(y_smooth, 0)
            except:
                # Fallback to linear if cubic fails (can happen with few points)
                f = interp1d(slots, minutes, kind='linear', bounds_error=False, fill_value=0)
                y_smooth = f(x_smooth)
            
            # Store curve data for finding peaks later
            all_shift_curves.append((shift, color, x_smooth, y_smooth, df))
            
            # Plot the curve
            ax.plot(x_smooth, y_smooth, '-', color=color, linewidth=3, alpha=0.9, label=f"Shift {shift}")
            
            # Add fill below
            ax.fill_between(x_smooth, y_smooth, color=color, alpha=0.2)
    
    # Add region markers at peaks
    for shift, color, x_smooth, y_smooth, df in all_shift_curves:
        # Find peak positions
        from scipy.signal import find_peaks
        
        # Get peak indices - require minimum height and separation
        peak_indices, _ = find_peaks(y_smooth, height=1.0, distance=20)
        
        if len(peak_indices) > 0:
            # Sort peaks by height and take top 2
            peak_heights = [(i, y_smooth[i]) for i in peak_indices]
            peak_heights.sort(key=lambda x: x[1], reverse=True)
            top_peaks = peak_heights[:2]
            
            for idx, height in top_peaks:
                if height > 1.0:  # Only label significant peaks
                    x_val = x_smooth[idx]
                    
                    # Find region at this peak
                    nearest_slot = (np.abs(df['slot'] - x_val)).idxmin()
                    region = df.loc[nearest_slot, 'most_common_region']
                    
                    # Add marker
                    ax.scatter(x_val, height, s=80, color=color, edgecolor='white', zorder=10)
                    
                    # Add label with time info and optionally region
                    label_text = f"{height:.1f}m"
                    if show_region_labels and region:
                        label_text += f"\n({region})"
                        
                    ax.annotate(
                        label_text,
                        xy=(x_val, height),
                        xytext=(0, 20),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontweight='bold',
                        color='black',
                        # No bbox for cleaner look
                        arrowprops=dict(arrowstyle='->', color=color)
                    )
    
    # Plot regions in bottom panel
    # Get unique regions
    all_regions = set()
    max_region_minutes = 0
    region_bar_data = []
    
    for slot, slot_data in region_data_by_slot.items():
        if slot_data['region']:
            all_regions.add(slot_data['region'])
            max_region_minutes = max(max_region_minutes, slot_data['duration'] / 60)
            
            # Add data for saving
            if x_min <= slot <= x_max:
                region_bar_data.append({
                    'slot': slot,
                    'region': slot_data['region'],
                    'minutes': slot_data['duration'] / 60,
                    'shift': slot_data['shift'],
                    'time_of_day': time_slot_labels[slot % slots_per_day]
                })
    
    # Get colors for regions
    region_colors = {}
    for region in all_regions:
        # First check custom colors
        color = get_region_color(region, custom_region_colors)
        if color:
            region_colors[region] = color
    
    # Fill in any missing regions with default color scheme
    missing_regions = [r for r in all_regions if r not in region_colors]
    if missing_regions:
        color_cycle = plt.cm.tab20(np.linspace(0, 1, len(missing_regions)))
        for i, region in enumerate(sorted(missing_regions)):
            region_colors[region] = color_cycle[i]
    
    # Plot region bars
    for slot, slot_data in region_data_by_slot.items():
        if x_min <= slot <= x_max:
            region = slot_data['region']
            duration_mins = slot_data['duration'] / 60
            
            if region:
                color = region_colors.get(region, 'gray')
                ax_regions.bar(
                    slot,
                    duration_mins,
                    width=0.8,
                    color=color,
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=0.5
                )
    
    # Set x-axis limits to our standard range
    ax.set_xlim(x_min, x_max)
    ax_regions.set_xlim(x_min, x_max)
    
    # Set y-axis limits
    ax.set_ylim(0, max_minutes * 1.2)
    ax_regions.set_ylim(0, max_region_minutes * 1.2)
    
    # Configure x-axis ticks
    span = x_max - x_min
    tick_every = max(1, span // 12)  # Around 12 ticks
    
    tick_positions = range(x_min, x_max + 1, tick_every)
    
    # Create x-tick labels that wrap back to time of day
    x_labels = []
    for pos in tick_positions:
        # Convert back to time of day
        time_idx = pos % slots_per_day
        x_labels.append(time_slot_labels[time_idx])
    
    # Hide x-ticks on top plot
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([])
    
    # Show x-ticks on bottom plot
    ax_regions.set_xticks(tick_positions)
    ax_regions.set_xticklabels(x_labels, rotation=45)
    
    # Set labels
    ax.set_title(f"Walking Patterns Across Shifts - Employee {emp_id}", 
                fontsize=18, fontweight='bold', color='green')
    ax.set_ylabel("Minutes Spent Walking", fontsize=14, fontweight='bold')
    
    ax_regions.set_ylabel("Minutes", fontsize=12)
    ax_regions.set_xlabel("Time of Day", fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax_regions.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add legends
    legend = ax.legend(title="Shift", loc="upper right", frameon=True, 
                      fontsize=12, fancybox=True, shadow=True)
    legend.get_title().set_fontweight('bold')
    
    # Add region legend - horizontal at bottom
    if region_colors:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=region) 
                          for region, color in region_colors.items()]
        
        # Create new axis for legend
        ax_legend = fig.add_axes([0.1, 0.01, 0.8, 0.03])
        ax_legend.axis('off')
        ax_legend.legend(
            handles=legend_elements,
            loc='center',
            title="Most Visited Regions",
            ncol=min(5, len(region_colors)),
            mode="expand",
            borderaxespad=0.1,
            frameon=True
        )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    if output_dir:
        # Save plot
        plot_path = output_dir / f"employee_{emp_id}_walking_pattern.png"
        save_figure(fig, plot_path)
        
        # Save data
        if all_plot_data:
            # Combine data from all shifts
            combined_df = pd.concat(all_plot_data)
            
            # Save density plot data
            density_path = output_dir / f"employee_{emp_id}_density_data.csv"
            combined_df.to_csv(density_path, index=False)
            
            # Save bar plot data
            bar_df = pd.DataFrame(region_bar_data)
            bar_path = output_dir / f"employee_{emp_id}_region_bars_data.csv"
            bar_df.to_csv(bar_path, index=False)
    
    plt.close()
    return fig

def create_region_activity_bar_plot(data, employee_id, time_slot_minutes=30, output_dir=None, 
                                  language='en', custom_region_colors=None, show_all_regions=True):
    """
    Create a bar plot showing region activity over time with stacked regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    employee_id : str
        Employee ID or None for all employees
    time_slot_minutes : int
        Size of time slots in minutes
    output_dir : Path
        Output directory
    language : str
        Language code ('en' or 'de')
    custom_region_colors : dict
        Custom color mapping for regions
    show_all_regions : bool
        Whether to show all regions or just the most common one per slot
    """
    set_visualization_style()
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    
    # Filter walking data
    walking_data = data[data['activity'] == 'Walk'].copy()
    
    # Filter by employee if specified
    if employee_id:
        walking_data = walking_data[walking_data['id'] == employee_id]
    
    if walking_data.empty:
        print(f"No walking data found for employee {employee_id}")
        return None
    
    # Process date and time to create continuous timeline
    walking_data, slots_per_day, time_slot_labels = prepare_continuous_timeline(walking_data, time_slot_minutes)
    
    # Define our standard x-axis range (7:30PM to 3:00PM next day)
    evening_slot = 19 * 60 // time_slot_minutes + 2  # 7:30 PM (19:30)
    afternoon_slot = 15 * 60 // time_slot_minutes  # 3:00 PM (15:00)
    
    # Calculate total range for plotting (for next day, add slots_per_day)
    x_min = evening_slot
    x_max = afternoon_slot + slots_per_day if afternoon_slot < evening_slot else afternoon_slot
    
    # Group by slot and region to get accurate totals per region
    slot_region_data = defaultdict(lambda: defaultdict(float))
    
    for _, row in walking_data.iterrows():
        time_slot = int(row['time_slot_idx'])
        
        # Map time slot to our standard viewing window
        if time_slot < evening_slot:
            # This is a next-morning slot, shift it for continuity
            display_slot = time_slot + slots_per_day
        else:
            display_slot = time_slot
            
        region = row['region']
        duration = row['duration']
        
        # Accumulate durations by slot and region
        slot_region_data[display_slot][region] += duration
    
    # Convert to list for plotting
    all_regions = set()
    for slot_data in slot_region_data.values():
        all_regions.update(slot_data.keys())
    
    # Get colors for regions
    region_colors = {}
    for region in all_regions:
        # First check custom colors
        color = get_region_color(region, custom_region_colors)
        if color:
            region_colors[region] = color
    
    # Fill in any missing regions with default color scheme
    missing_regions = [r for r in all_regions if r not in region_colors]
    if missing_regions:
        color_cycle = plt.cm.tab20(np.linspace(0, 1, len(missing_regions)))
        for i, region in enumerate(sorted(missing_regions)):
            region_colors[region] = color_cycle[i]
    
    # Create data for saving
    stacked_data = []
    
    if show_all_regions:
        # Create stacked bar chart showing all regions per slot
        if all_regions:
            # Sort regions by total time for consistent stacking order
            region_totals = {region: sum(slot_data.get(region, 0) for slot_data in slot_region_data.values())
                            for region in all_regions}
            sorted_regions = sorted(all_regions, key=lambda r: region_totals.get(r, 0), reverse=True)
        else:
            sorted_regions = []  # Ensure sorted_regions is defined even if all_regions is empty

        # Ensure plotted_regions is defined to avoid errors
        plotted_regions = set()

        # For each slot, plot stacked bars for each region that has data
        bottom = np.zeros(x_max + 1)  # Initialize bottom positions for stacking

        for region in sorted_regions:
            if region in plotted_regions:
                # Create array of values for this region across all slots
                heights = np.zeros(x_max + 1)

                for slot in range(x_min, x_max + 1):
                    if slot in slot_region_data and region in slot_region_data[slot]:
                        duration_mins = slot_region_data[slot][region] / 60
                        heights[slot] = duration_mins

                        # Record data for saving
                        if duration_mins > 0:
                            stacked_data.append({
                                'slot': slot,
                                'region': region,
                                'minutes': duration_mins,
                                'time_of_day': time_slot_labels[slot % slots_per_day]
                            })

                # Plot bars for this region
                color = region_colors.get(region, 'gray')
                slots = np.arange(len(heights))
                mask = heights > 0  # Only plot non-zero values

                if np.any(mask):
                    ax.bar(
                        slots[mask],
                        heights[mask],
                        bottom=bottom[mask],
                        width=0.8,
                        color=color,
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.5,
                        label=region
                    )

                    # Update bottom position for next region
                    bottom += heights
    else:
        # Original behavior - show only top region per slot
        for slot in range(x_min, x_max + 1):
            if slot in slot_region_data:
                # Find top region for this slot
                regions = [(region, duration) for region, duration in slot_region_data[slot].items()]
                if regions:
                    top_region, duration = max(regions, key=lambda x: x[1])
                    duration_mins = duration / 60
                    
                    color = region_colors.get(top_region, 'gray')
                    ax.bar(
                        slot,
                        duration_mins,
                        width=0.8,
                        color=color,
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.5
                    )
                    
                    # Record data for saving
                    stacked_data.append({
                        'slot': slot,
                        'region': top_region,
                        'minutes': duration_mins,
                        'time_of_day': time_slot_labels[slot % slots_per_day]
                    })
    
    # Set x-axis range
    ax.set_xlim(x_min, x_max)
    
    # Configure x-axis ticks
    span = x_max - x_min
    tick_every = max(1, span // 12)  # Around 12 ticks
    
    tick_positions = range(x_min, x_max + 1, tick_every)
    
    # Create x-tick labels that wrap back to time of day
    x_labels = []
    for pos in tick_positions:
        # Convert back to time of day
        time_idx = pos % slots_per_day
        x_labels.append(time_slot_labels[time_idx])
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_labels, rotation=45)
    
    # Set labels
    title = f"Region Walking Activity - Employee {employee_id}" if employee_id else "Region Walking Activity - All Employees"
    if not show_all_regions:
        title += " (Top Region per Slot)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Minutes", fontsize=12)
    ax.set_xlabel("Time of Day", fontsize=12)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add region legend
    if region_colors:
        # Only include regions that actually got plotted
        plotted_regions = {item['region'] for item in stacked_data}
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=region_colors.get(region, 'gray'), alpha=0.8)
            for region in sorted_regions if region in plotted_regions
        ]
        legend_labels = [region for region in sorted_regions if region in plotted_regions]
        
        if show_all_regions:
            # For stacked bars, legend is automatic if we used 'label'
            pass
        else:
            # For single top region, add manual legend
            ax.legend(
                legend_elements,
                legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=min(5, len(plotted_regions)),
                title="Most Visited Regions"
            )
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    if output_dir:
        # Save plot
        if employee_id:
            save_path = output_dir / f"employee_{employee_id}_region_bars{'_stacked' if show_all_regions else ''}.png"
        else:
            save_path = output_dir / f"all_employees_region_bars{'_stacked' if show_all_regions else ''}.png"
        save_figure(fig, save_path)
        
        # Save data
        if stacked_data:
            bar_df = pd.DataFrame(stacked_data)
            if employee_id:
                data_path = output_dir / f"employee_{employee_id}_region_bars_data.csv"
            else:
                data_path = output_dir / "all_employees_region_bars_data.csv"
            bar_df.to_csv(data_path, index=False)
    
    plt.close()
    return fig

def analyze_walking_by_shift(data, output_dir=None, time_slot_minutes=30, language='en', 
                          show_region_labels=False, custom_region_colors=None, stacked_bars=True):
    """
    Analyze walking patterns by shift for each employee with improved visualizations
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    output_dir : Path
        Output directory
    time_slot_minutes : int
        Size of time slots in minutes
    language : str
        Language code ('en' or 'de')
    show_region_labels : bool
        Whether to show region names in peak annotations
    custom_region_colors : dict
        Custom color mapping for regions
    stacked_bars : bool
        Whether to show stacked bars for all regions (True) or just top region (False)
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
        
        # Add separate directory for bar plots
        bar_dir = output_dir / 'region_bars'
        bar_dir.mkdir(exist_ok=True, parents=True)
        
        # Add directory for data files
        data_dir = output_dir / 'data'
        data_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each employee
    results = {}
    employees_processed = []
    
    for emp_id in walking_data['id'].unique():
        emp_data = walking_data[walking_data['id'] == emp_id]
        shifts = emp_data['shift'].unique()
        
        # Always process each employee
        # Create visualization comparing shifts
        create_shift_comparison_plot(
            emp_data, 
            emp_id, 
            shifts, 
            time_slot_minutes,
            shift_dir,
            language,
            show_region_labels,
            custom_region_colors
        )
        
        # Create standalone bar plot for regions
        create_region_activity_bar_plot(
            emp_data,
            emp_id,
            time_slot_minutes,
            bar_dir,
            language,
            custom_region_colors,
            show_all_regions=stacked_bars
        )
        
        # Store results
        results[emp_id] = {
            'shifts': shifts.tolist(),
            'shift_counts': emp_data.groupby('shift').size().to_dict()
        }
        
        employees_processed.append(emp_id)
    
    print(f"Created walking pattern visualizations for {len(employees_processed)} employees: {', '.join(employees_processed)}")
    print(f"Saved density plots to {shift_dir}/")
    print(f"Saved region bar plots to {bar_dir}/")
    print(f"Saved data files to {data_dir}/")
    
    # Create combined bar plot for all employees
    create_region_activity_bar_plot(
        walking_data,
        None,  # None means all employees
        time_slot_minutes,
        bar_dir,
        language,
        custom_region_colors,
        show_all_regions=stacked_bars
    )
    
    return results

def validate_region_visualization(data, hour, employee_id=None, output_path=None):
    """
    Validate region visualization by comparing with raw data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    hour : int
        Hour to validate (0-23)
    employee_id : str, optional
        Employee ID to filter on, or None for all employees
    output_path : str, optional
        Path to save validation data
        
    Returns:
    --------
    pandas.DataFrame
        Summary of walking by region for the specified hour
    """
    # Filter for walking activity
    walking_data = data[data['activity'] == 'Walk'].copy()
    
    # Filter by employee if specified
    if employee_id:
        walking_data = walking_data[walking_data['id'] == employee_id]
    
    # Extract hour from startTime
    walking_data['hour'] = (walking_data['startTime'] / 3600) % 24
    walking_data['hour'] = walking_data['hour'].astype(int)
    
    # Filter for specified hour
    target_data = walking_data[walking_data['hour'] == hour]
    
    # Group by region and calculate total duration
    region_stats = target_data.groupby('region').agg({
        'duration': ['sum', 'count'],
        'date': 'nunique',
    }).reset_index()
    
    # Flatten column names
    region_stats.columns = ['region', 'total_minutes', 'count', 'days']
    region_stats['total_minutes'] = region_stats['total_minutes'] / 60  # Convert to minutes
    
    # Sort by duration
    region_stats = region_stats.sort_values('total_minutes', ascending=False)
    
    # Save if path provided
    if output_path:
        region_stats.to_csv(output_path, index=False)
        print(f"Saved validation data to {output_path}")
    
    return region_stats