#!/usr/bin/env python3
"""
Employee Activity Region Analysis - Step 2
==========================================

Generate 35 graphs (7 employees × 5 activities) showing time spent by employee and activity
across 30-minute time slots, with bars colored by top region and labeled with region info.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import subprocess
import sys

# Import region colors from base
from src.visualization.base import get_region_color, get_region_colors

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 10

# Get region colors from centralized location
REGION_COLORS = get_region_colors()

# Activity types
ACTIVITY_TYPES = ['Walk', 'Stand', 'Handle center', 'Handle up', 'Handle down']

ACTIVITY_TYPE_GERMAN_PLURAL = {
    'Walk': 'Spaziergänge',
    'Stand': 'Stehzeiten',
    'Handle center': 'zentralen Handhabungen',
    'Handle up': 'Handhabungen nach oben',
    'Handle down': 'Handhabungen nach unten'
}

def check_data_files():
    """Check if the required CSV files exist"""
    print("🔍 Checking for required data files...")
    
    missing_files = []
    
    for activity_type in ACTIVITY_TYPES:
        filename = f"output/{activity_type.lower().replace(' ', '_')}_region_stats.csv"
        if not os.path.exists(filename):
            missing_files.append(filename)
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Run extract_data.py first to generate these files")
        return False
    
    print("✅ All required data files found")
    return True

def run_data_extraction():
    """Run the data extraction scripts if needed"""
    print("🔄 Checking if data extraction is needed...")
    
    # Check if we already have the files
    if check_data_files():
        print("✅ Data files already exist, skipping extraction")
        return True
    
    print("📊 Running data extraction scripts...")
    
    try:
        print("  📊 Running extract_data.py...")
        result = subprocess.run([sys.executable, 'extract_data.py'], 
                              capture_output=True, text=True, check=True)
        print("  ✅ extract_data.py completed successfully")
        
        print("  📊 Running plot_data.py...")
        result = subprocess.run([sys.executable, 'plot_data.py'], 
                              capture_output=True, text=True, check=True)
        print("  ✅ plot_data.py completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error running scripts: {e}")
        print(f"     Output: {e.stdout}")
        print(f"     Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  ❌ Could not find extract_data.py or plot_data.py")
        print("  💡 Make sure these scripts are in the current directory")
        return False
    
    return True

def get_region_color(region_name):
    """Get color for a region using the centralized color mapping"""
    from src.visualization.base import get_region_color as base_get_region_color
    # Use the imported function from base.py which handles all fallback logic
    return base_get_region_color(region_name)

def format_duration_mmss(total_minutes):
    """Convert minutes to mm:ss format"""
    minutes = int(total_minutes)
    seconds = int((total_minutes - minutes) * 60)
    return f"{minutes:02d}:{seconds:02d}"

def format_duration_hms_from_minutes(total_minutes):
    """Convert total minutes to hh:mm:ss format."""
    if pd.isna(total_minutes) or total_minutes < 0:
        return "00:00:00"
    total_seconds = int(total_minutes * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def load_activity_data(activity_type):
    """Load data for a specific activity type"""
    file_path = f"output/{activity_type.lower().replace(' ', '_')}_region_stats.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Warning: {file_path} not found")
        return None
    
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Loaded {len(data)} records for {activity_type}")
        return data
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def create_employee_activity_graph(data, employee_id, activity_type, output_dir, activity_duration_data, 
                                dpi=300, figure_size=(12, 8), line_width=2.0, font_size=10,
                                show_legend=True, show_statement=True):
    """Create a graph for one employee and one activity"""
    
    emp_data_for_graph = data[data['id'] == employee_id].copy() # Used for overall top region summary
    
    # Filter data for this employee (for plotting time categories)
    emp_plot_data_source = data[data['id'] == employee_id].copy()
    
    if emp_plot_data_source.empty:
        print(f"⚠️  No data for employee {employee_id} in {activity_type}")
        return
    
    # List of regions to specifically plot (target regions)
    target_regions_to_plot = [
        '1_Fettbacken',
        '2a_Konditorei_station',
        '2b_Konditorei_station', 
        '2c_Konditorei_station',
        '3_Konditorei_deco_raum',
        '4a_Brotstation',
        '4b_Brotstation',
        '4c_Brotstation',
        '4d_Brotstation',
        '5a_Stickenofen',
        '5b_Stickenofen',
        '5c_Etagenofen',
        '5d_Etagenofen',
        '6a_Mehl_Brötchen_Maschine',
        '6b_Mehl_Brötchen_Maschine',
        '6c_Mehl_Brötchen_Maschine',
        '7a_Brotwagon_und_kisten',
        '7b_Brotwagon_und_kisten',
        '7c_Brotwagon_und_kisten',
        '7d_Brotwagon_und_kisten',
        '7e_Brotwagon_und_kisten',
        '7f_Brotwagon_und_kisten',
        '8_Korridor',
        '9_Kekse_Verpackung',
    ]
    
    # For each time category, find the top region that is in our target_regions_to_plot list
    processed_data = []
    
    for time_cat in emp_plot_data_source['time_category'].unique():
        time_data = emp_plot_data_source[emp_plot_data_source['time_category'] == time_cat]
        
        # Sort the data by total_minutes in descending order
        sorted_data = time_data.sort_values('total_minutes', ascending=False)
        
        # Find the first region that IS IN the target_regions_to_plot list
        found_target_region = False
        
        for _, row in sorted_data.iterrows():
            region = row['region']
            if region in target_regions_to_plot:
                # Found a region that is in our target list
                top_target_region = region
                top_target_region_minutes = row['total_minutes']
                found_target_region = True
                break
        
        # Only add this time category if we found a region from our target list
        if found_target_region:
            processed_data.append({
                'time_category': time_cat,
                'top_region': top_target_region, # Note: this is the top *target* region
                'top_region_minutes': top_target_region_minutes
            })
    
    # Convert to DataFrame and sort by time
    plot_data = pd.DataFrame(processed_data)
    
    # Sort by time (convert to minutes since midnight for proper sorting)
    def time_to_minutes(time_str):
        h, m = time_str.split(':')
        return int(h) * 60 + int(m)
    
    plot_data['time_minutes'] = plot_data['time_category'].apply(time_to_minutes)
    plot_data = plot_data.sort_values('time_minutes')
    
    # Set up the figure with configurable size and DPI
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['lines.linewidth'] = line_width
    plt.rcParams['font.size'] = font_size
    
    # Create figure with configurable size
    fig, ax = plt.subplots(figsize=figure_size)
    plt.subplots_adjust(bottom=0.64) # Significantly increased bottom margin
    
    # Get colors for each bar based on top region
    colors = [get_region_color(region) for region in plot_data['top_region']]
    
    # Create bars using only the top region's time for each time slot
    bars = ax.bar(range(len(plot_data)), plot_data['top_region_minutes'], color=colors, alpha=0.8)
    
    # Create legend if enabled
    if show_legend:
        unique_regions_in_plot = plot_data[plot_data['top_region'].isin(target_regions_to_plot)]['top_region'].unique()
        if len(unique_regions_in_plot) > 0:
            legend_handles = [patches.Patch(facecolor=get_region_color(region), edgecolor='grey', label=region) 
                            for region in unique_regions_in_plot if region is not None and region != '']
            ax.legend(handles=legend_handles, title="Regions", loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Set labels and title
    ax.set_title(f"Employee {employee_id} - {activity_type}\nTop Region Time by 30-minute Intervals", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Time of Day (30-minute intervals)", fontsize=12)
    ax.set_ylabel("Minutes in Top Region", fontsize=12)
    
    # Set x-axis labels
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data['time_category'], rotation=45, ha='right')
    
    # Add grid
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Adjust the layout to make room for the statement
    plt.tight_layout(rect=[0.03, 0.15, 0.97, 0.95])  # left, bottom, right, top
    
    # Position the main plot area with more space at the bottom
    # left, bottom, width, height
    ax.set_position([0.1, 0.35, 0.85 if show_legend else 0.9, 0.6])  
    
    # Position the x-axis label lower to make room for the statement
    ax.xaxis.set_label_coords(0.5, -0.35)
    
    # --- Generate dynamic analytical sentence --- 
    sentence_part1 = ""
    sentence_part2 = "Keine Top-Zielregionen für diese Aktivität gefunden."

    # Part 1: Short duration activities
    try:
        if activity_duration_data is not None:
            duration_row = activity_duration_data[
                (activity_duration_data['employee_id'] == employee_id) & 
                (activity_duration_data['activity'] == activity_type)
            ]
            if not duration_row.empty:
                percentage_0_5_sec = duration_row['0-5 sec'].iloc[0]
                german_activity_name = ACTIVITY_TYPE_GERMAN_PLURAL.get(activity_type, activity_type)
                try:
                    percentage_0_5_sec_float = float(percentage_0_5_sec)
                    sentence_part1 = f"{percentage_0_5_sec_float:.1f}% der {german_activity_name} dauern weniger als 5 Sekunden."
                except ValueError:
                    sentence_part1 = f"{percentage_0_5_sec} der {german_activity_name} dauern weniger als 5 Sekunden (Formatierungsfehler bei Prozent)."
            else:
                sentence_part1 = f"Keine Kurzzeitdaten für {activity_type} gefunden."
        else:
            sentence_part1 = "Kurzzeitdaten-Datei nicht geladen."
    except Exception as e:
        print(f"Error generating sentence part 1 for {employee_id}, {activity_type}: {e}")
        sentence_part1 = "Fehler bei Kurzzeitdaten-Analyse."

    # Part 2: Top N target regions for this employee and activity
    try:
        if not emp_data_for_graph.empty:
            # Calculate total time per region for this employee & activity
            emp_activity_region_summary = emp_data_for_graph.groupby('region')['total_minutes'].sum().reset_index()
            
            # Filter by target_regions_to_plot (defined earlier in this function)
            target_region_summary = emp_activity_region_summary[emp_activity_region_summary['region'].isin(target_regions_to_plot)]
            
            # Sort by total_minutes and get top 3
            top_target_regions_df = target_region_summary.sort_values('total_minutes', ascending=False).head(3)
            
            if not top_target_regions_df.empty:
                total_activity_time_for_employee = emp_data_for_graph['total_minutes'].sum()
                region_strings = []
                for _, row in top_target_regions_df.iterrows():
                    region_name = row['region']
                    region_minutes = row['total_minutes']
                    region_percentage = (region_minutes / total_activity_time_for_employee * 100) if total_activity_time_for_employee > 0 else 0
                    region_time_hms = format_duration_hms_from_minutes(region_minutes)
                    region_strings.append(f"{region_name} - {region_percentage:.1f}% ({region_time_hms})")
                
                num_top_regions = len(region_strings)
                if num_top_regions == 1:
                    sentence_part2 = f"Top Region mit der meisten Zeit bei dieser Aktivität: {', '.join(region_strings)}."
                else:
                    sentence_part2 = f"Top {num_top_regions} Regionen mit der meisten Zeit bei dieser Aktivität: {', '.join(region_strings)}."
    except Exception as e:
        print(f"Error generating sentence part 2 for {employee_id}, {activity_type}: {e}")
        # sentence_part2 remains default

    # Add x-axis label with custom positioning
    ax.set_xlabel("Time of Day (30-minute intervals)", fontsize=12, labelpad=15)  # Increased labelpad
    
    # Add statement if enabled
    if show_statement:
        dynamic_sentence = f"{sentence_part1} {sentence_part2}"
        # Position the statement below the x-axis label
        # Using figure coordinates (0-1), with 0.05 from left and 0.05 from bottom
        fig.text(0.05, 0.02,  # x, y in figure coordinates (0-1)
                dynamic_sentence, 
                ha='left',
                va='top',
                fontsize=font_size, 
                style='italic',
                wrap=True,
                transform=fig.transFigure,
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.3, linewidth=0.5))

    # Save the plot with high quality settings and proper bounding box
    plot_filename = f"{employee_id}_{activity_type.replace(' ', '_').lower()}_regions.png"
    plot_filepath = output_dir / plot_filename
    try:
        # Get all figure artists that need to be included in the bbox
        all_artists = [fig, ax] + ax.get_legend_handles_labels()[0]
        
        # Save with tight layout and including all artists
        plt.savefig(plot_filepath, 
                   dpi=dpi,
                   format='png',
                   bbox_extra_artists=all_artists,
                   bbox_inches='tight',
                   pad_inches=0.5,  # Increased padding
                   facecolor=fig.get_facecolor(),
                   edgecolor='none')
        print(f"✅ Saved high-quality graph ({dpi} DPI): {plot_filepath}")
    except Exception as e:
        print(f"❌ Error saving graph {plot_filepath}: {e}")
    plt.close(fig) # Close the figure to free memory

    # --- Save graph data to CSV --- 
    try:
        csv_output_dir = output_dir / "graph_data_csvs"
        csv_output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        csv_filename = f"{employee_id}_{activity_type.replace(' ', '_').lower()}_graph_data.csv"
        csv_filepath = csv_output_dir / csv_filename
        
        # Prepare data for CSV: select relevant columns and add the note
        if not plot_data.empty:
            csv_data = plot_data[['time_category', 'top_region', 'top_region_minutes', 'time_minutes']].copy()
            csv_data['note'] = dynamic_sentence
            csv_data.to_csv(csv_filepath, index=False)
            # print(f"Saved CSV data: {csv_filepath}")
        else:
            # Create an empty CSV with note if plot_data was empty but sentence was generated
            # Or handle as per desired behavior for empty plot_data
            placeholder_df_for_note = pd.DataFrame({'note': [dynamic_sentence]})
            placeholder_df_for_note.to_csv(csv_filepath, index=False)
            print(f"Saved CSV with only note (empty plot_data): {csv_filepath}")

    except Exception as e:
        print(f"❌ Error saving CSV data for {employee_id} - {activity_type}: {e}")

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate employee activity analysis graphs with configurable quality settings')
    # Quality settings
    parser.add_argument('--dpi', type=int, default=300, 
                      help='DPI (dots per inch) for output images (default: 300)')
    parser.add_argument('--width', type=float, default=12.0,
                      help='Figure width in inches (default: 12.0)')
    parser.add_argument('--height', type=float, default=8.0,
                      help='Figure height in inches (default: 8.0)')
    parser.add_argument('--line-width', type=float, default=2.0,
                      help='Line width for plots (default: 2.0)')
    parser.add_argument('--font-size', type=int, default=10,
                      help='Base font size (default: 10)')
    
    # Display options
    parser.add_argument('--no-legend', action='store_true',
                      help='Hide the legend')
    parser.add_argument('--no-statement', action='store_true',
                      help='Hide the descriptive statement below the graph')
    
    return parser.parse_args()

def main():
    """Main function to generate all graphs"""
    args = parse_arguments()
    
    print("🏭 EMPLOYEE ACTIVITY REGION ANALYSIS - STEP 2")
    print("=" * 60)
    print(f"📐 Quality settings: DPI={args.dpi}, Figure size={args.width}x{args.height}",
          f"Line width={args.line_width}, Font size={args.font_size}")
    
    # Get output directory from environment variable or use default
    output_dir = Path(os.environ.get('STEP2_OUTPUT_DIR', 'step2_analysis_output'))
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📂 Saving output to: {output_dir.absolute()}")
    
    # Step 1: Run data extraction scripts
    if not run_data_extraction():
        print("❌ Failed to run data extraction scripts")
        return 1
    
    print(f"\n📊 Generating 35 graphs (7 employees × 5 activities)...")
    
    # Load activity duration data for dynamic sentences
    activity_duration_df = None
    try:
        duration_csv_path = Path('output/analysis/activity_duration_ranges.csv')
        if duration_csv_path.exists():
            activity_duration_df = pd.read_csv(duration_csv_path)
            print(f"✅Successfully loaded {duration_csv_path} for dynamic sentences.")
        else:
            print(f"⚠️  Warning: {duration_csv_path} not found. Dynamic sentences will use default text.")
    except Exception as e:
        print(f"❌ Error loading {duration_csv_path}: {e}. Dynamic sentences will use default text.")

    # Get list of employees (assuming they are consistent across activities)
    employees = ['32-A', '32-C', '32-D', '32-E', '32-F', '32-G', '32-H']
    
    # Generate graphs for each employee and activity combination
    graphs_created = 0
    
    for activity_type in ACTIVITY_TYPES:
        print(f"\n🎯 Processing {activity_type}...")
        
        # Load data for this activity
        activity_data = load_activity_data(activity_type)
        
        if activity_data is None:
            print(f"⚠️  Skipping {activity_type} due to missing data")
            continue
        
        # Create graph for each employee
        for employee_id in employees:
                create_employee_activity_graph(
                    activity_data, 
                    employee_id, 
                    activity_type, 
                    output_dir,
                    activity_duration_df if activity_duration_df is not None else None,
                    dpi=args.dpi,
                    figure_size=(args.width, args.height),
                    line_width=args.line_width,
                    font_size=args.font_size,
                    show_legend=not args.no_legend,
                    show_statement=not args.no_statement
                )
                graphs_created += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Created {graphs_created} graphs")
    print(f"💾 All graphs saved to: {output_dir}")
    print(f"\nEach graph shows:")
    print(f"  • X-axis: 30-minute time intervals")
    print(f"  • Y-axis: Total minutes spent on activity")
    print(f"  • Bar color: Top region for that time slot")  
    print(f"  • Bar height: Minutes spent in top region only")
    print(f"  • Legend: Shows regions present in the graph (top right)")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)