import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the format_seconds_to_hms function from time_utils.py
from src.utils.time_utils import format_seconds_to_hms, format_seconds_to_hours, format_duration
# Import region colors from base
from src.visualization.base import get_region_color, get_region_colors

# Get region colors from centralized location
REGION_COLORS = get_region_colors()

def load_and_prepare_data():
    """Load cart data and prepare it for analysis."""
    # Get paths from environment variables with fallbacks
    cart_data_dir = Path(os.environ.get('CART_DATA_DIR', 'output/cart_data'))
    idle_analysis_dir = Path(os.environ.get('IDLE_ANALYSIS_DIR', 'output/cart_analysis'))
    
    # Look for idle times data in the analysis directory
    idle_file = idle_analysis_dir / 'cart_idle_periods.csv'
    if not idle_file.exists():
        print(f"❌ Error: Idle analysis data not found at {idle_file}")
        print("   Please make sure idle_cart.py has been run first")
        return None, None
    
    print("\n📊 Loading idle times data...")
    try:
        idle_df = pd.read_csv(idle_file)
        # Ensure duration is numeric
        idle_df['duration'] = pd.to_numeric(idle_df['duration'], errors='coerce')
        print(f"   Loaded {len(idle_df)} idle periods")
    except Exception as e:
        print(f"❌ Error loading idle data: {str(e)}")
        return None, None
    
    # Load the cart data files
    cart_files = list(cart_data_dir.glob("*_data.csv"))  # Look for files like 7004_data.csv
    
    if not cart_files:
        print(f"⚠️  No cart data files found in {cart_data_dir}/")
        print("   This is expected if you haven't run cart.py yet")
        return idle_df, None
    
    print("Loading cart activity data...")
    all_cart_data = []
    for file in cart_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                df['source_file'] = file.name
                # Ensure duration is numeric
                df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
                all_cart_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read {file}: {str(e)}")
    
    if not all_cart_data:
        print("No valid cart data found")
        return idle_df, None
    
    cart_df = pd.concat(all_cart_data, ignore_index=True)
    
    # Print data summaries
    print(f"\nIdle times data: {len(idle_df)} records")
    print(f"Cart activity data: {len(cart_df)} records")
    
    return idle_df, cart_df

def analyze_and_print_console_data(idle_df, cart_df):
    """Generate and print the good console data"""
    print("\n" + "="*50)
    print("=== Cart Idle Time Analysis ===")
    
    if idle_df is not None and not idle_df.empty:
        # Top regions by idle time
        region_stats = idle_df.groupby('region').agg(
            total_seconds=('duration', 'sum'),
            avg_seconds=('duration', 'mean'),
            count=('cartId', 'count')
        ).sort_values('total_seconds', ascending=False).head(10)
        
        # Format the output
        top_regions = region_stats.copy()
        top_regions['total_time'] = top_regions['total_seconds'].apply(format_seconds_to_hms)
        top_regions['avg_time'] = top_regions['avg_seconds'].apply(format_seconds_to_hms)
        top_regions['percentage'] = (top_regions['total_seconds'] / top_regions['total_seconds'].sum() * 100).round(1)
        
        print("\nTop 10 Regions by Idle Time:")
        print(top_regions[['total_time', 'avg_time', 'percentage', 'count']].to_string())
        
        # Idle time by shift
        shift_stats = idle_df.groupby('shift')['duration'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        shift_stats['total_time'] = shift_stats['sum'].apply(format_seconds_to_hms)
        shift_stats['avg_time'] = shift_stats['mean'].apply(format_seconds_to_hms)
        shift_stats['percentage'] = (shift_stats['sum'] / shift_stats['sum'].sum() * 100).round(1)
        
        print("\nIdle Time by Shift:")
        print(shift_stats[['total_time', 'avg_time', 'percentage', 'count']].to_string())
    
    print("\n=== Cart Activity Analysis ===")
    
    if cart_df is not None and not cart_df.empty:
        # Activity distribution
        if 'activity' in cart_df.columns:
            activity_counts = cart_df['activity'].value_counts().reset_index()
            activity_counts.columns = ['activity', 'count']
            activity_counts['percentage'] = (activity_counts['count'] / len(cart_df)) * 100
            activity_counts['total_time'] = cart_df.groupby('activity')['duration'].sum().reindex(activity_counts['activity']).values
            activity_counts['avg_time'] = cart_df.groupby('activity')['duration'].mean().reindex(activity_counts['activity']).values
            
            display_df = activity_counts.copy()
            display_df['total_time_formatted'] = display_df['total_time'].apply(format_seconds_to_hms)
            display_df['avg_time_formatted'] = display_df['avg_time'].apply(format_seconds_to_hms)
            
            print("\nActivity Distribution:")
            print(display_df[['activity', 'count', 'percentage', 'total_time_formatted', 'avg_time_formatted']].to_string())
        
        # Most active regions by time spent
        region_usage = cart_df.groupby('region').agg(
            total_seconds=('duration', 'sum'),
            count=('cartId', 'count')
        ).sort_values('total_seconds', ascending=False).head(10)
        
        region_usage['percentage'] = (region_usage['total_seconds'] / region_usage['total_seconds'].sum() * 100).round(1)
        region_usage['avg_duration'] = (region_usage['total_seconds'] / region_usage['count']).round(1)
        
        display_df = region_usage.copy()
        display_df['total_time'] = display_df['total_seconds'].apply(format_seconds_to_hms)
        display_df['avg_duration_formatted'] = display_df['avg_duration'].apply(format_seconds_to_hms)
        
        print("\nMost Active Regions by Time Spent:")
        print(display_df[['total_time', 'avg_duration_formatted', 'count', 'percentage']].to_string())
    
    return top_regions if idle_df is not None else None, region_usage if cart_df is not None else None

# Using get_region_color from base.py

def create_enhanced_donut_charts(idle_data, usage_data, output_dir):
    """Create enhanced donut charts with duration and percentage info"""
    if (idle_data is None or idle_data.empty) and (usage_data is None or usage_data.empty):
        return
    
    print("Creating enhanced donut charts...")
    
    # Create high-resolution figure with larger size and higher DPI
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['font.size'] = 16  # Increase base font size
    
    # Create figure with increased size and resolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))  # Increased figure size
    
    # Idle donut
    if idle_data is not None and not idle_data.empty:
        top_3_idle = idle_data.head(3)
        colors_idle = [get_region_color(region) for region in top_3_idle.index]
        
        # Create the donut with percentage and duration labels
        wedges, texts, autotexts = ax1.pie(top_3_idle['total_seconds'], 
                                          labels=top_3_idle.index,
                                          colors=colors_idle,
                                          autopct='',  # We'll customize this
                                          startangle=90,
                                          pctdistance=0.85,
                                          labeldistance=1.15,
                                          textprops={'fontsize': 16, 'fontweight': 'bold'})
        
        # Create donut by adding a white circle in the center
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.set_title('Top 3 Regions by Idle Time', fontsize=16, fontweight='bold', pad=20)
        
        # Add center text with total
        total_idle = top_3_idle['total_seconds'].sum()
        ax1.text(0, 0, f"Total Idle\n{format_seconds_to_hms(total_idle)}", 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Add custom labels with duration and percentage
        for i, (wedge, (region, data)) in enumerate(zip(wedges, top_3_idle.iterrows())):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.85 * np.cos(np.radians(angle))
            y = 0.85 * np.sin(np.radians(angle))
            
            # Create label with percentage and duration
            label_text = f"{data['percentage']:.1f}%\n{data['total_time']}"
            ax1.text(x, y, label_text, ha='center', va='center', 
                    fontsize=13, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
    
    # Usage donut
    if usage_data is not None and not usage_data.empty:
        top_3_usage = usage_data.head(3)
        colors_usage = [get_region_color(region) for region in top_3_usage.index]
        
        # Create the donut with percentage and duration labels
        wedges, texts, autotexts = ax2.pie(top_3_usage['total_seconds'], 
                                          labels=top_3_usage.index,
                                          colors=colors_usage,
                                          autopct='',  # We'll customize this
                                          startangle=90,
                                          pctdistance=0.85,
                                          labeldistance=1.15,
                                          textprops={'fontsize': 16, 'fontweight': 'bold'})
        
        # Create donut by adding a white circle in the center
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax2.add_artist(centre_circle)
        ax2.set_title('Top 3 Regions by Usage Time', fontsize=16, fontweight='bold', pad=20)
        
        # Add center text with total
        total_usage = top_3_usage['total_seconds'].sum()
        ax2.text(0, 0, f"Total Usage\n{format_seconds_to_hms(total_usage)}", 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Add custom labels with duration and percentage
        for i, (wedge, (region, data)) in enumerate(zip(wedges, top_3_usage.iterrows())):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 0.85 * np.cos(np.radians(angle))
            y = 0.85 * np.sin(np.radians(angle))
            
            # Create label with percentage and duration
            label_text = f"{data['percentage']:.1f}%\n{format_seconds_to_hms(data['total_seconds'])}"
            ax2.text(x, y, label_text, ha='center', va='center', 
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
    
    # Adjust layout and save with high quality settings
    plt.tight_layout(pad=4.0)  # Add more padding around the plots
    plt.savefig(output_dir / 'donut_charts_with_details.png', 
               dpi=900,  # Increased DPI for higher resolution
               bbox_inches='tight', 
               facecolor='white', 
               edgecolor='none',
               quality=95,  # Maximum quality for PNG
               optimize=True)
    plt.close()
    print(f"✅ High-quality donut charts saved to: {output_dir / 'donut_charts_with_details.png'}")

def export_donut_data_to_csv(idle_data, usage_data, output_dir):
    """Export the data used in donut charts to CSV files"""
    print("Exporting donut chart data to CSV...")
    
    if idle_data is not None and not idle_data.empty:
        # Get top 3 idle data
        top_3_idle = idle_data.head(3).copy()
        top_3_idle = top_3_idle.reset_index()
        
        # Clean up the data for export
        idle_export = top_3_idle[['region', 'total_time', 'percentage']].copy()
        idle_export.columns = ['Region', 'Duration', 'Percentage']
        
        # Save idle data
        idle_csv_path = output_dir / 'top_3_idle_regions_data.csv'
        idle_export.to_csv(idle_csv_path, index=False)
        print(f"✅ Idle regions data saved to: {idle_csv_path}")
    
    if usage_data is not None and not usage_data.empty:
        # Get top 3 usage data
        top_3_usage = usage_data.head(3).copy()
        top_3_usage = top_3_usage.reset_index()
        
        # Clean up the data for export
        usage_export = top_3_usage[['region', 'total_seconds', 'percentage']].copy()
        usage_export['Duration'] = usage_export['total_seconds'].apply(format_seconds_to_hms)
        usage_export = usage_export[['region', 'Duration', 'percentage']].copy()
        usage_export.columns = ['Region', 'Duration', 'Percentage']
        
        # Save usage data
        usage_csv_path = output_dir / 'top_3_usage_regions_data.csv'
        usage_export.to_csv(usage_csv_path, index=False)
        print(f"✅ Usage regions data saved to: {usage_csv_path}")
    
    # Also create a combined summary file
    if idle_data is not None and usage_data is not None and not idle_data.empty and not usage_data.empty:
        # Create combined data
        combined_data = []
        
        # Add idle data
        top_3_idle = idle_data.head(3)
        for region, data in top_3_idle.iterrows():
            combined_data.append({
                'Region': region,
                'Type': 'Idle Time',
                'Duration': data['total_time'],
                'Percentage': f"{data['percentage']:.1f}%"
            })
        
        # Add usage data
        top_3_usage = usage_data.head(3)
        for region, data in top_3_usage.iterrows():
            combined_data.append({
                'Region': region,
                'Type': 'Usage Time',
                'Duration': format_seconds_to_hms(data['total_seconds']),
                'Percentage': f"{data['percentage']:.1f}%"
            })
        
        # Create DataFrame and save
        combined_df = pd.DataFrame(combined_data)
        combined_csv_path = output_dir / 'donut_charts_combined_data.csv'
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"✅ Combined donut chart data saved to: {combined_csv_path}")

def main():
    # Get output directory from environment variable or use default
    output_dir = Path(os.environ.get('STEP3_OUTPUT_DIR', 'output/cart_analysis'))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Saving output to: {output_dir.absolute()}")
    
    print("🚛 CART ANALYSIS - ENHANCED DONUT CHARTS")
    print("=" * 60)
    print("Creating detailed donut charts with duration and percentage info")
    
    # Load data
    idle_df, cart_df = load_and_prepare_data()
    if idle_df is None and cart_df is None:
        print("❌ No data available for analysis")
        return
    
    # Generate and print the good console data
    idle_regions, usage_regions = analyze_and_print_console_data(idle_df, cart_df)
    
    # Create enhanced donut charts
    print("\n🍩 Creating enhanced donut charts...")
    create_enhanced_donut_charts(idle_regions, usage_regions, output_dir)
    
    # Export the data to CSV
    print("\n📊 Exporting donut chart data to CSV...")
    export_donut_data_to_csv(idle_regions, usage_regions, output_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Files saved to: {output_dir}")
    print("   • donut_charts_with_details.png - High-quality donut charts")
    print("   • top_3_idle_regions_data.csv - Idle regions data")
    print("   • top_3_usage_regions_data.csv - Usage regions data") 
    print("   • donut_charts_combined_data.csv - Combined data")
    print("\n🎯 Features:")
    print("   • Uses your fixed region colors")
    print("   • Shows duration and percentage on each segment") 
    print("   • Total time displayed in center")
    print("   • Clean, modern donut design")
    print("   • High-quality 600 DPI export")
    print("   • Large, readable fonts")
    print("   • All data exported as CSV files")

if __name__ == "__main__":
    main()