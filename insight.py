#!/usr/bin/env python3
"""
Bakery Activity Insight Generator

Analyzes pre-processed activity data to identify statistically significant patterns 
within specific employee shifts that warrant video review.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
import json
from datetime import datetime

# Configure visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Define activity types and colors
ACTIVITY_TYPES = ['handle_center', 'handle_up', 'handle_down', 'stand', 'walk']
ACTIVITY_COLORS = {
    'handle_center': '#F1A983',  # Light orange
    'handle_up': '#FFC000',      # Yellow
    'handle_down': '#C00000',    # Red
    'stand': '#A6A6A6',          # Grey
    'walk': '#156082',           # Blue
}

# Define colors for specific regions
REGION_COLORS = {
    'Fettbacken': '#6BAED6',            # Light blue
    'Konditorei_station': '#3182BD',    # Blue
    'konditorei_deco': '#08519C',       # Dark blue          # Green
    '1_Brotstation': 'lawngreen',         # Lighter green
    '2_Brotstation': 'forestgreen',         # Even darker green
    '3_Brotstation': 'darkgreen',         # Much darker green
    '4_Brotstation': 'mediumspringgreen',         # Very dark green
    'Mehlmaschine': '#4A8A50',          # Darker light green
    'brotkisten_regal': '#FFFF00',      # Yellow
    '1_Konditorei_station': 'powderblue',  # Light blue
    '2_Konditorei_station': 'deepskyblue',  # Medium blue
    '3_Konditorei_station': 'steelblue',  # Darker blue
    'bereitstellen_prepared_goods': '#FFD700',  # Gold
    '1_corridor': '#FFC107',                   # Amber
    'leere_Kisten': '#FFB300',                 # Darker yellow
}

def load_activity_data(base_dir, activity_type):
    """Load pre-processed activity data from CSV files"""
    file_path = os.path.join(base_dir, f"{activity_type}_region_stats.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None
    
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def identify_significant_activities(activity_data, activity_type, z_threshold=2.0, percentage_threshold=70):
    """Find statistically significant activity durations within each employee shift"""
    significant_activities = []
    
    # Group by employee and shift
    for (emp_id, shift), group in activity_data.groupby(['id', 'shift']):
        # Calculate statistics for this employee-shift combination
        mean_duration = group['total_minutes'].mean()
        std_duration = group['total_minutes'].std()
        
        # Skip if there's insufficient data or no variation
        if len(group) < 3 or pd.isna(std_duration) or std_duration == 0:
            continue
            
        # Find significant outliers
        for _, row in group.iterrows():
            z_score = (row['total_minutes'] - mean_duration) / std_duration
            
            # Calculate percentage of 30-minute slot
            percentage_of_slot = (row['total_minutes'] / 30) * 100
            
            # Flag periods that are statistically significant OR use high percentage of time
            if abs(z_score) >= z_threshold or percentage_of_slot >= percentage_threshold:
                significant_activities.append({
                    'activity': activity_type.replace('_', ' '),
                    'id': emp_id,
                    'shift': shift,
                    'time_category': row['time_category'],
                    'region': row['region'],
                    'duration': row['total_minutes'],
                    'z_score': z_score,
                    'percentage_of_slot': percentage_of_slot,
                    'mean_duration': mean_duration,
                    'std_duration': std_duration,
                    'direction': 'above average' if z_score > 0 else 'below average',
                    'flagged_by': 'both metrics' if abs(z_score) >= z_threshold and percentage_of_slot >= percentage_threshold else 
                               ('statistical significance' if abs(z_score) >= z_threshold else 'high percentage')
                })
    
    return significant_activities

def visualize_employee_shift(activity_data, activity_type, emp_id, shift, output_dir):
    """Generate visualization for a specific employee-shift combination"""
    # Filter data for this employee and shift
    emp_shift_data = activity_data[(activity_data['id'] == emp_id) & (activity_data['shift'] == shift)]
    
    if emp_shift_data.empty:
        return False
        
    # Group by time category to handle multiple regions in same time slot
    time_summary = emp_shift_data.groupby('time_category')['total_minutes'].sum().reset_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate statistics
    mean_duration = time_summary['total_minutes'].mean()
    std_duration = time_summary['total_minutes'].std()
    
    # Sort by time category for proper x-axis ordering
    time_summary['hour'] = time_summary['time_category'].apply(lambda x: int(x.split(':')[0]))
    time_summary['minute'] = time_summary['time_category'].apply(lambda x: int(x.split(':')[1]))
    time_summary['sort_key'] = time_summary['hour'] * 60 + time_summary['minute']
    sorted_data = time_summary.sort_values('sort_key')
    
    # Add percentage of 30-minute slot
    sorted_data['percentage'] = (sorted_data['total_minutes'] / 30) * 100
    
    # Create bar chart
    bars = ax.bar(
        sorted_data['time_category'], 
        sorted_data['total_minutes'],
        color=ACTIVITY_COLORS.get(activity_type, '#1f77b4')
    )
    
    # Add mean line
    ax.axhline(mean_duration, color='red', linestyle='--', alpha=0.7, 
              label=f'Mean: {mean_duration:.2f}m')
    
    # Add standard deviation band
    if not pd.isna(std_duration) and std_duration > 0:
        ax.axhspan(mean_duration - std_duration, mean_duration + std_duration, 
                  alpha=0.2, color='gray', label=f'±1 StdDev: {std_duration:.2f}m')
    
    # Add 70% threshold line
    ax.axhline(21, color='green', linestyle=':', alpha=0.7, 
              label='70% of slot (21m)')
    
    # Highlight significant slots
    for i, row in enumerate(sorted_data.itertuples()):
        z_score = 0
        if not pd.isna(std_duration) and std_duration > 0:
            z_score = (row.total_minutes - mean_duration) / std_duration
        
        # Flag if statistically significant OR high percentage
        if abs(z_score) >= 2.0 or row.percentage >= 70:
            # Set bar color based on which threshold triggered
            if abs(z_score) >= 2.0 and row.percentage >= 70:
                # Both thresholds triggered
                bars[i].set_color('purple')
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(1.5)
            elif abs(z_score) >= 2.0:
                # Only statistical threshold
                bars[i].set_color('red' if z_score > 0 else 'blue')
            else:
                # Only percentage threshold
                bars[i].set_color('green')
            
            # Add annotation with both metrics
            ax.text(i, row.total_minutes + 0.5, 
                   f"z={z_score:.1f}\n{row.percentage:.0f}%", 
                   ha='center', va='bottom', fontweight='bold',
                   color='black')
    
    # Add region annotations
    for i, time_cat in enumerate(sorted_data['time_category']):
        # Get all regions for this time category
        regions = emp_shift_data[emp_shift_data['time_category'] == time_cat]
        
        if len(regions) == 1:
            # Single region - simple annotation
            region = regions.iloc[0]['region']
            ax.text(i, sorted_data.iloc[i]['total_minutes']/2, region,
                   ha='center', va='center', rotation=90,
                   fontsize=8, color='white', fontweight='bold')
        else:
            # Multiple regions - create region breakdown annotation
            region_text = []
            for _, reg_row in regions.iterrows():
                percentage = (reg_row['total_minutes'] / regions['total_minutes'].sum() * 100)
                region_text.append(f"{reg_row['region']}: {percentage:.0f}%")
            
            # Add text at top of bar
            total_minutes = sorted_data[sorted_data['time_category'] == time_cat]['total_minutes'].values[0]
            ax.annotate(
                "\n".join(region_text),
                xy=(i, total_minutes),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="gray")
            )
    
    # Format labels
    activity_label = activity_type.replace('_', ' ').title()
    ax.set_title(f"{activity_label} - Employee {emp_id} - Shift {shift}")
    ax.set_xlabel("Time (30-minute intervals)")
    ax.set_ylabel("Duration (minutes)")
    ax.set_ylim(0, max(30, sorted_data['total_minutes'].max() * 1.1))
    
    # Add legend
    ax.legend()
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{activity_type}_{emp_id}_shift_{shift}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return True

def visualize_top_recommendations(recommendations, output_dir, top_n=20):
    """Create visualization of top video review recommendations"""
    if recommendations.empty:
        return
        
    # Get top N recommendations
    top_recs = recommendations.head(top_n)
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Create labels
    labels = [f"Emp {row.id} - Shift {row.shift} - {row.time_category} - {row.region}" 
             for row in top_recs.itertuples()]
    
    # Create bars
    activity_colors = {act.replace('_', ' '): ACTIVITY_COLORS[act] for act in ACTIVITY_TYPES}
    
    # Color based on flagging method
    colors = []
    for row in top_recs.itertuples():
        base_color = activity_colors.get(row.activity, '#1f77b4')
        if row.flagged_by == 'both metrics':
            # Make color darker for both metrics
            colors.append(base_color)
        elif row.flagged_by == 'statistical significance':
            # Add transparency for statistical only
            colors.append(base_color + '99')  # Add alpha
        else:  # high percentage
            # Stipple pattern for percentage only
            colors.append(base_color + 'CC')  # Different alpha
    
    # Create horizontal bar chart
    bars = plt.barh(labels, top_recs['abs_z_score'], color=colors)
    
    # Add secondary bars for percentage (as small markers)
    for i, row in enumerate(top_recs.itertuples()):
        plt.plot(
            [0], [i],
            marker='|', markersize=row.percentage_of_slot/3,
            color='black', markeredgewidth=2
        )
    
    # Add activity, z-score and percentage labels
    for i, row in enumerate(top_recs.itertuples()):
        plt.text(
            row.abs_z_score + 0.1,
            i,
            f"{row.activity.title()} ({row.percentage_of_slot:.0f}% of slot) - z={row.z_score:.2f} - {row.flagged_by}",
            va='center',
            fontsize=10
        )
    
    # Set labels
    plt.title("Top Video Review Recommendations", fontsize=16)
    plt.xlabel("Statistical Significance (|z-score|)", fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend for activity types
    activity_handles = [plt.Rectangle((0,0), 1, 1, color=ACTIVITY_COLORS[act]) 
                       for act in ACTIVITY_TYPES]
    activity_labels = [act.replace('_', ' ').title() for act in ACTIVITY_TYPES]
    
    # Add legend for flagging methods
    flag_handles = [
        plt.Rectangle((0,0), 1, 1, color='gray'),
        plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.6),
        plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.8)
    ]
    flag_labels = ['Both metrics', 'Statistical only', 'Percentage only']
    
    # Combine legends
    plt.legend(activity_handles + flag_handles, 
              activity_labels + flag_labels, 
              loc='lower right', ncol=2)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'top_recommendations.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

def create_html_report(recommendations, output_dir):
    """Create an interactive HTML report for video review recommendations"""
    if recommendations.empty:
        return
        
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Bakery Activity Video Review Recommendations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .recommendations {{ margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .high {{ color: #C00000; font-weight: bold; }}
        .low {{ color: #156082; font-weight: bold; }}
        .handle-center {{ background-color: #F1A98333; }}
        .handle-up {{ background-color: #FFC00033; }}
        .handle-down {{ background-color: #C0000033; }}
        .stand {{ background-color: #A6A6A633; }}
        .walk {{ background-color: #15608233; }}
        .both-metrics {{ font-weight: bold; }}
        .percentage-bar {{ 
            height: 18px; 
            background-color: #4CAF50; 
            text-align: center;
            color: white;
            font-size: 12px;
        }}
        .metric-cell {{ 
            width: 150px; 
        }}
    </style>
</head>
<body>
    <h1>Bakery Activity Analysis - Video Review Recommendations</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <h2>Top Recommendations</h2>
    <div class="recommendations">
        <table>
            <tr>
                <th>Rank</th>
                <th>Employee</th>
                <th>Shift</th>
                <th>Time</th>
                <th>Activity</th>
                <th>Region</th>
                <th>Duration</th>
                <th>% of Slot</th>
                <th>Z-Score</th>
                <th>Flagged By</th>
            </tr>
    """
    
    # Add rows for each recommendation
    for i, row in enumerate(recommendations.head(50).itertuples()):
        # Determine CSS classes
        direction_class = "high" if row.direction == "above average" else "low"
        activity_class = row.activity.lower().replace(" ", "-")
        flagged_class = "both-metrics" if row.flagged_by == "both metrics" else ""
        
        # Format duration
        duration_text = f"{row.duration:.1f} min"
        
        # Create percentage bar
        pct_width = min(100, row.percentage_of_slot)  # Cap at 100%
        percentage_bar = f"""<div class="percentage-bar" style="width: {pct_width}%">{row.percentage_of_slot:.0f}%</div>"""
        
        html_content += f"""
            <tr class="{activity_class}">
                <td>{i+1}</td>
                <td>{row.id}</td>
                <td>{row.shift}</td>
                <td>{row.time_category}</td>
                <td>{row.activity.title()}</td>
                <td>{row.region}</td>
                <td>{duration_text}</td>
                <td class="metric-cell">{percentage_bar}</td>
                <td class="{direction_class}">{row.z_score:.2f}</td>
                <td class="{flagged_class}">{row.flagged_by}</td>
            </tr>"""
    
    # Complete the HTML
    html_content += """
        </table>
    </div>
    
    <h2>How to Use This Report</h2>
    <p>This report identifies significant patterns in employee activities using two metrics:</p>
    <ol>
        <li><strong>Percentage of time slot:</strong> How much of the 30-minute period was spent on this activity (flagged if ≥70%)</li>
        <li><strong>Statistical significance:</strong> How unusual this duration is compared to employee's pattern in this shift (z-score)</li>
    </ol>
    <p>High priority items are those flagged by both metrics. For each entry:</p>
    <ol>
        <li>Use the shift number to identify the video recording date</li>
        <li>Navigate to the specific time slot indicated</li>
        <li>Focus on the identified activity and region</li>
    </ol>
</body>
</html>
    """
    
    # Save the HTML file
    html_path = os.path.join(output_dir, 'video_review_recommendations.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved interactive HTML report to {html_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate bakery activity insights')
    parser.add_argument('--input-dir', type=str, default='output',
                      help='Directory containing pre-processed CSV files')
    parser.add_argument('--output-dir', type=str, default='output/insights',
                      help='Directory to save analysis results')
    parser.add_argument('--z-threshold', type=float, default=2.0,
                      help='Z-score threshold for significance (default: 2.0)')
    parser.add_argument('--percentage-threshold', type=float, default=70,
                      help='Percentage of 30-min slot threshold (default: 70)')
    parser.add_argument('--employee', type=str, help='Analyze only a specific employee')
    parser.add_argument('--shift', type=int, help='Analyze only a specific shift')
    parser.add_argument('--top-n', type=int, default=20,
                      help='Number of top recommendations to visualize')
    parser.add_argument('--skip-charts', action='store_true',
                      help='Skip generating individual charts (faster)')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = datetime.now()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    charts_dir = os.path.join(args.output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Process all activity types
    all_significant_activities = []
    processed_combinations = 0
    
    for activity_type in ACTIVITY_TYPES:
        print(f"\nProcessing {activity_type}...")
        
        # Load data
        activity_data = load_activity_data(args.input_dir, activity_type)
        if activity_data is None:
            continue
            
        # Filter by employee if specified
        if args.employee:
            activity_data = activity_data[activity_data['id'] == args.employee]
            if activity_data.empty:
                print(f"No data for employee {args.employee}")
                continue
                
        # Filter by shift if specified
        if args.shift:
            activity_data = activity_data[activity_data['shift'] == args.shift]
            if activity_data.empty:
                print(f"No data for shift {args.shift}")
                continue
        
        # Find significant activities
        significant = identify_significant_activities(
            activity_data, 
            activity_type, 
            args.z_threshold,
            args.percentage_threshold
        )
        all_significant_activities.extend(significant)
        
        if significant:
            print(f"Found {len(significant)} significant periods for {activity_type}")
            
            # Skip chart generation if requested
            if args.skip_charts:
                continue
                
            # Create activity-specific directory
            activity_dir = os.path.join(charts_dir, activity_type)
            os.makedirs(activity_dir, exist_ok=True)
            
            # Generate visualizations for each employee-shift combination
            for emp_id in activity_data['id'].unique():
                for shift in activity_data[activity_data['id'] == emp_id]['shift'].unique():
                    processed_combinations += 1
                    visualize_employee_shift(activity_data, activity_type, emp_id, shift, activity_dir)
    
    # Convert findings to DataFrame and sort by significance
    if all_significant_activities:
        recommendations = pd.DataFrame(all_significant_activities)
        
        # Add absolute z-score column
        recommendations['abs_z_score'] = recommendations['z_score'].abs()
        
        # Create a composite score - weight both metrics equally
        recommendations['composite_score'] = (
            0.5 * recommendations['abs_z_score'] + 
            0.5 * (recommendations['percentage_of_slot'] / 100 * 6)  # Scale to similar range as z-scores
        )
        
        # Sort by composite score
        recommendations = recommendations.sort_values('composite_score', ascending=False)
        
        # Save recommendations to CSV
        csv_path = os.path.join(args.output_dir, 'video_review_recommendations.csv')
        recommendations.to_csv(csv_path, index=False)
        
        # Generate visualization
        visualize_top_recommendations(recommendations, args.output_dir, args.top_n)
        
        # Create HTML report
        create_html_report(recommendations, args.output_dir)
        
        # Print top recommendations for quick review
        print("\nTop Video Review Recommendations:")
        for i, row in enumerate(recommendations.head(10).itertuples()):
            print(f"{i+1}. {row.activity.title()} - Employee {row.id}, Shift {row.shift}")
            print(f"   Time: {row.time_category}, Region: {row.region}")
            print(f"   Duration: {row.duration:.2f} minutes ({row.percentage_of_slot:.0f}% of slot)")
            print(f"   Significance: z-score: {row.z_score:.2f}, Flagged by: {row.flagged_by}")
            print()
        
        print(f"\nSaved {len(recommendations)} recommendations to {csv_path}")
    else:
        print("\nNo significant patterns found for video review")
        
    # Report execution statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nAnalysis completed in {duration:.2f} seconds")
    print(f"Processed {processed_combinations} employee-shift-activity combinations")
    print(f"Found {len(all_significant_activities)} significant activities for video review")

if __name__ == "__main__":
    main()