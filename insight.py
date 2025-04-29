#!/usr/bin/env python3
"""
Bakery Activity Insight Generator

Analyzes activity data across time slots to identify statistically significant 
patterns within specific employee shifts that warrant video review.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
from collections import defaultdict
import json

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Define activity types and their colors for consistent visualization
ACTIVITY_TYPES = ['Walk', 'Stand', 'Handle center', 'Handle down', 'Handle up']
ACTIVITY_COLORS = {
    'Walk': '#156082',       # Blue
    'Stand': '#A6A6A6',      # Grey
    'Handle center': '#F1A983',  # Light orange
    'Handle down': '#C00000',  # Red
    'Handle up': '#FFC000'   # Yellow
}

# Define region colors for consistent visualization
REGION_COLORS = {
    'Fettbacken': '#6BAED6',
    'Konditorei_station': '#3182BD',
    'konditorei_deco': '#08519C',
    'Brotstation': '#31A354',
    '1_Brotstation': '#31A354',
    '2_Brotstation': '#74C476',
    '3_Brotstation': '#A1D99B',
    '4_Brotstation': '#C7E9C0',
    'Mehlmaschine': '#74C476',
    'brotkisten_regal': '#FFFF00',
    'Gärraum': '#FF7F0E',
    'bereitstellen_prepared_goods': '#FF7F0E',
    'leere_Kisten': '#9467BD',
}

def load_and_preprocess_data(file_path):
    """
    Load raw sensor data and preprocess it for analysis
    
    Parameters:
    -----------
    file_path : str
        Path to the processed_sensor_data.csv file
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data for analysis
    """
    print(f"Loading data from {file_path}...")
    
    # Load raw data
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} records")
    
    # Convert startTime to datetime
    data['start_seconds'] = data['startTime'] % 86400  # Seconds since midnight
    
    # Create 30-minute time slots
    data['time_slot'] = (data['start_seconds'] // 1800 * 1800).astype(int)
    
    # Format time slot as HH:MM
    data['time_category'] = data['time_slot'].apply(
        lambda x: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}"
    )
    
    # Ensure shift is integer
    if 'shift' in data.columns:
        data['shift'] = data['shift'].astype(int)
    
    return data

def extract_activity_time_slots(data, activity_type, interval_seconds=1800):
    """
    Extract time slot data for a specific activity type
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed sensor data
    activity_type : str
        Activity type to analyze
    interval_seconds : int
        Size of time slots in seconds (default: 1800 = 30 minutes)
    
    Returns:
    --------
    pandas.DataFrame
        Activity data by time slot
    """
    # Filter data for the specific activity
    activity_data = data[data['activity'] == activity_type].copy()
    
    if activity_data.empty:
        print(f"No data found for activity: {activity_type}")
        return None
    
    # Create a dictionary to store the accumulated durations
    # Key: (employee_id, shift, time_category, region), Value: total_duration
    accumulated_durations = defaultdict(float)
    
    # Process the data
    for _, row in activity_data.iterrows():
        # Calculate start and end time slots (ensure they're integers)
        start_slot = int((row['startTime'] // interval_seconds) * interval_seconds)
        end_slot = int((row['endTime'] // interval_seconds) * interval_seconds)
        
        if start_slot == end_slot:
            # Activity doesn't cross time slot boundaries
            accumulated_durations[(row['id'], int(row['shift']), row['time_category'], row['region'])] += row['duration']
        else:
            # Handle activities that span multiple time slots
            current_slot = start_slot
            current_time = row['startTime']
            
            while current_slot < end_slot:
                next_slot_start = current_slot + interval_seconds
                duration_in_slot = min(next_slot_start, row['endTime']) - current_time
                
                # Format time using integer division
                hour = int(current_slot // 3600)
                minute = int((current_slot % 3600) // 60)
                slot_time_category = f"{hour:02d}:{minute:02d}"
                
                accumulated_durations[(row['id'], int(row['shift']), slot_time_category, row['region'])] += duration_in_slot
                
                current_time = next_slot_start
                current_slot = next_slot_start
            
            # Handle the final time slot
            if current_time < row['endTime']:
                # Format time using integer division
                hour = int(current_slot // 3600)
                minute = int((current_slot % 3600) // 60)
                slot_time_category = f"{hour:02d}:{minute:02d}"
                
                duration_in_slot = row['endTime'] - current_time
                accumulated_durations[(row['id'], int(row['shift']), slot_time_category, row['region'])] += duration_in_slot
    
    # Convert accumulated durations to DataFrame
    result_data = []
    for (emp_id, shift, time_category, region), duration in accumulated_durations.items():
        result_data.append({
            'id': emp_id,
            'shift': shift,
            'time_category': time_category,
            'region': region,
            'duration': duration,
            'total_minutes': round(duration / 60, 2)
        })
    
    return pd.DataFrame(result_data)

def find_significant_time_slots(activity_data, z_threshold=2.0):
    """
    Find statistically significant time slots within each employee-shift combination
    
    Parameters:
    -----------
    activity_data : pandas.DataFrame
        Activity data by time slot
    z_threshold : float
        Z-score threshold for considering a time slot significant
    
    Returns:
    --------
    pandas.DataFrame
        Significant time slots with z-scores
    """
    if activity_data is None or activity_data.empty:
        return pd.DataFrame()
    
    # Group by employee and shift
    employee_shifts = activity_data.groupby(['id', 'shift'])
    
    # Store significant time slots
    significant_slots = []
    
    for (emp_id, shift), group in employee_shifts:
        # Calculate mean and std for this employee-shift
        mean_duration = group['total_minutes'].mean()
        std_duration = group['total_minutes'].std()
        
        if pd.isna(std_duration) or std_duration == 0:
            # Skip if standard deviation is zero (all values are identical)
            continue
        
        # Calculate z-scores
        for _, row in group.iterrows():
            z_score = (row['total_minutes'] - mean_duration) / std_duration
            
            if abs(z_score) > z_threshold:
                significant_slots.append({
                    'id': emp_id,
                    'shift': shift,
                    'time_category': row['time_category'],
                    'region': row['region'],
                    'total_minutes': row['total_minutes'],
                    'z_score': z_score,
                    'shift_mean': mean_duration,
                    'shift_std': std_duration
                })
    
    # Convert to DataFrame and sort by absolute z-score
    if significant_slots:
        result = pd.DataFrame(significant_slots)
        result['abs_z_score'] = result['z_score'].abs()
        return result.sort_values('abs_z_score', ascending=False)
    else:
        return pd.DataFrame()

def plot_time_slot_distribution(activity_data, activity_type, emp_id, shift, output_dir):
    """
    Create a visualization showing time slot distribution for a specific employee and shift
    
    Parameters:
    -----------
    activity_data : pandas.DataFrame
        Activity data by time slot
    activity_type : str
        Activity type being analyzed
    emp_id : str
        Employee ID
    shift : int
        Shift number
    output_dir : str
        Directory to save visualizations
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Filter data for this employee and shift
    emp_shift_data = activity_data[(activity_data['id'] == emp_id) & (activity_data['shift'] == shift)]
    
    if emp_shift_data.empty:
        print(f"No data for employee {emp_id} shift {shift} in {activity_type}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Extract hour from time category for sorting
    emp_shift_data['hour'] = emp_shift_data['time_category'].apply(lambda x: int(x.split(':')[0]))
    emp_shift_data['minute'] = emp_shift_data['time_category'].apply(lambda x: int(x.split(':')[1]))
    emp_shift_data['sort_key'] = emp_shift_data['hour'] * 60 + emp_shift_data['minute']
    
    # Sort by time category
    sorted_data = emp_shift_data.sort_values('sort_key')
    
    # Group by time category (in case multiple regions in same time slot)
    time_summary = sorted_data.groupby('time_category')['total_minutes'].sum().reset_index()
    
    # Create bar chart
    bars = ax.bar(
        time_summary['time_category'],
        time_summary['total_minutes'],
        color=ACTIVITY_COLORS.get(activity_type, '#1f77b4')
    )
    
    # Add region labels to the bars
    for time_cat in time_summary['time_category']:
        regions_in_slot = sorted_data[sorted_data['time_category'] == time_cat]
        
        # If multiple regions in this time slot, annotate
        if len(regions_in_slot) > 1:
            # Calculate total minutes for this time slot
            total_mins = regions_in_slot['total_minutes'].sum()
            
            # Create annotation with region breakdown
            region_text = "\n".join([
                f"{row['region']}: {row['total_minutes']:.1f}m ({row['total_minutes']/total_mins*100:.0f}%)"
                for _, row in regions_in_slot.iterrows()
            ])
            
            # Find index of this time category in the plot
            idx = list(time_summary['time_category']).index(time_cat)
            
            # Add annotation
            ax.annotate(
                region_text,
                xy=(idx, total_mins),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )
    
    # Calculate employee-shift statistics
    mean_duration = sorted_data['total_minutes'].mean()
    median_duration = sorted_data['total_minutes'].median()
    std_duration = sorted_data['total_minutes'].std()
    
    # Add mean line
    ax.axhline(
        mean_duration,
        color='red',
        linestyle='--',
        alpha=0.7,
        label=f'Mean: {mean_duration:.2f}m'
    )
    
    # Add median line
    ax.axhline(
        median_duration,
        color='green',
        linestyle=':',
        alpha=0.7,
        label=f'Median: {median_duration:.2f}m'
    )
    
    # Add standard deviation band
    if not pd.isna(std_duration) and std_duration > 0:
        ax.axhspan(
            mean_duration - std_duration,
            mean_duration + std_duration,
            alpha=0.2,
            color='gray',
            label=f'±1 Std Dev: {std_duration:.2f}m'
        )
    
    # Highlight significant time slots (if any)
    for idx, bar in enumerate(bars):
        time_cat = time_summary.iloc[idx]['time_category']
        duration = time_summary.iloc[idx]['total_minutes']
        
        # Calculate z-score
        if not pd.isna(std_duration) and std_duration > 0:
            z_score = (duration - mean_duration) / std_duration
            
            # Highlight significant bars
            if abs(z_score) > 2.0:
                bar.set_color('red' if z_score > 0 else 'blue')
                bar.set_alpha(0.8)
                
                # Add z-score label
                ax.text(
                    idx,
                    duration + 0.5,
                    f"z={z_score:.1f}",
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    color='red' if z_score > 0 else 'blue'
                )
    
    # Set title and labels
    ax.set_title(f"{activity_type} - Employee {emp_id} - Shift {shift}")
    ax.set_xlabel("Time of Day (30-minute slots)")
    ax.set_ylabel("Duration (minutes)")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f"{activity_type.lower().replace(' ', '_')}_{emp_id}_shift_{shift}.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Saved figure to {fig_path}")
    
    return fig

def create_heatmap_for_employee_activity(activity_data, activity_type, emp_id, output_dir):
    """
    Create a heatmap showing activity patterns across all shifts for an employee
    
    Parameters:
    -----------
    activity_data : pandas.DataFrame
        Activity data by time slot
    activity_type : str
        Activity type being analyzed
    emp_id : str
        Employee ID
    output_dir : str
        Directory to save visualizations
    """
    # Filter data for this employee
    emp_data = activity_data[activity_data['id'] == emp_id]
    
    if emp_data.empty:
        print(f"No data for employee {emp_id} in {activity_type}")
        return None
    
    # Extract hour from time category for consistency
    emp_data['hour'] = emp_data['time_category'].apply(lambda x: int(x.split(':')[0]))
    emp_data['minute'] = emp_data['time_category'].apply(lambda x: int(x.split(':')[1]))
    
    # Create a pivot table: shifts as rows, hours as columns
    # Use half-hour resolution
    pivot_data = pd.pivot_table(
        emp_data,
        values='total_minutes',
        index='shift',
        columns=['hour', 'minute'],
        aggfunc='sum',
        fill_value=0
    )
    
    # Flatten column MultiIndex to create time labels
    time_labels = [f"{h:02d}:{m:02d}" for h, m in pivot_data.columns]
    pivot_data.columns = time_labels
    
    # Create figure for heatmap
    plt.figure(figsize=(16, len(pivot_data) + 2))
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        cmap='YlOrRd',
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        cbar_kws={'label': 'Duration (minutes)'}
    )
    
    plt.title(f"{activity_type} Pattern - Employee {emp_id} Across Shifts")
    plt.xlabel("Time of Day (30-minute slots)")
    plt.ylabel("Shift")
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f"{activity_type.lower().replace(' ', '_')}_{emp_id}_heatmap.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Saved heatmap to {fig_path}")
    
    plt.close()

def generate_recommendations(significant_slots_by_activity, output_dir):
    """
    Generate video review recommendations based on significant time slots
    
    Parameters:
    -----------
    significant_slots_by_activity : dict
        Dictionary of significant time slots by activity type
    output_dir : str
        Directory to save recommendations
    
    Returns:
    --------
    pandas.DataFrame
        Recommendations sorted by significance
    """
    all_recommendations = []
    
    for activity_type, slots in significant_slots_by_activity.items():
        if slots.empty:
            continue
        
        for _, row in slots.iterrows():
            all_recommendations.append({
                'activity': activity_type,
                'employee': row['id'],
                'shift': row['shift'],
                'time_category': row['time_category'],
                'region': row['region'],
                'duration_minutes': row['total_minutes'],
                'z_score': row['z_score'],
                'abs_z_score': abs(row['z_score']),
                'shift_mean': row['shift_mean'],
                'shift_std': row['shift_std'],
                'direction': 'above average' if row['z_score'] > 0 else 'below average',
                'reason': (f"Unusually {'high' if row['z_score'] > 0 else 'low'} duration "
                         f"of {activity_type} (z-score: {row['z_score']:.2f})")
            })
    
    if not all_recommendations:
        return pd.DataFrame()
    
    # Convert to DataFrame and sort by absolute z-score
    recommendations = pd.DataFrame(all_recommendations)
    recommendations = recommendations.sort_values('abs_z_score', ascending=False)
    
    # Save recommendations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'video_review_recommendations.csv')
        recommendations.to_csv(csv_path, index=False)
        print(f"Saved recommendations to {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(output_dir, 'video_review_recommendations.json')
        recommendations.to_json(json_path, orient='records', indent=2)
        print(f"Saved recommendations to {json_path}")
    
    return recommendations

def visualize_top_recommendations(recommendations, output_dir, top_n=20):
    """
    Create a visualization of the top recommendations
    
    Parameters:
    -----------
    recommendations : pandas.DataFrame
        Recommendations data
    output_dir : str
        Directory to save visualization
    top_n : int
        Number of top recommendations to visualize
    """
    if recommendations.empty:
        print("No recommendations to visualize")
        return
    
    # Get top N recommendations
    top_recs = recommendations.head(top_n)
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Create labels
    labels = [f"Employee {row.employee} - Shift {row.shift} - {row.time_category} - {row.region}" 
             for _, row in top_recs.iterrows()]
    
    # Create bars colored by activity
    bars = plt.barh(
        labels,
        top_recs['abs_z_score'],
        color=[ACTIVITY_COLORS.get(activity, '#1f77b4') for activity in top_recs['activity']]
    )
    
    # Add activity and z-score labels
    for i, (_, row) in enumerate(top_recs.iterrows()):
        plt.text(
            row['abs_z_score'] + 0.1,
            i,
            f"{row['activity']} ({row['direction']}) - z={row['z_score']:.2f}",
            va='center',
            fontsize=10
        )
    
    # Set title and labels
    plt.title("Top Video Review Recommendations", fontsize=16)
    plt.xlabel("Absolute Z-Score (Statistical Significance)", fontsize=14)
    plt.ylabel("Employee - Shift - Time - Region", fontsize=14)
    
    # Add a legend for activities
    activity_handles = [plt.Rectangle((0,0), 1, 1, color=ACTIVITY_COLORS.get(act, '#1f77b4')) 
                       for act in ACTIVITY_TYPES if act in top_recs['activity'].values]
    activity_labels = [act for act in ACTIVITY_TYPES if act in top_recs['activity'].values]
    plt.legend(activity_handles, activity_labels, loc='lower right')
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        fig_path = os.path.join(output_dir, 'top_recommendations.png')
        plt.savefig(fig_path, dpi=300)
        print(f"Saved visualization to {fig_path}")
    
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate insights from activity data')
    parser.add_argument('--data-file', type=str, default='data/raw/processed_sensor_data.csv',
                      help='Path to the processed sensor data CSV file')
    parser.add_argument('--output-dir', type=str, default='output/insights',
                      help='Directory to save analysis results')
    parser.add_argument('--z-threshold', type=float, default=2.0,
                      help='Z-score threshold for outliers (default: 2.0)')
    parser.add_argument('--top-n', type=int, default=20,
                      help='Number of top results to include (default: 20)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    charts_dir = os.path.join(args.output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Load and preprocess data
    data = load_and_preprocess_data(args.data_file)
    
    # Process each activity type
    significant_slots_by_activity = {}
    
    for activity_type in ACTIVITY_TYPES:
        print(f"\nProcessing {activity_type}...")
        
        # Extract time slot data for this activity
        activity_data = extract_activity_time_slots(data, activity_type)
        
        if activity_data is None or activity_data.empty:
            print(f"No data for {activity_type}")
            continue
        
        # Find statistically significant time slots
        significant_slots = find_significant_time_slots(activity_data, args.z_threshold)
        significant_slots_by_activity[activity_type] = significant_slots
        
        if not significant_slots.empty:
            print(f"Found {len(significant_slots)} significant time slots for {activity_type}")
        else:
            print(f"No significant time slots found for {activity_type}")
        
        # Generate visualizations for each employee-shift combination
        employee_shifts = activity_data.groupby(['id', 'shift']).size().reset_index().rename(columns={0: 'count'})
        
        for _, row in employee_shifts.iterrows():
            emp_id = row['id']
            shift = row['shift']
            
            # Create distribution plot
            plot_time_slot_distribution(activity_data, activity_type, emp_id, shift, charts_dir)
        
        # Create heatmaps for each employee
        for emp_id in activity_data['id'].unique():
            create_heatmap_for_employee_activity(activity_data, activity_type, emp_id, charts_dir)
    
    # Generate recommendations
    recommendations = generate_recommendations(significant_slots_by_activity, args.output_dir)
    
    if not recommendations.empty:
        print(f"\nGenerated {len(recommendations)} video review recommendations")
        
        # Visualize top recommendations
        visualize_top_recommendations(recommendations, args.output_dir, args.top_n)
        
        # Print top recommendations for quick review
        print("\nTop Video Review Recommendations:")
        for i, (_, row) in enumerate(recommendations.head(10).iterrows()):
            print(f"{i+1}. {row['reason']}")
            print(f"   Employee: {row['employee']}, Shift: {row['shift']}, Time: {row['time_category']}, Region: {row['region']}")
            print(f"   Duration: {row['duration_minutes']:.2f} minutes (z-score: {row['z_score']:.2f})")
            print()
    else:
        print("No significant patterns found for video review")
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()