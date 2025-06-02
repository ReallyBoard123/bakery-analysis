import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Define paths
DATA_PATH = 'data/raw/processed_sensor_data.csv'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define constants
INTERVAL_SECONDS = 1800  # 30 minutes in seconds
SECONDS_PER_DAY = 86400
ACTIVITY_TYPES = ['Walk', 'Stand', 'Handle center', 'Handle up', 'Handle down']
TOP_N_REGIONS = 3  # Keep top 3 regions per time category

# Function to create time category string from seconds
def seconds_to_time_category(seconds):
    seconds = int(seconds)
    hours = (seconds // 3600) % 24
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"

# Function to format seconds as mm:ss
def format_duration(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

# Load data
print(f"Loading data from {DATA_PATH}...")
data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(data)} rows")

# Process each activity type
for activity in ACTIVITY_TYPES:
    print(f"Processing activity: {activity}")
    
    # Filter data for current activity
    activity_data = data[data['activity'] == activity].copy()
    
    if activity_data.empty:
        print(f"No data found for activity: {activity}")
        continue
    
    # Include shift numbers in the aggregated data
    # Modify the key in the accumulated_durations dictionary to include shift
    accumulated_durations = defaultdict(float)

    # Process the data in a vectorized way
    for _, row in activity_data.iterrows():
        # Get normalized times
        start_time = row['startTime'] % SECONDS_PER_DAY
        end_time = row['endTime'] % SECONDS_PER_DAY

        # Handle overnight activities
        if end_time < start_time:
            end_time += SECONDS_PER_DAY

        # Calculate the intervals this activity spans
        start_interval = (start_time // INTERVAL_SECONDS) * INTERVAL_SECONDS
        end_interval = (end_time // INTERVAL_SECONDS) * INTERVAL_SECONDS

        # Calculate all interval thresholds this activity crosses
        interval_thresholds = np.arange(
            start_interval + INTERVAL_SECONDS, 
            end_interval + INTERVAL_SECONDS, 
            INTERVAL_SECONDS
        )

        # Initialize current point
        current = start_time

        # Add segment for first interval
        if len(interval_thresholds) == 0:
            # Activity doesn't cross interval boundaries
            accumulated_durations[(row['id'], row['shift'], seconds_to_time_category(start_interval), row['region'])] += row['duration']
        else:
            # Process first interval
            first_threshold = interval_thresholds[0]
            first_duration = first_threshold - current
            accumulated_durations[(row['id'], row['shift'], seconds_to_time_category(start_interval), row['region'])] += first_duration
            current = first_threshold

            # Process middle intervals
            for i in range(len(interval_thresholds) - 1):
                interval_start = interval_thresholds[i]
                interval_end = interval_thresholds[i + 1]
                segment_duration = interval_end - interval_start
                interval_category = seconds_to_time_category(interval_start)
                accumulated_durations[(row['id'], row['shift'], interval_category, row['region'])] += segment_duration

            # Process last interval if activity doesn't end exactly on a threshold
            if current < end_time:
                last_interval = (current // INTERVAL_SECONDS) * INTERVAL_SECONDS
                last_duration = end_time - current
                accumulated_durations[(row['id'], row['shift'], seconds_to_time_category(last_interval), row['region'])] += last_duration

    # Convert accumulated durations to DataFrame
    result_data = []
    for (emp_id, shift, time_category, region), total_duration in accumulated_durations.items():
        result_data.append({
            'id': emp_id,
            'shift': shift,
            'time_category': time_category,
            'region': region,
            'duration': total_duration,
            'total_minutes': round(total_duration / 60, 2),
            'formatted_duration': format_duration(total_duration)
        })
    
    if not result_data:
        print(f"No results for {activity}")
        continue
        
    # Create DataFrame
    result_df = pd.DataFrame(result_data)
    
    # For each employee and time category, keep the top N regions
    top_regions = result_df.sort_values(['id', 'shift', 'time_category', 'total_minutes'], ascending=[True, True, True, False])
    top_regions = top_regions.groupby(['id', 'shift', 'time_category']).head(TOP_N_REGIONS).reset_index(drop=True)
    
    # Format the final DataFrame
    top_regions = top_regions[['id', 'shift', 'region', 'time_category', 'formatted_duration', 'total_minutes']]
    top_regions = top_regions.rename(columns={'formatted_duration': 'duration'})
    
    # Save to CSV
    output_file = f"{OUTPUT_DIR}/{activity.lower().replace(' ', '_')}_region_stats.csv"
    top_regions.to_csv(output_file, index=False)
    print(f"Saved {activity} data to {output_file} with {len(top_regions)} rows")
    
    # Print row counts per employee
    employee_counts = top_regions.groupby('id').size()
    print("Rows per employee:")
    for emp_id, count in employee_counts.items():
        print(f"  Employee {emp_id}: {count} rows")

# Function to analyze activity durations by time ranges
def analyze_activity_durations():
    print('\nAnalyzing activity durations by time ranges...')
    
    # Define duration ranges (in seconds)
    ranges = [
        (0, 5),
        (5, 10),
        (10, 20),
        (20, 30),
        (30, 60),
        (60, 120),
        (120, 300),
        (300, float('inf'))
    ]

    # Define range labels
    range_labels = [
        '0-5 sec',
        '5-10 sec',
        '10-20 sec',
        '20-30 sec',
        '30-60 sec',
        '1-2 min',
        '2-5 min',
        '5+ min'
    ]
    
    # Ensure analysis directory exists
    analysis_dir = f'{OUTPUT_DIR}/analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load the raw data again to analyze individual instances
    print(f'Loading data from {DATA_PATH}...')
    data = pd.read_csv(DATA_PATH)
    
    # Get unique employee IDs
    employee_ids = sorted(data['id'].unique())
    
    # Prepare data for CSV
    csv_data = []
    
    # Process each employee
    for emp_id in employee_ids:
        print(f'Analyzing employee {emp_id}...')
        # Filter data for current employee
        emp_data = data[data['id'] == emp_id]
        
        # Process each activity type
        for activity in ACTIVITY_TYPES:
            # Filter data for current activity
            activity_data = emp_data[emp_data['activity'] == activity]
            
            if activity_data.empty:
                # Create a row with zeros if no data
                row = {
                    'employee_id': emp_id,
                    'activity': activity,
                    'total_instances': 0
                }
                # Add zeros for each range
                for label in range_labels:
                    row[label] = 0.0
                
                csv_data.append(row)
                continue
            
            # Total number of instances for this activity
            total_instances = len(activity_data)
            
            # Initialize row
            row = {
                'employee_id': emp_id,
                'activity': activity,
                'total_instances': total_instances
            }
            
            # Calculate percentages for each range
            for i, (min_val, max_val) in enumerate(ranges):
                count = len(activity_data[(activity_data['duration'] >= min_val) & 
                                         (activity_data['duration'] < max_val)])
                percentage = count / total_instances * 100
                row[range_labels[i]] = round(percentage, 1)
            
            csv_data.append(row)
    
    # Create and save DataFrame
    result_df = pd.DataFrame(csv_data)
    output_file = f'{analysis_dir}/activity_duration_ranges.csv'
    result_df.to_csv(output_file, index=False)
    print(f'Saved activity duration analysis to {output_file}')
    
    # Also create a summary table with average by activity type
    print('Generating summary by activity type...')
    summary_data = []
    
    for activity in ACTIVITY_TYPES:
        activity_rows = result_df[result_df['activity'] == activity]
        
        if activity_rows.empty:
            continue
            
        # Calculate average percentages across employees
        row = {
            'activity': activity,
            'total_instances': activity_rows['total_instances'].sum()
        }
        
        # Calculate weighted average percentages
        for label in range_labels:
            weighted_avg = sum(activity_rows[label] * activity_rows['total_instances']) / row['total_instances']
            row[label] = round(weighted_avg, 1)
        
        summary_data.append(row)
    
    # Create and save summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_file = f'{analysis_dir}/activity_duration_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f'Saved activity duration summary to {summary_file}')

# Function to analyze region concentration by duration ranges
def analyze_region_concentration():
    print('\nAnalyzing region concentration by duration ranges...')
    
    # Define duration ranges (in seconds)
    ranges = [
        (0, 5),
        (5, 10),
        (10, 20),
        (20, 30),
        (30, 60),
        (60, 120),
        (120, 300),
        (300, float('inf'))
    ]

    # Define range labels
    range_labels = [
        '0-5 sec',
        '5-10 sec',
        '10-20 sec',
        '20-30 sec',
        '30-60 sec',
        '1-2 min',
        '2-5 min',
        '5+ min'
    ]
    
    # Ensure analysis directory exists
    analysis_dir = f'{OUTPUT_DIR}/analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load the raw data again to analyze individual instances
    print(f'Loading data from {DATA_PATH}...')
    data = pd.read_csv(DATA_PATH)
    
    # Get unique employee IDs
    employee_ids = sorted(data['id'].unique())
    
    # Prepare data for CSV
    region_analysis = []
    
    # Process each employee
    for emp_id in employee_ids:
        print(f'Analyzing regions for employee {emp_id}...')
        # Filter data for current employee
        emp_data = data[data['id'] == emp_id]
        
        # Process each activity type
        for activity in ACTIVITY_TYPES:
            # Filter data for current activity
            activity_data = emp_data[emp_data['activity'] == activity]
            
            if activity_data.empty:
                continue
            
            # Process each duration range
            for i, (min_val, max_val) in enumerate(ranges):
                range_label = range_labels[i]
                
                # Filter by duration range
                range_data = activity_data[(activity_data['duration'] >= min_val) & 
                                          (activity_data['duration'] < max_val)]
                
                if range_data.empty:
                    continue
                
                # Count instances by region
                region_counts = range_data['region'].value_counts()
                total_instances = len(range_data)
                
                # Get top region
                top_region = region_counts.index[0]
                top_count = region_counts.iloc[0]
                percentage = (top_count / total_instances) * 100
                
                # Store the results
                region_analysis.append({
                    'employee_id': emp_id,
                    'activity': activity,
                    'duration_range': range_label,
                    'total_instances': total_instances,
                    'top_region': top_region,
                    'top_region_instances': top_count,
                    'top_region_percentage': round(percentage, 1)
                })
    
    # Create and save DataFrame
    result_df = pd.DataFrame(region_analysis)
    output_file = f'{analysis_dir}/region_concentration_by_duration.csv'
    result_df.to_csv(output_file, index=False)
    print(f'Saved region concentration analysis to {output_file}')

# Run additional analyses
if __name__ == "__main__":
    analyze_activity_durations()
    analyze_region_concentration()
    print("\nProcessing complete!")