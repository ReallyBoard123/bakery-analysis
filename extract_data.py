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
TOP_N_REGIONS = 1  # Keep top 3 regions per time category

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
    
    # Create a dictionary to store the accumulated durations
    # Key: (employee_id, time_category, region), Value: total_duration
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
            accumulated_durations[(row['id'], seconds_to_time_category(start_interval), row['region'])] += row['duration']
        else:
            # Process first interval
            first_threshold = interval_thresholds[0]
            first_duration = first_threshold - current
            accumulated_durations[(row['id'], seconds_to_time_category(start_interval), row['region'])] += first_duration
            current = first_threshold
            
            # Process middle intervals
            for i in range(len(interval_thresholds) - 1):
                interval_start = interval_thresholds[i]
                interval_end = interval_thresholds[i + 1]
                segment_duration = interval_end - interval_start
                interval_category = seconds_to_time_category(interval_start)
                accumulated_durations[(row['id'], interval_category, row['region'])] += segment_duration
            
            # Process last interval if activity doesn't end exactly on a threshold
            if current < end_time:
                last_interval = (current // INTERVAL_SECONDS) * INTERVAL_SECONDS
                last_duration = end_time - current
                accumulated_durations[(row['id'], seconds_to_time_category(last_interval), row['region'])] += last_duration
    
    # Convert accumulated durations to DataFrame
    result_data = []
    for (emp_id, time_category, region), total_duration in accumulated_durations.items():
        result_data.append({
            'id': emp_id,
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
    top_regions = result_df.sort_values(['id', 'time_category', 'total_minutes'], ascending=[True, True, False])
    top_regions = top_regions.groupby(['id', 'time_category']).head(TOP_N_REGIONS).reset_index(drop=True)
    
    # Format the final DataFrame
    top_regions = top_regions[['id', 'region', 'time_category', 'formatted_duration', 'total_minutes']]
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

print("Processing complete!")