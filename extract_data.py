import pandas as pd
import os
import glob

# Define paths
DATA_PATH = 'data/raw/processed_sensor_data.csv'
CONSOLIDATED_OUTPUT_FILE = 'consolidated_region_stats.csv'

# Load data
print(f"Loading data from {DATA_PATH}...")
data = pd.read_csv(DATA_PATH)

# Convert startTime to hours and minutes for filtering
data['hour'] = (data['startTime'] / 3600) % 24
data['hour'] = data['hour'].astype(int)
data['minute'] = (data['startTime'] % 3600) // 60
data['time_slot'] = data['hour'] * 60 + data['minute']

# Define 30-minute intervals
intervals = [(hour, minute) for hour in range(24) for minute in range(0, 60, 30)]

# Define activity types to process
activity_types = ['Walk', 'Stand', 'Handle center', 'Handle up', 'Handle down']

for activity in activity_types:
    print(f"Processing activity: {activity}...")

    # Initialize an empty DataFrame for consolidated data
    consolidated_data = pd.DataFrame()

    for hour, minute in intervals:
        # Calculate start and end of the interval in minutes
        start_time_slot = hour * 60 + minute
        end_time_slot = start_time_slot + 30

        # Filter data for the current interval and activity type
        target_data = data[(data['time_slot'] >= start_time_slot) & (data['time_slot'] < end_time_slot) & (data['activity'] == activity)]

        if target_data.empty:
            continue

        # Group by employee and region, calculate total duration
        region_stats = target_data.groupby(['id', 'region']).agg({
            'duration': ['sum', 'count'],
            'date': 'nunique',
        }).reset_index()

        # Flatten column names
        region_stats.columns = ['id', 'region', 'total_minutes', 'count', 'unique_dates']
        region_stats['total_minutes'] = region_stats['total_minutes'] / 60  # Convert to minutes

        # Add time category
        region_stats['time_category'] = f"{hour:02d}:{minute:02d}"

        # Append to the consolidated DataFrame
        consolidated_data = pd.concat([consolidated_data, region_stats], ignore_index=True)

    # Keep only the top region for each employee and time interval
    consolidated_data = consolidated_data.sort_values(['id', 'time_category', 'total_minutes'], ascending=[True, True, False])
    consolidated_data = consolidated_data.groupby(['id', 'time_category']).head(1).reset_index(drop=True)

    # Add a new column for 'duration' in [mm:ss] format
    consolidated_data['duration'] = consolidated_data['total_minutes'].apply(lambda x: f"{int(x):02d}:{int((x - int(x)) * 60):02d}")

    # Convert 'total_minutes' to float and round to 2 decimal places
    consolidated_data['total_minutes'] = consolidated_data['total_minutes'].round(2)

    # Save the consolidated data to a separate CSV file for each activity type
    output_file = f"output/{activity.lower().replace(' ', '_')}_region_stats.csv"
    consolidated_data[['id', 'region', 'time_category', 'duration', 'total_minutes']].to_csv(output_file, index=False)
    print(f"Consolidated data for activity '{activity}' saved to {output_file}")

# Group data by employee and region to calculate total walking time
employee_region_stats = data[data['activity'] == 'Walk'].groupby(['id', 'region']).agg({
    'duration': 'sum'
}).reset_index()

# Convert total duration to hours, minutes, and seconds
employee_region_stats['duration_hhmmss'] = employee_region_stats['duration'].apply(
    lambda x: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}"
)

# Sort by employee and total duration in descending order
employee_region_stats = employee_region_stats.sort_values(['id', 'duration'], ascending=[True, False])

# Keep only the top 3 regions for each employee
top_regions = employee_region_stats.groupby('id').head(3)

# Save each employee's top regions to a separate CSV file
output_dir = 'output/top_regions_by_employee/'
os.makedirs(output_dir, exist_ok=True)

for employee_id, group in top_regions.groupby('id'):
    output_file = os.path.join(output_dir, f"employee_{employee_id}_top_regions.csv")
    group[['region', 'duration_hhmmss']].to_csv(output_file, index=False)
    print(f"Top regions for employee {employee_id} saved to {output_file}")