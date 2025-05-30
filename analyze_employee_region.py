#!/usr/bin/env python3
"""
Analyze time spent by an employee in a specific region by activity type.
"""
import pandas as pd
from datetime import timedelta

def format_duration(seconds):
    """Convert seconds to [hh]:mm:ss format"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    # Load the data
    print('Loading data...')
    df = pd.read_csv('data/raw/processed_sensor_data.csv')
    
    # Filter for employee 32-F and the specific region
    employee_id = '32-F'
    region = '3_Konditorei_deco_raum'
    filtered = df[(df['id'] == employee_id) & (df['region'] == region)]
    
    if filtered.empty:
        print(f"No data found for employee {employee_id} in region {region}")
        return
    
    # Group by activity and sum the duration (in seconds)
    activity_times = filtered.groupby('activity')['duration'].sum()
    
    # Calculate total time and percentages
    total_seconds = activity_times.sum()
    
    # Print results
    print(f"\nTime spent by employee {employee_id} in {region} by activity:")
    print('-' * 80)
    print(f"{'Activity':<20} | {'Duration':<10} | {'Percentage':<8} | Records")
    print('-' * 80)
    
    # Sort activities by duration (descending)
    for activity, seconds in activity_times.sort_values(ascending=False).items():
        percentage = (seconds / total_seconds * 100) if total_seconds > 0 else 0
        duration_str = format_duration(seconds)
        records = len(filtered[filtered['activity'] == activity])
        print(f"{activity:<20} | {duration_str:<10} | {percentage:5.1f}%    | {records:,}")
    
    print('-' * 80)
    print(f"{'TOTAL':<20} | {format_duration(total_seconds):<10} | 100.0%    | {len(filtered):,}")

if __name__ == "__main__":
    main()
