"""
Activity Exploratory Analysis

Basic functions to analyze activity patterns and region transitions in bakery data.
This will help us understand the data structure before creating specialized modules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

def analyze_region_transitions(data):
    """
    Analyze how regions change and what activities cause the transitions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    
    Returns:
    --------
    dict
        Dictionary with region transition statistics
    """
    results = {}
    
    # Analyze by employee
    for emp_id, emp_data in data.groupby('id'):
        # Sort by time to ensure proper sequence
        emp_data = emp_data.sort_values('startTime')
        
        transitions = []
        transition_causes = defaultdict(int)
        
        # Identify region transitions
        prev_region = None
        for i, row in emp_data.iterrows():
            current_region = row['region']
            current_activity = row['activity']
            
            if prev_region and current_region != prev_region:
                transition = (prev_region, current_region)
                transitions.append(transition)
                transition_causes[current_activity] += 1
            
            prev_region = current_region
        
        # Calculate statistics
        total_transitions = len(transitions)
        walk_transitions = transition_causes.get('Walk', 0)
        walk_percentage = (walk_transitions / total_transitions * 100) if total_transitions > 0 else 0
        
        results[emp_id] = {
            'total_transitions': total_transitions,
            'by_activity': dict(transition_causes),
            'walk_transitions': walk_transitions,
            'walk_percentage': walk_percentage,
            'common_paths': Counter(transitions).most_common(5)
        }
    
    return results

def analyze_activity_durations(data, activity_type=None):
    """
    Analyze duration distributions for activities
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    activity_type : str, optional
        Specific activity to analyze (e.g., 'Walk', 'Stand')
        If None, analyze all activities
    
    Returns:
    --------
    dict
        Dictionary with duration statistics by employee and activity
    """
    # Filter data if activity type specified
    if activity_type:
        filtered_data = data[data['activity'] == activity_type]
    else:
        filtered_data = data
    
    # Filter out 'Unknown' activity
    filtered_data = filtered_data[filtered_data['activity'] != 'Unknown']
    
    results = {'overall': {}, 'by_employee': {}}
    
    # Overall statistics by activity
    for activity, act_data in filtered_data.groupby('activity'):
        durations = act_data['duration']
        
        # Define duration bins for categorization
        duration_bins = [0, 5, 10, 30, 60, 120, 300, 600, float('inf')]
        bin_labels = ['0-5s', '5-10s', '10-30s', '30s-1m', '1-2m', '2-5m', '5-10m', '>10m']
        
        # Categorize durations
        duration_categories = pd.cut(durations, bins=duration_bins, labels=bin_labels)
        category_counts = duration_categories.value_counts().sort_index()
        
        # Calculate statistics
        stats = {
            'count': len(durations),
            'min': durations.min(),
            'max': durations.max(),
            'mean': durations.mean(),
            'median': durations.median(),
            'std': durations.std(),
            'total': durations.sum(),
            'distribution': category_counts.to_dict()
        }
        
        results['overall'][activity] = stats
    
    # Statistics by employee and activity
    for emp_id, emp_data in filtered_data.groupby('id'):
        emp_results = {}
        
        for activity, act_data in emp_data.groupby('activity'):
            durations = act_data['duration']
            
            # Skip if no data
            if durations.empty:
                continue
                
            # Calculate statistics
            stats = {
                'count': len(durations),
                'min': durations.min(),
                'max': durations.max(),
                'mean': durations.mean(),
                'median': durations.median(),
                'std': durations.std(),
                'total': durations.sum()
            }
            
            emp_results[activity] = stats
        
        results['by_employee'][emp_id] = emp_results
    
    return results

def analyze_time_in_region(data):
    """
    Analyze the average time spent in each region before changing
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    
    Returns:
    --------
    dict
        Dictionary with time-in-region statistics
    """
    results = {'overall': {}, 'by_employee': {}}
    
    # Calculate time in region before changes
    region_durations = defaultdict(list)
    
    # Overall statistics
    for emp_id, emp_data in data.groupby('id'):
        # Sort by time
        emp_data = emp_data.sort_values('startTime')
        
        current_region = None
        region_start_time = None
        
        for i, row in emp_data.iterrows():
            if current_region is None:
                # First record
                current_region = row['region']
                region_start_time = row['startTime']
            elif row['region'] != current_region:
                # Region changed
                duration = row['startTime'] - region_start_time
                region_durations[current_region].append(duration)
                
                # Reset for next region
                current_region = row['region']
                region_start_time = row['startTime']
    
    # Calculate statistics for each region
    for region, durations in region_durations.items():
        if durations:
            stats = {
                'count': len(durations),
                'min': min(durations),
                'max': max(durations),
                'mean': sum(durations) / len(durations),
                'median': sorted(durations)[len(durations) // 2],
                'total': sum(durations)
            }
            results['overall'][region] = stats
    
    # Calculate by employee
    for emp_id, emp_data in data.groupby('id'):
        # Sort by time
        emp_data = emp_data.sort_values('startTime')
        
        emp_region_durations = defaultdict(list)
        current_region = None
        region_start_time = None
        
        for i, row in emp_data.iterrows():
            if current_region is None:
                # First record
                current_region = row['region']
                region_start_time = row['startTime']
            elif row['region'] != current_region:
                # Region changed
                duration = row['startTime'] - region_start_time
                emp_region_durations[current_region].append(duration)
                
                # Reset for next region
                current_region = row['region']
                region_start_time = row['startTime']
        
        # Calculate statistics for each region for this employee
        emp_results = {}
        for region, durations in emp_region_durations.items():
            if durations:
                stats = {
                    'count': len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'mean': sum(durations) / len(durations),
                    'median': sorted(durations)[len(durations) // 2],
                    'total': sum(durations)
                }
                emp_results[region] = stats
        
        results['by_employee'][emp_id] = emp_results
    
    return results

def print_exploratory_summary(data):
    """
    Print a summary of exploratory analysis findings
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    """
    # Get overall activity statistics
    activity_counts = data['activity'].value_counts()
    total_records = len(data)
    
    print("\n=== ACTIVITY DISTRIBUTION ===")
    for activity, count in activity_counts.items():
        percentage = (count / total_records) * 100
        print(f"{activity}: {count} records ({percentage:.2f}%)")
    
    # Get overall walking statistics
    walk_data = data[data['activity'] == 'Walk']
    total_duration = data['duration'].sum()
    walk_duration = walk_data['duration'].sum()
    walk_percentage = (walk_duration / total_duration) * 100
    
    print("\n=== WALKING SUMMARY ===")
    print(f"Total walking records: {len(walk_data)}")
    print(f"Total walking duration: {walk_duration:.2f} seconds ({walk_percentage:.2f}% of total time)")
    
    # Analyze walking durations
    walk_durations = walk_data['duration']
    if not walk_durations.empty:
        print(f"Walking duration range: {walk_durations.min():.2f} - {walk_durations.max():.2f} seconds")
        print(f"Mean walking duration: {walk_durations.mean():.2f} seconds")
        print(f"Median walking duration: {walk_durations.median():.2f} seconds")
    
    # Analyze region transitions
    transition_stats = analyze_region_transitions(data)
    
    print("\n=== REGION TRANSITIONS ===")
    for emp_id, stats in transition_stats.items():
        print(f"Employee {emp_id}:")
        print(f"  Total region transitions: {stats['total_transitions']}")
        print(f"  Walk transitions: {stats['walk_transitions']} ({stats['walk_percentage']:.2f}%)")
        
        if stats['common_paths']:
            print("  Most common paths:")
            for path, count in stats['common_paths']:
                from_region, to_region = path
                print(f"    {from_region} → {to_region}: {count} times")
        
        print()