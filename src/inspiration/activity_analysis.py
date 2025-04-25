"""
Activity Analysis Module

Analyzes the distribution of activities (walking, standing, handling)
across regions and employees. Calculates efficiency metrics and
productivity ratios.
"""

import pandas as pd
import numpy as np

def analyze_activities(data_dict):
    """
    Analyze employee activities in different regions
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing processed data and metadata
    
    Returns:
    --------
    dict
        Dictionary containing activity analysis results
    """
    data = data_dict['processed_data']
    region_categories = data_dict['region_categories']
    
    print("  Analyzing activity distributions...")
    
    # Calculate time spent on different activities
    activity_summary = data.groupby('activity')['duration'].agg(['sum', 'mean', 'count']).reset_index()
    activity_summary['percentage'] = activity_summary['sum'] / activity_summary['sum'].sum() * 100
    activity_summary = activity_summary.sort_values('sum', ascending=False)
    
    # Calculate time spent in each region
    region_summary = data.groupby('region')['duration'].agg(['sum', 'mean', 'count']).reset_index()
    region_summary['percentage'] = region_summary['sum'] / region_summary['sum'].sum() * 100
    region_summary = region_summary.sort_values('sum', ascending=False)
    
    # Calculate time spent by each employee
    employee_summary = data.groupby('id')['duration'].agg(['sum', 'mean', 'count']).reset_index()
    employee_summary['percentage'] = employee_summary['sum'] / employee_summary['sum'].sum() * 100
    
    # Calculate activity distribution by region
    region_activity = data.pivot_table(
        index='region',
        columns='activity',
        values='duration',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate activity distribution by employee
    employee_activity = data.pivot_table(
        index='id',
        columns='activity',
        values='duration',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate region efficiency metrics
    region_efficiency = {}
    for region in data['region'].unique():
        region_data = data[data['region'] == region]
        total_time = region_data['duration'].sum()
        
        if total_time > 0:
            # Get time spent on each activity
            walk_time = region_data[region_data['activity'] == 'Walk']['duration'].sum()
            stand_time = region_data[region_data['activity'] == 'Stand']['duration'].sum()
            handle_time = region_data[region_data['activity'].str.contains('Handle', case=False, na=False)]['duration'].sum()
            
            # Normalize by total time
            walk_ratio = walk_time / total_time
            stand_ratio = stand_time / total_time
            handle_ratio = handle_time / total_time
            
            # Calculate efficiency metrics
            movement_ratio = walk_time / (stand_time + 1e-6)  # Avoid division by zero
            productivity = handle_time / total_time  # Higher is better
            
            region_efficiency[region] = {
                'total_time': total_time,
                'walk_time': walk_time,
                'stand_time': stand_time,
                'handle_time': handle_time,
                'walk_ratio': walk_ratio,
                'stand_ratio': stand_ratio,
                'handle_ratio': handle_ratio,
                'movement_ratio': movement_ratio,
                'productivity': productivity
            }
    
    # Calculate employee efficiency metrics
    employee_efficiency = {}
    for emp_id in data['id'].unique():
        emp_data = data[data['id'] == emp_id]
        total_time = emp_data['duration'].sum()
        
        if total_time > 0:
            # Get time spent on each activity
            walk_time = emp_data[emp_data['activity'] == 'Walk']['duration'].sum()
            stand_time = emp_data[emp_data['activity'] == 'Stand']['duration'].sum()
            handle_time = emp_data[emp_data['activity'].str.contains('Handle', case=False, na=False)]['duration'].sum()
            
            # Calculate efficiency metrics
            movement_ratio = walk_time / (stand_time + 1e-6)  # Avoid division by zero
            productivity = handle_time / total_time  # Higher is better
            
            # Count region transitions as a measure of movement
            emp_sorted = emp_data.sort_values('startTime')
            transitions = 0
            prev_region = None
            
            for _, row in emp_sorted.iterrows():
                if prev_region and prev_region != row['region']:
                    transitions += 1
                prev_region = row['region']
            
            # Calculate transition frequency (transitions per hour)
            hours = total_time / 3600  # Convert seconds to hours
            transition_rate = transitions / hours if hours > 0 else 0
            
            employee_efficiency[emp_id] = {
                'total_time': total_time,
                'walk_time': walk_time,
                'stand_time': stand_time,
                'handle_time': handle_time,
                'walk_ratio': walk_time / total_time,
                'stand_ratio': stand_time / total_time,
                'handle_ratio': handle_time / total_time,
                'movement_ratio': movement_ratio,
                'productivity': productivity,
                'transitions': transitions,
                'transition_rate': transition_rate
            }
    
    # Calculate activity by region category
    category_activity = pd.DataFrame()
    
    if region_categories:
        # Create a region to category mapping
        category_map = pd.Series(region_categories)
        
        # Get region durations
        region_durations = data.groupby('region')['duration'].sum().reset_index()
        region_durations['category'] = region_durations['region'].map(category_map)
        
        # Group by category
        category_activity = region_durations.groupby('category')['duration'].sum().sort_values(ascending=False)
    
    # Calculate handling type distribution
    handling_distribution = data[data['activity'].str.contains('Handle', case=False, na=False)]
    handling_types = handling_distribution.groupby('activity')['duration'].sum()
    handling_types = handling_types.sort_values(ascending=False)
    
    print(f"  Analyzed activities across {len(region_efficiency)} regions and {len(employee_efficiency)} employees")
    
    return {
        'activity_summary': activity_summary,
        'region_summary': region_summary,
        'employee_summary': employee_summary,
        'region_activity': region_activity,
        'employee_activity': employee_activity,
        'region_efficiency': region_efficiency,
        'employee_efficiency': employee_efficiency,
        'category_activity': category_activity,
        'handling_types': handling_types
    }