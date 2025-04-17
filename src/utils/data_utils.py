"""
Data Utilities Module

Functions for loading, preprocessing, and aggregating bakery employee tracking data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from PIL import Image
from ..utils.time_utils import format_seconds_to_hms

def load_sensor_data(file_path):
    """
    Load sensor data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Loaded and preprocessed dataframe
    """
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} records from {file_path}")
    return data

def load_floor_plan_data(layout_path, metadata_path, connections_path):
    """
    Load floor plan data from JSON files
    
    Parameters:
    -----------
    layout_path : str
        Path to the floor plan image
    metadata_path : str
        Path to the process metadata file
    connections_path : str
        Path to the floor plan connections file
    
    Returns:
    --------
    dict
        Dictionary containing floor plan data
    """
    try:
        # Load floor plan image
        floor_plan = Image.open(layout_path)
        print(f"Loaded floor plan image: {layout_path}")
    except Exception as e:
        print(f"Error loading floor plan image: {e}")
        return None
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata_path}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None
    
    # Load connections
    try:
        with open(connections_path, 'r') as f:
            connections = json.load(f)
        print(f"Loaded connections: {connections_path}")
    except Exception as e:
        print(f"Error loading connections: {e}")
        return None
    
    # Create uuid to name mapping
    uuid_to_name = {}
    name_to_uuid = {}
    region_coordinates = {}
    region_dimensions = {}
    region_categories = {}
    
    for region in metadata['layout']['regions']:
        uuid_to_name[region['uuid']] = region['name']
        name_to_uuid[region['name']] = region['uuid']
        
        # Calculate coordinates and dimensions for visualization
        x_center = (region['position_top_left_x'] + region['position_bottom_right_x']) / 2
        y_center = (region['position_top_left_y'] + region['position_bottom_right_y']) / 2
        width = region['position_bottom_right_x'] - region['position_top_left_x']
        height = region['position_bottom_right_y'] - region['position_top_left_y']
        
        region_coordinates[region['name']] = (x_center, y_center)
        region_dimensions[region['name']] = (width, height)
        
        # Get region category
        label_uuid = region['region_label_uuid']
        label = next((l for l in metadata['layout']['region_labels'] if l['uuid'] == label_uuid), None)
        if label:
            region_categories[region['name']] = label['name']
    
    # Convert connections from UUID to region names
    name_connections = {}
    for source_uuid, target_uuids in connections.items():
        if source_uuid in uuid_to_name:
            source_name = uuid_to_name[source_uuid]
            name_connections[source_name] = [uuid_to_name.get(uuid, '') for uuid in target_uuids if uuid in uuid_to_name]
    
    return {
        'floor_plan': floor_plan,
        'metadata': metadata,
        'connections': connections,
        'name_connections': name_connections,
        'uuid_to_name': uuid_to_name,
        'name_to_uuid': name_to_uuid,
        'region_coordinates': region_coordinates,
        'region_dimensions': region_dimensions,
        'region_categories': region_categories
    }

def filter_data(data, employees=None, shifts=None, regions=None, activities=None, date_range=None):
    """
    Filter data by multiple criteria
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    employees : list, optional
        List of employee IDs to include
    shifts : list, optional
        List of shifts to include
    regions : list, optional
        List of regions to include
    activities : list, optional
        List of activities to include
    date_range : tuple, optional
        (start_date, end_date) in YYYY-MM-DD format
    
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    filtered_data = data.copy()
    
    if employees:
        filtered_data = filtered_data[filtered_data['id'].isin(employees)]
    
    if shifts:
        filtered_data = filtered_data[filtered_data['shift'].isin(shifts)]
    
    if regions:
        filtered_data = filtered_data[filtered_data['region'].isin(regions)]
    
    if activities:
        filtered_data = filtered_data[filtered_data['activity'].isin(activities)]
    
    if date_range:
        start_date, end_date = date_range
        filtered_data = filtered_data[(filtered_data['date'] >= start_date) & 
                                    (filtered_data['date'] <= end_date)]
    
    return filtered_data

def classify_departments(data):
    """
    Add department classification (bread/cake)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with department column added
    """
    # Make a copy to avoid modifying the original
    data_with_dept = data.copy()
    
    # Add department based on employee ID
    bread_dept = ['32-A', '32-C', '32-D', '32-E']
    cake_dept = ['32-F', '32-G', '32-H']
    
    data_with_dept['department'] = data_with_dept['id'].apply(
        lambda x: 'Bread' if x in bread_dept else ('Cake' if x in cake_dept else 'Other')
    )
    
    return data_with_dept

def aggregate_by_shift(data):
    """
    Aggregate data by shift
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe aggregated by shift with total duration, percentage, and formatted time
    """
    shift_summary = data.groupby('shift')['duration'].sum().reset_index()
    shift_summary['percentage'] = (shift_summary['duration'] / shift_summary['duration'].sum() * 100).round(1)
    shift_summary['formatted_time'] = shift_summary['duration'].apply(format_seconds_to_hms)
    shift_summary = shift_summary.sort_values('shift')
    return shift_summary

def aggregate_by_activity(data):
    """
    Aggregate data by activity
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe aggregated by activity
    """
    activity_summary = data.groupby('activity')['duration'].sum().reset_index()
    activity_summary['percentage'] = (activity_summary['duration'] / activity_summary['duration'].sum() * 100).round(1)
    activity_summary['formatted_time'] = activity_summary['duration'].apply(format_seconds_to_hms)
    activity_summary = activity_summary.sort_values('duration', ascending=False)
    return activity_summary

def create_activity_profile_by_employee(data):
    """
    Create activity profile by employee
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame
        Pivot table with employees as index, activities as columns, and percentage as values
    """
    # Group by employee and activity
    activity_emp = data.groupby(['id', 'activity'])['duration'].sum().reset_index()
    
    # Get total duration by employee for percentage calculation
    emp_totals = data.groupby('id')['duration'].sum().reset_index()
    emp_totals = emp_totals.rename(columns={'duration': 'emp_total'})
    
    # Merge to get employee totals
    activity_emp = pd.merge(activity_emp, emp_totals, on='id')
    
    # Calculate percentage within each employee
    activity_emp['percentage'] = (activity_emp['duration'] / activity_emp['emp_total'] * 100).round(1)
    
    # Create pivot table
    profile = activity_emp.pivot_table(
        index='id',
        columns='activity',
        values='percentage',
        fill_value=0
    )
    
    return profile