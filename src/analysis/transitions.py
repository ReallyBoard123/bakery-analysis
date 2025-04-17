"""
Transitions Analysis Module

Analyzes movement and transitions between regions for bakery employees.
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

def calculate_region_transitions(data, employee_id=None, shift=None):
    """
    Calculate transitions between regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    employee_id : str, optional
        ID of the employee to analyze (if None, includes all employees)
    shift : int, optional
        Shift to analyze (if None, includes all shifts)
    
    Returns:
    --------
    dict
        Dictionary with region transition counts
    """
    # Filter data if employee_id or shift is specified
    filtered_data = data.copy()
    
    if employee_id is not None:
        filtered_data = filtered_data[filtered_data['id'] == employee_id]
    
    if shift is not None:
        filtered_data = filtered_data[filtered_data['shift'] == shift]
    
    # Dictionary to store transitions for each employee
    employee_transitions = {}
    
    # Calculate transitions for each employee
    for emp_id, emp_data in filtered_data.groupby('id'):
        transitions = []
        
        # Sort by start time
        emp_data = emp_data.sort_values('startTime')
        
        # Extract transitions
        prev_region = None
        for _, row in emp_data.iterrows():
            if prev_region and prev_region != row['region']:
                transitions.append((prev_region, row['region']))
            prev_region = row['region']
        
        # Count transitions
        transition_counts = Counter(transitions)
        
        employee_transitions[emp_id] = transition_counts
    
    return employee_transitions

def analyze_movement_patterns(data):
    """
    Analyze movement patterns and metrics per employee
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary containing movement metrics for each employee
    """
    employee_metrics = {}
    
    # Process each employee
    for emp_id, emp_data in data.groupby('id'):
        # Sort by time
        emp_data_sorted = emp_data.sort_values('startTime')
        
        # Calculate transitions
        transitions = []
        prev_region = None
        
        for _, row in emp_data_sorted.iterrows():
            if prev_region and prev_region != row['region']:
                transitions.append((prev_region, row['region']))
            prev_region = row['region']
        
        # Calculate average time spent in regions
        region_durations = emp_data.groupby('region')['duration'].agg(['sum', 'mean', 'count']).reset_index()
        
        # Store metrics
        employee_metrics[emp_id] = {
            'transition_count': len(transitions),
            'unique_transitions': len(set(transitions)),
            'regions_visited': len(region_durations),
            'region_durations': region_durations,
            'transition_details': transitions
        }
        
    return employee_metrics

def analyze_department_transitions(data, floor_plan_data=None):
    """
    Analyze transitions between departments
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    floor_plan_data : dict, optional
        Dictionary with floor plan data including region categories
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with department transition counts
    """
    # Get region categories if floor_plan_data is provided
    region_categories = {}
    if floor_plan_data and 'region_categories' in floor_plan_data:
        region_categories = floor_plan_data['region_categories']
    
    # Categorize regions into departments if categories not provided
    if not region_categories:
        for region in data['region'].unique():
            if 'Bread' in region or 'brot' in region.lower():
                region_categories[region] = 'Bread'
            elif 'Cake' in region or 'konditorei' in region.lower():
                region_categories[region] = 'Cake'
            else:
                region_categories[region] = 'Other'
    
    # Add category column to data
    data_with_categories = data.copy()
    data_with_categories['region_category'] = data_with_categories['region'].map(region_categories)
    
    # Calculate transitions between categories
    category_transitions = []
    
    for emp_id, emp_data in data_with_categories.groupby('id'):
        # Sort by start time
        emp_data = emp_data.sort_values('startTime')
        
        # Extract transitions
        prev_category = None
        for _, row in emp_data.iterrows():
            if prev_category and prev_category != row['region_category']:
                category_transitions.append({
                    'id': emp_id,
                    'from_department': prev_category,
                    'to_department': row['region_category'],
                    'time': row['startTime']
                })
            prev_category = row['region_category']
    
    # Convert to DataFrame
    if category_transitions:
        transitions_df = pd.DataFrame(category_transitions)
        
        # Count transitions
        transition_counts = transitions_df.groupby(['from_department', 'to_department']).size().reset_index(name='count')
        
        return transition_counts
    else:
        return pd.DataFrame(columns=['from_department', 'to_department', 'count'])