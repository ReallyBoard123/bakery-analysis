"""
Statistical Analysis Module

Performs statistical analysis of bakery employee tracking data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

from ..utils.time_utils import format_seconds_to_hms

def analyze_employee_differences(data):
    """
    Analyze differences in employee behavior
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary containing results of various analyses
    """
    results = {}
    
    # 1. Analyze walking patterns by employee
    walk_data = data[data['activity'] == 'Walk']
    if len(walk_data) > 0:
        # Summarize walking duration by employee
        walk_summary = walk_data.groupby('id')['duration'].agg(['mean', 'median', 'std', 'count']).reset_index()
        walk_summary['mean_mins'] = (walk_summary['mean'] / 60).round(2)  # Convert to minutes
        walk_summary['median_mins'] = (walk_summary['median'] / 60).round(2)  # Convert to minutes
        
        # Format time columns
        walk_summary['formatted_mean'] = walk_summary['mean'].apply(format_seconds_to_hms)
        walk_summary['formatted_median'] = walk_summary['median'].apply(format_seconds_to_hms)
        
        results['walking_patterns'] = walk_summary
    
    # 2. Analyze handling patterns by employee
    handle_data = data[data['activity'].str.contains('Handle')]
    if len(handle_data) > 0:
        # Summarize handling duration by employee and type
        handle_summary = handle_data.groupby(['id', 'activity'])['duration'].agg(['mean', 'median', 'std', 'count']).reset_index()
        handle_summary['mean_mins'] = (handle_summary['mean'] / 60).round(2)  # Convert to minutes
        handle_summary['median_mins'] = (handle_summary['median'] / 60).round(2)  # Convert to minutes
        
        # Format time columns
        handle_summary['formatted_mean'] = handle_summary['mean'].apply(format_seconds_to_hms)
        handle_summary['formatted_median'] = handle_summary['median'].apply(format_seconds_to_hms)
        
        results['handling_patterns'] = handle_summary
    
    # 3. Calculate productivity metrics
    productive_activities = ['Handle center', 'Handle up', 'Handle down']
    
    productivity_metrics = []
    for emp_id, group in data.groupby('id'):
        productive_time = group[group['activity'].isin(productive_activities)]['duration'].sum()
        walking_time = group[group['activity'] == 'Walk']['duration'].sum()
        standing_time = group[group['activity'] == 'Stand']['duration'].sum()
        total_time = group['duration'].sum()
        
        productivity_metrics.append({
            'id': emp_id,
            'productive_time': productive_time,
            'walking_time': walking_time,
            'standing_time': standing_time,
            'total_time': total_time,
            'productive_percentage': (productive_time / total_time * 100).round(2) if total_time > 0 else 0,
            'walking_percentage': (walking_time / total_time * 100).round(2) if total_time > 0 else 0,
            'standing_percentage': (standing_time / total_time * 100).round(2) if total_time > 0 else 0
        })
    
    if productivity_metrics:
        productivity_df = pd.DataFrame(productivity_metrics)
        
        # Format time columns
        for col in ['productive_time', 'walking_time', 'standing_time', 'total_time']:
            productivity_df[f'formatted_{col}'] = productivity_df[col].apply(format_seconds_to_hms)
        
        results['productivity_metrics'] = productivity_df
    
    return results

def analyze_ergonomic_patterns(data):
    """
    Analyze ergonomic patterns based on handling positions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary containing ergonomic analysis results
    """
    results = {}
    
    # Filter to handling activities
    handle_data = data[data['activity'].str.contains('Handle')]
    
    if len(handle_data) > 0:
        # Summarize time spent in different handling positions
        position_summary = handle_data.groupby('activity')['duration'].sum().reset_index()
        total_handling_time = position_summary['duration'].sum()
        position_summary['percentage'] = (position_summary['duration'] / total_handling_time * 100).round(2)
        position_summary['formatted_time'] = position_summary['duration'].apply(format_seconds_to_hms)
        results['position_summary'] = position_summary
        
        # Analyze handling positions by employee
        emp_position_summary = handle_data.groupby(['id', 'activity'])['duration'].sum().reset_index()
        
        # Calculate total handling time by employee
        emp_totals = handle_data.groupby('id')['duration'].sum().reset_index()
        emp_totals = emp_totals.rename(columns={'duration': 'total_handling'})
        
        # Calculate percentages
        emp_position_summary = pd.merge(emp_position_summary, emp_totals, on='id')
        emp_position_summary['percentage'] = (emp_position_summary['duration'] / emp_position_summary['total_handling'] * 100).round(2)
        emp_position_summary['formatted_time'] = emp_position_summary['duration'].apply(format_seconds_to_hms)
        
        results['employee_position_summary'] = emp_position_summary
    
    return results

def compare_shifts(data):
    """
    Compare efficiency and patterns across different shifts
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Dictionary containing shift comparison results
    """
    results = {}
    
    # Calculate metrics by shift
    shift_metrics = []
    for shift, group in data.groupby('shift'):
        # Activity distribution
        activity_distribution = group.groupby('activity')['duration'].sum()
        total_duration = activity_distribution.sum()
        
        # Calculate productive time (handling)
        productive_time = group[group['activity'].str.contains('Handle')]['duration'].sum()
        walking_time = group[group['activity'] == 'Walk']['duration'].sum()
        standing_time = group[group['activity'] == 'Stand']['duration'].sum()
        
        # Calculate employee count
        employee_count = group['id'].nunique()
        
        # Calculate region count
        region_count = group['region'].nunique()
        
        # Store metrics
        shift_metrics.append({
            'shift': shift,
            'total_duration': total_duration,
            'formatted_total': format_seconds_to_hms(total_duration),
            'productive_time': productive_time,
            'formatted_productive': format_seconds_to_hms(productive_time),
            'productive_percentage': (productive_time / total_duration * 100).round(2) if total_duration > 0 else 0,
            'walking_time': walking_time,
            'formatted_walking': format_seconds_to_hms(walking_time),
            'walking_percentage': (walking_time / total_duration * 100).round(2) if total_duration > 0 else 0,
            'standing_time': standing_time,
            'formatted_standing': format_seconds_to_hms(standing_time),
            'standing_percentage': (standing_time / total_duration * 100).round(2) if total_duration > 0 else 0,
            'employee_count': employee_count,
            'region_count': region_count,
            'activities_per_hour': (group['activity'].nunique() / (total_duration / 3600)).round(2) if total_duration > 0 else 0,
            'regions_per_hour': (region_count / (total_duration / 3600)).round(2) if total_duration > 0 else 0
        })
    
    if shift_metrics:
        results['shift_metrics'] = pd.DataFrame(shift_metrics)
    
    return results