"""
Workflow Analysis Module

Analyzes employee workflow patterns, movement, and productivity.
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import warnings

def analyze_walking_patterns(data, min_walk_duration=10):
    """
    Analyze walking paths to identify inefficient movement patterns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    min_walk_duration : int, optional
        Minimum walking duration in seconds to consider
    
    Returns:
    --------
    dict
        Dictionary containing walking pattern analysis by employee
    """
    # Get only walking activities of meaningful duration
    walking_data = data[(data['activity'] == 'Walk') & (data['duration'] >= min_walk_duration)]
    
    employee_walking_patterns = {}
    for emp_id, emp_walks in walking_data.groupby('id'):
        # Sort by time to get walking paths
        emp_walks = emp_walks.sort_values('startTime')
        
        # Track region transitions
        transitions = []
        regions_visited = []
        prev_region = None
        
        for _, row in emp_walks.iterrows():
            current_region = row['region']
            if prev_region and prev_region != current_region:
                transitions.append((prev_region, current_region))
                regions_visited.append(current_region)
            elif prev_region is None:
                regions_visited.append(current_region)
            prev_region = current_region
        
        # Calculate metrics for long-distance movements
        long_walks = emp_walks[emp_walks['duration'] > 30]  # Walks > 30 seconds
        
        # Identify frequent long walks between non-adjacent regions
        long_walk_regions = []
        for i in range(len(regions_visited) - 1):
            if i < len(long_walks):
                long_walk_regions.append((regions_visited[i], regions_visited[i+1]))
        
        employee_walking_patterns[emp_id] = {
            'total_walking_time': emp_walks['duration'].sum(),
            'avg_walk_duration': emp_walks['duration'].mean(),
            'long_walks_count': len(long_walks),
            'long_walks_total_time': long_walks['duration'].sum() if not long_walks.empty else 0,
            'common_long_walks': Counter(long_walk_regions).most_common(5),
            'most_visited_regions': Counter(regions_visited).most_common(5)
        }
        
    return employee_walking_patterns

def identify_workflow_patterns(data, min_sequence_support=3):
    """
    Identify common activity sequences to detect workflow patterns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    min_sequence_support : int, optional
        Minimum support count for sequences
    
    Returns:
    --------
    dict
        Dictionary containing workflow pattern analysis by employee
    """
    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import fpgrowth
    except ImportError:
        warnings.warn("mlxtend package not installed. Install with: pip install mlxtend")
        return {"error": "Required package 'mlxtend' not installed"}
    
    workflow_patterns = {}
    
    for emp_id, emp_data in data.groupby('id'):
        # Sort by time
        sorted_activities = emp_data.sort_values('startTime')
        
        # Create sequences of activities
        activity_sequences = []
        current_sequence = []
        
        for _, row in sorted_activities.iterrows():
            # Add region-activity pair to capture contextual information
            activity_context = f"{row['activity']}@{row['region']}"
            current_sequence.append(activity_context)
            
            # When we have a sequence of meaningful length, add it
            if len(current_sequence) >= 3:
                activity_sequences.append(tuple(current_sequence[-3:]))  # Use last 3 activities
        
        # Find frequent sequence patterns
        if activity_sequences:
            # Convert sequences to transaction format
            te = TransactionEncoder()
            te_ary = te.fit_transform(activity_sequences)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Extract frequent patterns
            frequent_patterns = fpgrowth(df, min_support=min_sequence_support/len(activity_sequences),
                                      use_colnames=True)
            
            if not frequent_patterns.empty:
                workflow_patterns[emp_id] = {
                    'patterns': frequent_patterns.sort_values('support', ascending=False),
                    'total_sequences': len(activity_sequences)
                }
    
    return workflow_patterns

def analyze_productivity_rhythm(data, window_size=15*60):  # 15-minute windows
    """
    Analyze productivity rhythms throughout shifts
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    window_size : int, optional
        Size of time window in seconds
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing productivity rhythm analysis
    """
    # Calculate handling-to-movement ratio as productivity proxy
    results = []
    
    for shift_id, shift_data in data.groupby('shift'):
        shift_start = shift_data['startTime'].min()
        shift_end = shift_data['endTime'].max()
        
        # Process each employee in this shift
        for emp_id, emp_shift_data in shift_data.groupby('id'):
            current_time = shift_start
            
            # Process time windows
            while current_time < shift_end:
                window_end = min(current_time + window_size, shift_end)
                
                # Get activities in this window
                window_data = emp_shift_data[(emp_shift_data['startTime'] >= current_time) & 
                                         (emp_shift_data['startTime'] < window_end)]
                
                if not window_data.empty:
                    # Calculate handling vs. movement in this window
                    handling_time = window_data[window_data['activity'].str.contains('Handle')]['duration'].sum()
                    walking_time = window_data[window_data['activity'] == 'Walk']['duration'].sum()
                    total_time = window_data['duration'].sum()
                    
                    # Calculate metrics
                    handling_ratio = handling_time / total_time if total_time > 0 else 0
                    walking_ratio = walking_time / total_time if total_time > 0 else 0
                    
                    # Get hour of day for circadian analysis
                    hour_of_day = (current_time / 3600) % 24
                    
                    results.append({
                        'shift': shift_id,
                        'employee': emp_id,
                        'window_start': current_time,
                        'window_end': window_end,
                        'hour_of_day': hour_of_day,
                        'handling_ratio': handling_ratio,
                        'walking_ratio': walking_ratio,
                        'handling_to_walking_ratio': (handling_time / walking_time) if walking_time > 0 else float('inf')
                    })
                
                current_time = window_end
    
    return pd.DataFrame(results)

def cluster_employee_work_patterns(data, n_clusters=3):
    """
    Cluster employees based on behavioral patterns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    n_clusters : int, optional
        Number of clusters to form
    
    Returns:
    --------
    dict
        Dictionary containing employee clustering results
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        warnings.warn("scikit-learn package not installed. Install with: pip install scikit-learn")
        return {"error": "Required package 'scikit-learn' not installed"}
    
    # Calculate behavioral features for each employee
    feature_data = []
    
    for emp_id, emp_data in data.groupby('id'):
        # Filter out potential false readings
        emp_data = emp_data[emp_data['duration'] >= 5]
        
        # Calculate basic metrics
        total_time = emp_data['duration'].sum()
        
        # Activity breakdown
        activity_times = {}
        for activity in ['Walk', 'Stand', 'Handle up', 'Handle center', 'Handle down']:
            act_time = emp_data[emp_data['activity'] == activity]['duration'].sum()
            activity_times[f"{activity.replace(' ', '_')}_pct"] = (act_time / total_time) * 100 if total_time > 0 else 0
        
        # Handling stats
        handling_data = emp_data[emp_data['activity'].str.contains('Handle')]
        avg_handling_time = handling_data['duration'].mean() if not handling_data.empty else 0
        
        # Region stats - how many regions visited
        distinct_regions = len(emp_data['region'].unique())
        
        # Transition frequency
        transitions = 0
        prev_region = None
        for _, row in emp_data.sort_values('startTime').iterrows():
            if prev_region and prev_region != row['region']:
                transitions += 1
            prev_region = row['region']
        
        # Transition rate (per hour)
        hours_worked = total_time / 3600
        transition_rate = transitions / hours_worked if hours_worked > 0 else 0
        
        # Build feature vector
        features = {
            'employee_id': emp_id,
            'avg_handling_time': avg_handling_time,
            'distinct_regions': distinct_regions,
            'transition_rate': transition_rate,
            **activity_times
        }
        
        feature_data.append(features)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_data)
    
    if len(feature_df) < n_clusters:
        return {"error": f"Not enough employees ({len(feature_df)}) for {n_clusters} clusters"}
    
    # Prepare for clustering
    X = feature_df.drop('employee_id', axis=1)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignments
    feature_df['cluster'] = clusters
    
    # Calculate cluster characteristics
    cluster_profiles = {}
    for cluster_id in range(n_clusters):
        cluster_data = feature_df[feature_df['cluster'] == cluster_id]
        cluster_profiles[cluster_id] = {
            'size': len(cluster_data),
            'employees': cluster_data['employee_id'].tolist(),
            'avg_features': cluster_data.drop(['employee_id', 'cluster'], axis=1).mean().to_dict()
        }
    
    return {
        'employee_clusters': feature_df,
        'cluster_profiles': cluster_profiles
    }

def identify_optimization_opportunities(walking_patterns, productivity_data, workflow_patterns):
    """
    Identify concrete optimization opportunities based on analysis results
    
    Parameters:
    -----------
    walking_patterns : dict
        Results from analyze_walking_patterns
    productivity_data : pandas.DataFrame
        Results from analyze_productivity_rhythm
    workflow_patterns : dict
        Results from identify_workflow_patterns
    
    Returns:
    --------
    dict
        Dictionary containing optimization recommendations
    """
    opportunities = {
        'movement_optimization': [],
        'workflow_optimization': [],
        'scheduling_recommendations': []
    }
    
    # Analyze walking patterns for movement optimization
    for emp_id, walking_data in walking_patterns.items():
        if walking_data['long_walks_count'] > 5:
            common_walks = walking_data['common_long_walks']
            for (src, dst), count in common_walks:
                if count > 3:  # If a path is taken frequently
                    opportunities['movement_optimization'].append({
                        'employee': emp_id,
                        'recommendation': f"Optimize path between {src} and {dst} (taken {count} times)",
                        'priority': 'High' if count > 5 else 'Medium'
                    })
    
    # Analyze productivity for scheduling recommendations
    if not productivity_data.empty:
        # Group by hour of day
        hourly_productivity = productivity_data.groupby('hour_of_day')['handling_ratio'].mean().reset_index()
        
        # Find productivity dips
        overall_avg = hourly_productivity['handling_ratio'].mean()
        low_productivity_hours = hourly_productivity[hourly_productivity['handling_ratio'] < overall_avg * 0.8]
        
        for _, row in low_productivity_hours.iterrows():
            hour = int(row['hour_of_day'])
            opportunities['scheduling_recommendations'].append({
                'target': 'All employees',
                'recommendation': f"Schedule breaks around hour {hour} (low productivity period)",
                'priority': 'Medium'
            })
    
    # Analyze workflow patterns
    for emp_id, workflow_data in workflow_patterns.items():
        if 'patterns' in workflow_data:
            patterns_df = workflow_data['patterns']
            if not patterns_df.empty:
                # Look for inefficient patterns (e.g., activity combinations with Walk between two Handles)
                for _, row in patterns_df.iterrows():
                    items = row['itemsets']
                    if len(items) >= 3:
                        items_list = list(items)
                        # Check for patterns like "Handle" -> "Walk" -> "Handle"
                        handle_walk_handle = any(
                            'Handle' in items_list[i] and 'Walk' in items_list[i+1] and 'Handle' in items_list[i+2]
                            for i in range(len(items_list)-2)
                        )
                        
                        if handle_walk_handle:
                            opportunities['workflow_optimization'].append({
                                'employee': emp_id,
                                'recommendation': "Reorganize workspace to reduce interruptions in handling activities",
                                'priority': 'High' if row['support'] > 0.1 else 'Medium'
                            })
                            break
    
    return opportunities