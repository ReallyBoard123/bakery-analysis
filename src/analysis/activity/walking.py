"""
Walking Analysis Module

Analyzes walking patterns of bakery employees to identify efficiency opportunities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

from src.utils.time_utils import format_seconds_to_hms
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import get_text

def analyze_walking_patterns(data, output_dir=None, language='en'):
    """
    Analyze walking patterns in the bakery dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with walking pattern analysis results
    """
    # Create subdirectories
    if output_dir:
        stats_dir = ensure_dir_exists(output_dir / 'statistics')
        vis_dir = ensure_dir_exists(output_dir / 'visualizations')
    
    # Filter for walking activities only
    walk_data = data[data['activity'] == 'Walk']
    
    if walk_data.empty:
        print("No walking data found in the dataset.")
        return {}
    
    results = {}
    
    # 1. Walk Duration Distribution
    print("  Analyzing walking duration distribution...")
    duration_results = analyze_walk_duration_distribution(walk_data, stats_dir, vis_dir, language)
    results['durations'] = duration_results
    
    # 2. Region Transition Analysis
    print("  Analyzing region transitions...")
    transition_results = analyze_region_transitions(data, stats_dir, vis_dir, language)
    results['transitions'] = transition_results
    
    # 3. Walking Path Analysis
    print("  Analyzing walking paths...")
    path_results = analyze_common_paths(data, stats_dir, vis_dir, language)
    results['paths'] = path_results
    
    # 4. Time of Day Analysis
    print("  Analyzing walking patterns by time of day...")
    time_results = analyze_walking_by_time(walk_data, stats_dir, vis_dir, language)
    results['time'] = time_results
    
    print(f"Walking analysis complete. Results saved to {output_dir}")
    return results

def analyze_walk_duration_distribution(walk_data, stats_dir=None, vis_dir=None, language='en'):
    """Analyze distribution of walking durations"""
    
    # Calculate overall statistics
    results = {
        'count': len(walk_data),
        'min': walk_data['duration'].min(),
        'max': walk_data['duration'].max(),
        'mean': walk_data['duration'].mean(),
        'median': walk_data['duration'].median(),
        'total_duration': walk_data['duration'].sum(),
        'formatted_total': format_seconds_to_hms(walk_data['duration'].sum()),
        'formatted_mean': format_seconds_to_hms(walk_data['duration'].mean())
    }
    
    # Calculate duration distribution
    duration_bins = [0, 5, 15, 30, 60, 120, 300, 600, float('inf')]
    bin_labels = ['0-5s', '5-15s', '15-30s', '30-60s', '1-2min', '2-5min', '5-10min', '>10min']
    walk_data['duration_category'] = pd.cut(walk_data['duration'], bins=duration_bins, labels=bin_labels)
    
    duration_dist = walk_data['duration_category'].value_counts().sort_index()
    results['duration_distribution'] = duration_dist.to_dict()
    
    # Save statistics
    if stats_dir:
        # Overall statistics
        stats_df = pd.DataFrame([results])
        stats_df.to_csv(stats_dir / 'walk_duration_statistics.csv', index=False)
        
        # Distribution data
        dist_df = pd.DataFrame({
            'duration_category': duration_dist.index,
            'count': duration_dist.values,
            'percentage': (duration_dist.values / duration_dist.sum() * 100).round(2)
        })
        dist_df.to_csv(stats_dir / 'walk_duration_distribution.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Main distribution plot
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(duration_dist.index, duration_dist.values, color='#156082')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:,}", ha='center', va='bottom')
        
        plt.title(get_text('Distribution of Walking Durations', language))
        plt.xlabel(get_text('Duration Category', language))
        plt.ylabel(get_text('Count', language))
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'walk_duration_distribution.png', dpi=300)
        plt.close()
        
        # Focus on short durations
        short_dist = duration_dist.iloc[:3]  # 0-5s, 5-15s, 15-30s
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(short_dist.index, short_dist.values, color='#156082')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:,}", ha='center', va='bottom')
        
        plt.title(get_text('Distribution of Short Walking Durations', language))
        plt.xlabel(get_text('Duration Category', language))
        plt.ylabel(get_text('Count', language))
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'walk_duration_distribution_short.png', dpi=300)
        plt.close()
    
    return results

def analyze_region_transitions(data, stats_dir=None, vis_dir=None, language='en'):
    """Analyze region transitions and whether they involve walking"""
    
    # Sort data by employee and time
    sorted_data = data.sort_values(['id', 'startTime'])
    
    # Track transitions
    transitions = []
    walking_transitions = []
    non_walking_transitions = []
    
    # For each employee
    prev_id = None
    prev_region = None
    prev_activity = None
    
    for _, row in sorted_data.iterrows():
        curr_id = row['id']
        curr_region = row['region']
        curr_activity = row['activity']
        
        # Reset if employee changed
        if curr_id != prev_id:
            prev_id = curr_id
            prev_region = curr_region
            prev_activity = curr_activity
            continue
        
        # Check for region change
        if prev_region and curr_region != prev_region:
            transition = (prev_region, curr_region)
            transitions.append(transition)
            
            # Check if transition involved walking
            if prev_activity == 'Walk' or curr_activity == 'Walk':
                walking_transitions.append(transition)
            else:
                non_walking_transitions.append(transition)
        
        # Update previous values
        prev_region = curr_region
        prev_activity = curr_activity
    
    # Calculate statistics
    total_transitions = len(transitions)
    walking_pct = (len(walking_transitions) / total_transitions * 100) if total_transitions > 0 else 0
    non_walking_pct = (len(non_walking_transitions) / total_transitions * 100) if total_transitions > 0 else 0
    
    # Calculate most common transitions
    all_transition_counts = Counter(transitions)
    walking_transition_counts = Counter(walking_transitions)
    non_walking_transition_counts = Counter(non_walking_transitions)
    
    results = {
        'total_transitions': total_transitions,
        'walking_transitions': len(walking_transitions),
        'non_walking_transitions': len(non_walking_transitions),
        'walking_percentage': walking_pct,
        'non_walking_percentage': non_walking_pct,
        'most_common_all': all_transition_counts.most_common(20),
        'most_common_walking': walking_transition_counts.most_common(20),
        'most_common_non_walking': non_walking_transition_counts.most_common(20)
    }
    
    # Save statistics
    if stats_dir:
        # Save overall summary
        summary_df = pd.DataFrame([{
            'total_transitions': results['total_transitions'],
            'walking_transitions': results['walking_transitions'],
            'non_walking_transitions': results['non_walking_transitions'],
            'walking_percentage': results['walking_percentage'],
            'non_walking_percentage': results['non_walking_percentage']
        }])
        summary_df.to_csv(stats_dir / 'region_transition_summary.csv', index=False)
        
        # Save most common transitions
        common_all_df = pd.DataFrame([
            {'from_region': src, 'to_region': dst, 'count': count, 
             'percentage': round(count / total_transitions * 100, 2) if total_transitions > 0 else 0}
            for (src, dst), count in all_transition_counts.most_common(50)
        ])
        if not common_all_df.empty:
            common_all_df.to_csv(stats_dir / 'most_common_transitions.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Pie chart of walking vs non-walking transitions
        plt.figure(figsize=(10, 8))
        
        plt.pie(
            [results['walking_transitions'], results['non_walking_transitions']],
            labels=['With Walking', 'Without Walking'],
            autopct='%1.1f%%',
            colors=['#156082', '#D3D3D3'],
            startangle=90
        )
        
        plt.title(get_text('Region Transitions: With vs. Without Walking', language))
        plt.tight_layout()
        plt.savefig(vis_dir / 'walking_vs_non_walking_transitions.png', dpi=300)
        plt.close()
    
    return results

def analyze_common_paths(data, stats_dir=None, vis_dir=None, language='en'):
    """Analyze common walking paths between regions"""
    
    # We'll focus on walking sequences - consecutive walking activities
    sorted_data = data.sort_values(['id', 'startTime'])
    
    # Track walking sequences
    walking_sequences = []
    current_sequence = None
    
    for emp_id, emp_data in sorted_data.groupby('id'):
        emp_data = emp_data.sort_values('startTime')
        
        for _, row in emp_data.iterrows():
            if row['activity'] == 'Walk':
                # Start a new sequence or continue current one
                if current_sequence is None or current_sequence['employee_id'] != emp_id:
                    if current_sequence is not None:
                        walking_sequences.append(current_sequence)
                        
                    current_sequence = {
                        'employee_id': emp_id,
                        'regions': [row['region']],
                        'durations': [row['duration']],
                        'start_time': row['startTime'],
                        'end_time': row['endTime']
                    }
                else:
                    # Continue current sequence
                    current_sequence['regions'].append(row['region'])
                    current_sequence['durations'].append(row['duration'])
                    current_sequence['end_time'] = row['endTime']
            else:
                # End current sequence
                if current_sequence is not None:
                    walking_sequences.append(current_sequence)
                    current_sequence = None
    
    # Add final sequence if exists
    if current_sequence is not None:
        walking_sequences.append(current_sequence)
    
    # Extract paths (consecutive unique regions)
    paths = []
    
    for seq in walking_sequences:
        if len(seq['regions']) > 1:
            # Extract unique sequential regions (remove consecutive duplicates)
            regions = seq['regions']
            unique_path = [regions[0]]
            
            for region in regions[1:]:
                if region != unique_path[-1]:
                    unique_path.append(region)
            
            # Only include paths with at least 2 unique regions
            if len(unique_path) >= 2:
                path_str = ' → '.join(unique_path)
                total_duration = sum(seq['durations'])
                
                paths.append({
                    'employee_id': seq['employee_id'],
                    'path': path_str,
                    'regions': unique_path,
                    'duration': total_duration,
                    'formatted_duration': format_seconds_to_hms(total_duration)
                })
    
    # Count common paths
    path_counts = Counter([p['path'] for p in paths])
    common_paths = path_counts.most_common(20)
    
    results = {
        'total_sequences': len(walking_sequences),
        'total_paths': len(paths),
        'unique_paths': len(path_counts),
        'common_paths': common_paths
    }
    
    # Save statistics
    if stats_dir and common_paths:
        # Save common paths
        common_paths_df = pd.DataFrame([
            {'path': path, 'count': count, 
             'percentage': round(count / len(paths) * 100, 2) if paths else 0}
            for path, count in common_paths
        ])
        common_paths_df.to_csv(stats_dir / 'common_walking_paths.csv', index=False)
    
    return results

def analyze_walking_by_time(walk_data, stats_dir=None, vis_dir=None, language='en'):
    """Analyze walking patterns by time of day"""
    
    results = {}
    
    # Check if time information is available
    if 'startTimeClock' not in walk_data.columns:
        print("  Time information not available in the dataset.")
        return results
    
    # Extract hour from startTimeClock
    walk_data['hour'] = walk_data['startTimeClock'].str.split(':', expand=True)[0].astype(int)
    
    # Calculate walking stats by hour
    hourly_stats = walk_data.groupby('hour').agg({
        'duration': ['sum', 'count', 'mean'],
        'id': 'nunique'  # Count unique employees
    })
    
    hourly_stats.columns = ['total_duration', 'count', 'mean_duration', 'employee_count']
    hourly_stats = hourly_stats.reset_index()
    
    # Add formatted time
    hourly_stats['formatted_total'] = hourly_stats['total_duration'].apply(format_seconds_to_hms)
    
    # Calculate percentage of total walking time
    total_walking = hourly_stats['total_duration'].sum()
    hourly_stats['percentage'] = (hourly_stats['total_duration'] / total_walking * 100).round(2)
    
    # Fill in missing hours
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly_stats = pd.merge(all_hours, hourly_stats, on='hour', how='left').fillna(0)
    
    results['hourly_stats'] = hourly_stats.to_dict('records')
    
    # Save statistics
    if stats_dir:
        hourly_stats.to_csv(stats_dir / 'walking_by_hour.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Walking time by hour
        plt.figure(figsize=(14, 8))
        
        # Convert to minutes for better visualization
        minutes = hourly_stats['total_duration'] / 60
        
        plt.bar(hourly_stats['hour'], minutes, color='#156082')
        
        plt.title(get_text('Walking Time by Hour of Day', language))
        plt.xlabel(get_text('Hour of Day', language))
        plt.ylabel(get_text('Duration (minutes)', language))
        plt.xticks(range(24))
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'walking_by_hour.png', dpi=300)
        plt.close()
    
    return results