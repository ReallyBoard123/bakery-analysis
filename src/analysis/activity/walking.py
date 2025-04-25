"""
Improved Walking Analysis Module

Analyzes walking patterns of bakery employees to identify inefficiencies and optimization opportunities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.patches as mpatches

from src.utils.time_utils import format_seconds_to_hms, format_seconds_to_minutes
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import save_figure, set_visualization_style, get_text

# Updated color scheme as specified
ACTIVITY_COLORS = {
    'Walk': '#156082',
    'Stand': '#A6A6A6',
    'Handle center': '#F1A983',
    'Handle down': '#C00000',
    'Handle up': '#FFC000'
}

# Standardized activity order
ACTIVITY_ORDER = ['Walk', 'Stand', 'Handle center', 'Handle down', 'Handle up']

def analyze_walking_patterns(data, output_dir=None, language='en'):
    """
    Analyze walking patterns in the bakery dataset with improved methodology
    
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
    walk_data = data[data['activity'] == 'Walk'].copy()
    
    if walk_data.empty:
        print("No walking data found in the dataset.")
        return {}
    
    results = {}
    
    # 1. Walk Duration Distribution
    print("  Analyzing walking duration distribution...")
    duration_results = analyze_walk_duration_distribution(walk_data, stats_dir, vis_dir, language)
    results['duration_distribution'] = duration_results
    
    # 2. Region Transition Analysis
    print("  Analyzing region transitions...")
    transition_results = analyze_region_transitions(data, stats_dir, vis_dir, language)
    results['region_transitions'] = transition_results
    
    # 3. Walking Sequence Analysis
    print("  Analyzing walking sequences...")
    sequence_results = analyze_walking_sequences(data, stats_dir, vis_dir, language)
    results['walking_sequences'] = sequence_results
    
    # 4. Time of Day Analysis
    print("  Analyzing walking patterns by time of day...")
    time_results = analyze_walking_by_time(walk_data, stats_dir, vis_dir, language)
    results['time_of_day'] = time_results
    
    # 5. Employee and Department Comparison
    print("  Comparing walking patterns between employees and departments...")
    comparison_results = compare_walking_patterns(data, walk_data, stats_dir, vis_dir, language)
    results['comparisons'] = comparison_results
    
    print(f"Walking analysis complete. Results saved to {output_dir}")
    return results

def analyze_walk_duration_distribution(walk_data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze the distribution of walking durations with improved granularity
    
    Parameters:
    -----------
    walk_data : pandas.DataFrame
        Input dataframe filtered for walking activities
    stats_dir : Path, optional
        Directory to save statistics
    vis_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with walking duration statistics
    """
    results = {}
    
    # Create more granular bins for short durations
    duration_bins = [0, 1, 2, 3, 4, 5, 10, 15, 30, 60, 120, 300, float('inf')]
    bin_labels = ['0-1s', '1-2s', '2-3s', '3-4s', '4-5s', '5-10s', '10-15s', 
                 '15-30s', '30-60s', '1-2min', '2-5min', '>5min']
    
    walk_data['duration_category'] = pd.cut(walk_data['duration'], bins=duration_bins, labels=bin_labels)
    
    # Calculate overall distribution
    duration_dist = walk_data['duration_category'].value_counts().sort_index()
    results['overall_distribution'] = duration_dist.to_dict()
    
    # Calculate distribution by employee
    emp_distributions = {}
    
    for emp_id, emp_data in walk_data.groupby('id'):
        emp_dist = emp_data['duration_category'].value_counts().sort_index()
        emp_distributions[emp_id] = emp_dist.to_dict()
    
    results['by_employee'] = emp_distributions
    
    # Calculate summary statistics
    overall_stats = {
        'count': len(walk_data),
        'total_duration': walk_data['duration'].sum(),
        'min': walk_data['duration'].min(),
        'max': walk_data['duration'].max(),
        'mean': walk_data['duration'].mean(),
        'median': walk_data['duration'].median(),
        'std': walk_data['duration'].std(),
        'percentiles': {
            '10th': walk_data['duration'].quantile(0.1),
            '25th': walk_data['duration'].quantile(0.25),
            '75th': walk_data['duration'].quantile(0.75),
            '90th': walk_data['duration'].quantile(0.9),
            '95th': walk_data['duration'].quantile(0.95)
        }
    }
    
    results['overall_stats'] = overall_stats
    
    # Save statistics to CSV
    if stats_dir:
        # Overall distribution
        dist_df = pd.DataFrame({
            'duration_category': duration_dist.index,
            'count': duration_dist.values,
            'percentage': (duration_dist.values / duration_dist.sum() * 100).round(2)
        })
        dist_df.to_csv(stats_dir / 'walk_duration_distribution.csv', index=False)
        
        # Overall statistics
        stats_df = pd.DataFrame([{
            'count': overall_stats['count'],
            'total_duration': overall_stats['total_duration'],
            'min': overall_stats['min'],
            'max': overall_stats['max'],
            'mean': overall_stats['mean'],
            'median': overall_stats['median'],
            'std': overall_stats['std'],
            'p10': overall_stats['percentiles']['10th'],
            'p25': overall_stats['percentiles']['25th'],
            'p75': overall_stats['percentiles']['75th'],
            'p90': overall_stats['percentiles']['90th'],
            'p95': overall_stats['percentiles']['95th'],
            'formatted_total': format_seconds_to_hms(overall_stats['total_duration']),
            'formatted_mean': format_seconds_to_hms(overall_stats['mean']),
            'formatted_median': format_seconds_to_hms(overall_stats['median'])
        }])
        stats_df.to_csv(stats_dir / 'walk_duration_statistics.csv', index=False)
        
        # Employee distributions
        emp_dist_data = []
        for emp_id, dist in emp_distributions.items():
            for category, count in dist.items():
                emp_dist_data.append({
                    'employee_id': emp_id,
                    'duration_category': category,
                    'count': count
                })
        
        if emp_dist_data:
            pd.DataFrame(emp_dist_data).to_csv(stats_dir / 'walk_duration_by_employee.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Walking duration distribution
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate percentages for better visualization
        percentages = (duration_dist.values / duration_dist.sum() * 100).round(2)
        
        # Create bar chart
        bars = ax.bar(duration_dist.index, percentages, color=ACTIVITY_COLORS['Walk'])
        
        # Add data labels
        for bar, percentage, count in zip(bars, percentages, duration_dist.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.5,
                f"{percentage:.1f}%\n({count:,})",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Distribution of Walking Durations', language), fontsize=16)
        ax.set_xlabel(get_text('Duration Category', language), fontsize=14)
        ax.set_ylabel(get_text('Percentage (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, vis_dir / 'walk_duration_distribution.png')
        
        # Create a zoomed-in version focusing on short durations (0-5s)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get only the first 5 categories (0-5s)
        short_dist = duration_dist.iloc[:5]
        short_percentages = (short_dist.values / duration_dist.sum() * 100).round(2)
        
        # Create bar chart
        bars = ax.bar(short_dist.index, short_percentages, color=ACTIVITY_COLORS['Walk'])
        
        # Add data labels
        for bar, percentage, count in zip(bars, short_percentages, short_dist.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.5,
                f"{percentage:.1f}%\n({count:,})",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Distribution of Short Walking Durations (0-5s)', language), fontsize=16)
        ax.set_xlabel(get_text('Duration Category', language), fontsize=14)
        ax.set_ylabel(get_text('Percentage (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, vis_dir / 'walk_duration_distribution_short.png')
    
    return results

def analyze_region_transitions(data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze region transitions and whether they occur during walking or other activities
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    stats_dir : Path, optional
        Directory to save statistics
    vis_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with region transition analysis results
    """
    results = {'overall': {}, 'by_employee': {}}
    
    # Sort data by time for each employee
    sorted_data = data.sort_values(['id', 'startTime'])
    
    # Initialize counters
    all_transitions = []
    walking_transitions = []
    non_walking_transitions = []
    employee_transitions = defaultdict(lambda: {'walking': [], 'non_walking': []})
    
    # Identify region transitions
    prev_id = None
    prev_region = None
    prev_activity = None
    
    for i, row in sorted_data.iterrows():
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
        if curr_region != prev_region:
            transition = (prev_region, curr_region)
            all_transitions.append(transition)
            
            # Check if transition happened during walking
            if prev_activity == 'Walk' or curr_activity == 'Walk':
                walking_transitions.append(transition)
                employee_transitions[curr_id]['walking'].append(transition)
            else:
                non_walking_transitions.append(transition)
                employee_transitions[curr_id]['non_walking'].append(transition)
        
        # Update previous values
        prev_region = curr_region
        prev_activity = curr_activity
    
    # Calculate transition statistics
    all_transition_counts = Counter(all_transitions)
    walking_transition_counts = Counter(walking_transitions)
    non_walking_transition_counts = Counter(non_walking_transitions)
    
    # Calculate percentage of transitions with walking vs. without
    total_transitions = len(all_transitions)
    walking_percentage = (len(walking_transitions) / total_transitions * 100) if total_transitions > 0 else 0
    non_walking_percentage = (len(non_walking_transitions) / total_transitions * 100) if total_transitions > 0 else 0
    
    # Store overall results
    results['overall'] = {
        'total_transitions': total_transitions,
        'walking_transitions': len(walking_transitions),
        'non_walking_transitions': len(non_walking_transitions),
        'walking_percentage': walking_percentage,
        'non_walking_percentage': non_walking_percentage,
        'most_common_all': all_transition_counts.most_common(20),
        'most_common_walking': walking_transition_counts.most_common(20),
        'most_common_non_walking': non_walking_transition_counts.most_common(20)
    }
    
    # Calculate employee-specific statistics
    for emp_id, transitions in employee_transitions.items():
        walking_trans = transitions['walking']
        non_walking_trans = transitions['non_walking']
        
        total_emp_transitions = len(walking_trans) + len(non_walking_trans)
        walking_pct = (len(walking_trans) / total_emp_transitions * 100) if total_emp_transitions > 0 else 0
        non_walking_pct = (len(non_walking_trans) / total_emp_transitions * 100) if total_emp_transitions > 0 else 0
        
        results['by_employee'][emp_id] = {
            'total_transitions': total_emp_transitions,
            'walking_transitions': len(walking_trans),
            'non_walking_transitions': len(non_walking_trans),
            'walking_percentage': walking_pct,
            'non_walking_percentage': non_walking_pct,
            'most_common_walking': Counter(walking_trans).most_common(10),
            'most_common_non_walking': Counter(non_walking_trans).most_common(10)
        }
    
    # Save statistics
    if stats_dir:
        # Overall transition statistics
        overall_df = pd.DataFrame([{
            'total_transitions': results['overall']['total_transitions'],
            'walking_transitions': results['overall']['walking_transitions'],
            'non_walking_transitions': results['overall']['non_walking_transitions'],
            'walking_percentage': results['overall']['walking_percentage'],
            'non_walking_percentage': results['overall']['non_walking_percentage']
        }])
        overall_df.to_csv(stats_dir / 'region_transition_summary.csv', index=False)
        
        # Top transitions
        top_all_df = pd.DataFrame([
            {
                'from_region': from_region,
                'to_region': to_region,
                'count': count,
                'percentage': (count / total_transitions * 100).round(2) if total_transitions > 0 else 0
            }
            for (from_region, to_region), count in all_transition_counts.most_common(50)
        ])
        if not top_all_df.empty:
            top_all_df.to_csv(stats_dir / 'top_region_transitions.csv', index=False)
        
        # Transitions with walking
        top_walking_df = pd.DataFrame([
            {
                'from_region': from_region,
                'to_region': to_region,
                'count': count,
                'percentage': (count / len(walking_transitions) * 100).round(2) if walking_transitions else 0
            }
            for (from_region, to_region), count in walking_transition_counts.most_common(50)
        ])
        if not top_walking_df.empty:
            top_walking_df.to_csv(stats_dir / 'top_walking_transitions.csv', index=False)
        
        # Transitions without walking
        top_non_walking_df = pd.DataFrame([
            {
                'from_region': from_region,
                'to_region': to_region,
                'count': count,
                'percentage': (count / len(non_walking_transitions) * 100).round(2) if non_walking_transitions else 0
            }
            for (from_region, to_region), count in non_walking_transition_counts.most_common(50)
        ])
        if not top_non_walking_df.empty:
            top_non_walking_df.to_csv(stats_dir / 'top_non_walking_transitions.csv', index=False)
        
        # Employee transition statistics
        emp_transition_data = []
        for emp_id, stats in results['by_employee'].items():
            emp_transition_data.append({
                'employee_id': emp_id,
                'total_transitions': stats['total_transitions'],
                'walking_transitions': stats['walking_transitions'],
                'non_walking_transitions': stats['non_walking_transitions'],
                'walking_percentage': stats['walking_percentage'],
                'non_walking_percentage': stats['non_walking_percentage']
            })
        
        if emp_transition_data:
            pd.DataFrame(emp_transition_data).to_csv(stats_dir / 'employee_transition_statistics.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Pie chart of walking vs. non-walking transitions
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = [
            get_text('Transitions with Walking', language), 
            get_text('Transitions without Walking', language)
        ]
        values = [results['overall']['walking_transitions'], results['overall']['non_walking_transitions']]
        colors = [ACTIVITY_COLORS['Walk'], '#D3D3D3']  # Blue for walking, grey for non-walking
        
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'width': 0.5, 'edgecolor': 'w'}
        )
        
        # Customize text
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title(get_text('Region Transitions: Walking vs. Non-Walking', language), fontsize=16)
        
        plt.tight_layout()
        save_figure(fig, vis_dir / 'transition_walking_distribution.png')
        
        # Bar chart of top walking transitions
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get top 15 walking transitions
        top_walking = walking_transition_counts.most_common(15)
        
        if top_walking:
            # Create labels and values
            labels = [f"{from_region} → {to_region}" for (from_region, to_region), _ in top_walking]
            values = [count for _, count in top_walking]
            
            # Create horizontal bar chart
            bars = ax.barh(labels, values, color=ACTIVITY_COLORS['Walk'])
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + 0.5,
                    bar.get_y() + bar.get_height()/2,
                    f"{width:,}",
                    ha='left',
                    va='center',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.set_title(get_text('Top Region Transitions with Walking', language), fontsize=16)
            ax.set_xlabel(get_text('Number of Transitions', language), fontsize=14)
            ax.set_ylabel(get_text('Transition Path', language), fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'top_walking_transitions.png')
        
        # Bar chart of walking transition percentage by employee
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare employee data
        emp_data = pd.DataFrame([
            {
                'employee_id': emp_id,
                'walking_percentage': stats['walking_percentage'],
                'walking_count': stats['walking_transitions'],
                'total_transitions': stats['total_transitions']
            }
            for emp_id, stats in results['by_employee'].items()
            if stats['total_transitions'] > 0  # Only include employees with transitions
        ])
        
        if not emp_data.empty:
            # Sort by walking percentage
            emp_data = emp_data.sort_values('walking_percentage', ascending=False)
            
            # Create bar chart
            bars = ax.bar(
                emp_data['employee_id'],
                emp_data['walking_percentage'],
                color=ACTIVITY_COLORS['Walk']
            )
            
            # Add data labels
            for bar, walking_count, total in zip(
                bars, emp_data['walking_count'], emp_data['total_transitions']
            ):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,
                    f"{height:.1f}%\n({walking_count:,}/{total:,})",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.set_title(get_text('Percentage of Region Transitions with Walking by Employee', language), fontsize=16)
            ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
            ax.set_ylabel(get_text('Walking Transitions (%)', language), fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            # Add average line
            avg_pct = results['overall']['walking_percentage']
            ax.axhline(
                avg_pct, 
                color='red', 
                linestyle='--', 
                label=f"{get_text('Overall Average', language)}: {avg_pct:.1f}%"
            )
            ax.legend()
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'walking_transition_by_employee.png')
    
    return results

def analyze_walking_sequences(data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze walking sequences and identify continuous walking episodes across regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    stats_dir : Path, optional
        Directory to save statistics
    vis_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with walking sequence analysis results
    """
    results = {'overall': {}, 'by_employee': {}}
    
    # Sort data by time for each employee
    sorted_data = data.sort_values(['id', 'startTime'])
    
    # Identify continuous walking sequences
    walking_sequences = []
    current_sequence = None
    
    for i, row in sorted_data.iterrows():
        curr_id = row['id']
        curr_activity = row['activity']
        
        if curr_activity == 'Walk':
            # Start a new sequence if needed
            if current_sequence is None or current_sequence['employee_id'] != curr_id:
                if current_sequence is not None:
                    walking_sequences.append(current_sequence)
                
                current_sequence = {
                    'employee_id': curr_id,
                    'start_time': row['startTime'],
                    'regions': [row['region']],
                    'durations': [row['duration']],
                    'end_time': row['endTime']
                }
            else:
                # Continue the current sequence
                current_sequence['regions'].append(row['region'])
                current_sequence['durations'].append(row['duration'])
                current_sequence['end_time'] = row['endTime']
        else:
            # End the current sequence if exists
            if current_sequence is not None:
                walking_sequences.append(current_sequence)
                current_sequence = None
    
    # Add the last sequence if exists
    if current_sequence is not None:
        walking_sequences.append(current_sequence)
    
    # Calculate sequence statistics
    for seq in walking_sequences:
        seq['total_duration'] = sum(seq['durations'])
        seq['region_count'] = len(seq['regions'])
        seq['unique_region_count'] = len(set(seq['regions']))
        seq['formatted_duration'] = format_seconds_to_hms(seq['total_duration'])
    
    # Calculate overall statistics
    if walking_sequences:
        sequence_durations = [seq['total_duration'] for seq in walking_sequences]
        region_counts = [seq['region_count'] for seq in walking_sequences]
        unique_region_counts = [seq['unique_region_count'] for seq in walking_sequences]
        
        results['overall'] = {
            'total_sequences': len(walking_sequences),
            'avg_duration': np.mean(sequence_durations),
            'median_duration': np.median(sequence_durations),
            'min_duration': min(sequence_durations),
            'max_duration': max(sequence_durations),
            'avg_region_count': np.mean(region_counts),
            'median_region_count': np.median(region_counts),
            'avg_unique_region_count': np.mean(unique_region_counts),
            'median_unique_region_count': np.median(unique_region_counts),
            'formatted_avg_duration': format_seconds_to_hms(np.mean(sequence_durations)),
            'formatted_median_duration': format_seconds_to_hms(np.median(sequence_durations)),
            'formatted_min_duration': format_seconds_to_hms(min(sequence_durations)),
            'formatted_max_duration': format_seconds_to_hms(max(sequence_durations))
        }
    
    # Calculate employee-specific statistics
    employee_stats = defaultdict(list)
    
    for seq in walking_sequences:
        employee_stats[seq['employee_id']].append(seq)
    
    for emp_id, sequences in employee_stats.items():
        sequence_durations = [seq['total_duration'] for seq in sequences]
        region_counts = [seq['region_count'] for seq in sequences]
        unique_region_counts = [seq['unique_region_count'] for seq in sequences]
        
        results['by_employee'][emp_id] = {
            'total_sequences': len(sequences),
            'avg_duration': np.mean(sequence_durations),
            'median_duration': np.median(sequence_durations),
            'min_duration': min(sequence_durations),
            'max_duration': max(sequence_durations),
            'avg_region_count': np.mean(region_counts),
            'median_region_count': np.median(region_counts),
            'avg_unique_region_count': np.mean(unique_region_counts),
            'median_unique_region_count': np.median(unique_region_counts),
            'formatted_avg_duration': format_seconds_to_hms(np.mean(sequence_durations)),
            'formatted_median_duration': format_seconds_to_hms(np.median(sequence_durations)),
            'formatted_min_duration': format_seconds_to_hms(min(sequence_durations)),
            'formatted_max_duration': format_seconds_to_hms(max(sequence_durations))
        }
    
    # Save statistics
    if stats_dir and walking_sequences:
        # Overall sequence statistics
        overall_df = pd.DataFrame([results['overall']])
        overall_df.to_csv(stats_dir / 'walking_sequence_statistics.csv', index=False)
        
        # Employee statistics
        emp_seq_data = []
        for emp_id, stats in results['by_employee'].items():
            emp_seq_data.append({
                'employee_id': emp_id,
                'total_sequences': stats['total_sequences'],
                'avg_duration': stats['avg_duration'],
                'median_duration': stats['median_duration'],
                'min_duration': stats['min_duration'],
                'max_duration': stats['max_duration'],
                'avg_region_count': stats['avg_region_count'],
                'median_region_count': stats['median_region_count'],
                'avg_unique_region_count': stats['avg_unique_region_count'],
                'median_unique_region_count': stats['median_unique_region_count'],
                'formatted_avg_duration': stats['formatted_avg_duration'],
                'formatted_median_duration': stats['formatted_median_duration']
            })
        
        if emp_seq_data:
            pd.DataFrame(emp_seq_data).to_csv(stats_dir / 'walking_sequence_by_employee.csv', index=False)
        
        # Top long walking sequences
        long_sequences = sorted(walking_sequences, key=lambda x: x['total_duration'], reverse=True)[:50]
        long_seq_data = []
        
        for i, seq in enumerate(long_sequences, 1):
            long_seq_data.append({
                'rank': i,
                'employee_id': seq['employee_id'],
                'start_time': seq['start_time'],
                'end_time': seq['end_time'],
                'total_duration': seq['total_duration'],
                'formatted_duration': seq['formatted_duration'],
                'region_count': seq['region_count'],
                'unique_region_count': seq['unique_region_count'],
                'regions_traversed': ' → '.join(seq['regions'])
            })
        
        if long_seq_data:
            pd.DataFrame(long_seq_data).to_csv(stats_dir / 'longest_walking_sequences.csv', index=False)
    
    # Create visualizations
    if vis_dir and walking_sequences:
        # Distribution of walking sequence durations
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create duration bins for analysis
        duration_bins = [0, 5, 10, 15, 30, 60, 120, 300, 600, float('inf')]
        bin_labels = ['0-5s', '5-10s', '10-15s', '15-30s', '30-60s', '1-2min', '2-5min', '5-10min', '>10min']
        
        sequence_durations = [seq['total_duration'] for seq in walking_sequences]
        duration_categories = pd.cut(sequence_durations, bins=duration_bins, labels=bin_labels)
        duration_dist = duration_categories.value_counts().sort_index()
        
        # Calculate percentages
        percentages = (duration_dist.values / duration_dist.sum() * 100).round(2)
        
        # Create bar chart
        bars = ax.bar(duration_dist.index, percentages, color=ACTIVITY_COLORS['Walk'])
        
        # Add data labels
        for bar, percentage, count in zip(bars, percentages, duration_dist.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.5,
                f"{percentage:.1f}%\n({count:,})",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Distribution of Walking Sequence Durations', language), fontsize=16)
        ax.set_xlabel(get_text('Duration Category', language), fontsize=14)
        ax.set_ylabel(get_text('Percentage (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, vis_dir / 'walking_sequence_durations.png')
        
        # Distribution of regions traversed per walking sequence
        fig, ax = plt.subplots(figsize=(14, 8))
        
        region_count_bins = [1, 2, 3, 4, 5, 10, float('inf')]
        region_count_labels = ['1', '2', '3', '4', '5-9', '10+']
        
        region_counts = [seq['region_count'] for seq in walking_sequences]
        region_count_categories = pd.cut(region_counts, bins=region_count_bins, labels=region_count_labels)
        region_count_dist = region_count_categories.value_counts().sort_index()
        
        # Calculate percentages
        percentages = (region_count_dist.values / region_count_dist.sum() * 100).round(2)
        
        # Create bar chart
        bars = ax.bar(region_count_dist.index, percentages, color=ACTIVITY_COLORS['Walk'])
        
        # Add data labels
        for bar, percentage, count in zip(bars, percentages, region_count_dist.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.5,
                f"{percentage:.1f}%\n({count:,})",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Number of Regions Traversed per Walking Sequence', language), fontsize=16)
        ax.set_xlabel(get_text('Number of Regions', language), fontsize=14)
        ax.set_ylabel(get_text('Percentage (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, vis_dir / 'regions_per_walking_sequence.png')
        
        # Walking sequence statistics by employee
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare employee data
        emp_data = pd.DataFrame([
            {
                'employee_id': emp_id,
                'avg_duration': stats['avg_duration'] / 60,  # Convert to minutes
                'total_sequences': stats['total_sequences']
            }
            for emp_id, stats in results['by_employee'].items()
        ])
        
        if not emp_data.empty:
            # Sort by average duration
            emp_data = emp_data.sort_values('avg_duration', ascending=False)
            
            # Create bar chart
            bars = ax.bar(
                emp_data['employee_id'],
                emp_data['avg_duration'],
                color=ACTIVITY_COLORS['Walk']
            )
            
            # Add data labels
            for bar, total_sequences in zip(bars, emp_data['total_sequences']):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.1,
                    f"{height:.1f} min\n({total_sequences:,} walks)",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.set_title(get_text('Average Walking Sequence Duration by Employee', language), fontsize=16)
            ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
            ax.set_ylabel(get_text('Average Duration (minutes)', language), fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'walking_sequence_by_employee.png')
    
    return results

def analyze_walking_by_time(walk_data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze walking patterns by time of day
    
    Parameters:
    -----------
    walk_data : pandas.DataFrame
        Input dataframe filtered for walking activities
    stats_dir : Path, optional
        Directory to save statistics
    vis_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with time-of-day analysis results
    """
    results = {}
    
    # Extract hour from startTimeClock (format: HH:MM:SS)
    if 'startTimeClock' in walk_data.columns:
        walk_data['hour'] = walk_data['startTimeClock'].str.split(':', expand=True)[0].astype(int)
        
        # Calculate walking time and count by hour
        hourly_stats = walk_data.groupby('hour').agg({
            'duration': ['sum', 'count', 'mean', 'median'],
            'id': 'nunique'  # Count unique employees
        })
        
        hourly_stats.columns = ['total_duration', 'count', 'mean_duration', 'median_duration', 'employee_count']
        hourly_stats = hourly_stats.reset_index()
        
        # Add formatted duration
        hourly_stats['formatted_total'] = hourly_stats['total_duration'].apply(format_seconds_to_hms)
        hourly_stats['formatted_mean'] = hourly_stats['mean_duration'].apply(format_seconds_to_hms)
        hourly_stats['formatted_median'] = hourly_stats['median_duration'].apply(format_seconds_to_hms)
        
        # Calculate percentage of total walking time
        total_walking = hourly_stats['total_duration'].sum()
        hourly_stats['percentage'] = (hourly_stats['total_duration'] / total_walking * 100).round(2)
        
        # Fill in missing hours with zeros
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_stats = pd.merge(all_hours, hourly_stats, on='hour', how='left').fillna(0)
        
        # Sort by hour
        hourly_stats = hourly_stats.sort_values('hour')
        
        results['hourly_stats'] = hourly_stats.to_dict('records')
        
        # Save statistics
        if stats_dir:
            hourly_stats.to_csv(stats_dir / 'walking_by_hour.csv', index=False)
        
        # Create visualizations
        if vis_dir:
            # Walking time by hour
            set_visualization_style()
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Convert to minutes for better visualization
            minutes = hourly_stats['total_duration'] / 60
            
            # Create bar chart
            bars = ax.bar(hourly_stats['hour'], minutes, color=ACTIVITY_COLORS['Walk'])
            
            # Add data labels
            for bar, formatted_time, percentage in zip(
                bars, hourly_stats['formatted_total'], hourly_stats['percentage']
            ):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height + 1,
                        f"{percentage:.1f}%\n{formatted_time}",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        rotation=0
                    )
            
            ax.set_title(get_text('Walking Time by Hour of Day', language), fontsize=16)
            ax.set_xlabel(get_text('Hour of Day', language), fontsize=14)
            ax.set_ylabel(get_text('Duration (minutes)', language), fontsize=14)
            ax.set_xticks(range(24))
            ax.set_xlim(-0.5, 23.5)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'walking_by_hour.png')
            
            # Walking count by hour
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create bar chart
            bars = ax.bar(hourly_stats['hour'], hourly_stats['count'], color=ACTIVITY_COLORS['Walk'])
            
            # Add data labels
            for bar, count, emp_count in zip(bars, hourly_stats['count'], hourly_stats['employee_count']):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height + 1,
                        f"{count:,.0f}\n({emp_count:.0f} emps)",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold'
                    )
            
            ax.set_title(get_text('Number of Walking Activities by Hour of Day', language), fontsize=16)
            ax.set_xlabel(get_text('Hour of Day', language), fontsize=14)
            ax.set_ylabel(get_text('Count', language), fontsize=14)
            ax.set_xticks(range(24))
            ax.set_xlim(-0.5, 23.5)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'walking_count_by_hour.png')
            
            # Average walking duration by hour
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Convert to seconds for better visualization
            mean_durations = hourly_stats['mean_duration']
            
            # Create bar chart
            bars = ax.bar(hourly_stats['hour'], mean_durations, color=ACTIVITY_COLORS['Walk'])
            
            # Add data labels
            for bar, formatted_mean in zip(bars, hourly_stats['formatted_mean']):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height + 0.1,
                        f"{formatted_mean}",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold'
                    )
            
            ax.set_title(get_text('Average Walking Duration by Hour of Day', language), fontsize=16)
            ax.set_xlabel(get_text('Hour of Day', language), fontsize=14)
            ax.set_ylabel(get_text('Average Duration (seconds)', language), fontsize=14)
            ax.set_xticks(range(24))
            ax.set_xlim(-0.5, 23.5)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'average_walking_duration_by_hour.png')
    else:
        print("  Time of day information not available in the dataset.")
    
    return results

def compare_walking_patterns(data, walk_data, stats_dir=None, vis_dir=None, language='en'):
    """
    Compare walking patterns between employees and departments
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    walk_data : pandas.DataFrame
        Input dataframe filtered for walking activities
    stats_dir : Path, optional
        Directory to save statistics
    vis_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with comparison results
    """
    results = {'by_employee': {}, 'by_department': {}}
    
    # Calculate walking metrics by employee
    employee_metrics = []
    
    for emp_id, emp_data in data.groupby('id'):
        total_time = emp_data['duration'].sum()
        
        # Get walking data for this employee
        emp_walk = walk_data[walk_data['id'] == emp_id]
        walking_time = emp_walk['duration'].sum() if not emp_walk.empty else 0
        walking_percentage = (walking_time / total_time * 100) if total_time > 0 else 0
        walking_count = len(emp_walk)
        
        # Get average walking duration
        avg_duration = emp_walk['duration'].mean() if not emp_walk.empty else 0
        
        # Get department information
        department = emp_data['department'].iloc[0] if 'department' in emp_data.columns else 'Unknown'
        
        # Store metrics
        metrics = {
            'employee_id': emp_id,
            'department': department,
            'total_time': total_time,
            'walking_time': walking_time,
            'walking_percentage': walking_percentage,
            'walking_count': walking_count,
            'avg_walking_duration': avg_duration,
            'formatted_total': format_seconds_to_hms(total_time),
            'formatted_walking': format_seconds_to_hms(walking_time),
            'formatted_avg_duration': format_seconds_to_hms(avg_duration)
        }
        
        employee_metrics.append(metrics)
        results['by_employee'][emp_id] = metrics
    
    # Calculate department metrics
    if 'department' in data.columns:
        department_metrics = []
        
        for dept, dept_data in data.groupby('department'):
            total_time = dept_data['duration'].sum()
            
            # Get walking data for this department
            dept_walk = walk_data[walk_data['id'].isin(dept_data['id'].unique())]
            walking_time = dept_walk['duration'].sum() if not dept_walk.empty else 0
            walking_percentage = (walking_time / total_time * 100) if total_time > 0 else 0
            walking_count = len(dept_walk)
            
            # Get average walking duration
            avg_duration = dept_walk['duration'].mean() if not dept_walk.empty else 0
            
            # Get employee count
            employee_count = dept_data['id'].nunique()
            
            # Store metrics
            metrics = {
                'department': dept,
                'total_time': total_time,
                'walking_time': walking_time,
                'walking_percentage': walking_percentage,
                'walking_count': walking_count,
                'avg_walking_duration': avg_duration,
                'employee_count': employee_count,
                'formatted_total': format_seconds_to_hms(total_time),
                'formatted_walking': format_seconds_to_hms(walking_time),
                'formatted_avg_duration': format_seconds_to_hms(avg_duration)
            }
            
            department_metrics.append(metrics)
            results['by_department'][dept] = metrics
    
    # Save statistics
    if stats_dir:
        # Employee metrics
        if employee_metrics:
            pd.DataFrame(employee_metrics).to_csv(stats_dir / 'walking_metrics_by_employee.csv', index=False)
        
        # Department metrics
        if 'department' in data.columns and 'by_department' in results:
            dept_data = [results['by_department'][dept] for dept in results['by_department']]
            if dept_data:
                pd.DataFrame(dept_data).to_csv(stats_dir / 'walking_metrics_by_department.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Walking percentage by employee
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if employee_metrics:
            # Convert to DataFrame and sort
            emp_df = pd.DataFrame(employee_metrics)
            emp_df = emp_df.sort_values('walking_percentage', ascending=False)
            
            # Create bar chart
            bars = ax.bar(
                emp_df['employee_id'],
                emp_df['walking_percentage'],
                color=[ACTIVITY_COLORS['Walk']] * len(emp_df)
            )
            
            # Add data labels
            for bar, walking_time, total_time in zip(
                bars, emp_df['formatted_walking'], emp_df['formatted_total']
            ):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 1,
                    f"{height:.1f}%\n{walking_time}",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.set_title(get_text('Walking Time Percentage by Employee', language), fontsize=16)
            ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
            ax.set_ylabel(get_text('Walking Time (%)', language), fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'walking_percentage_by_employee.png')
        
        # Walking percentage by department
        if 'department' in data.columns and 'by_department' in results:
            dept_data = [results['by_department'][dept] for dept in results['by_department']]
            if dept_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Convert to DataFrame and sort
                dept_df = pd.DataFrame(dept_data)
                dept_df = dept_df.sort_values('walking_percentage', ascending=False)
                
                # Create bar chart
                bars = ax.bar(
                    dept_df['department'],
                    dept_df['walking_percentage'],
                    color=[ACTIVITY_COLORS['Walk']] * len(dept_df)
                )
                
                # Add data labels
                for bar, walking_time, total_time, emp_count in zip(
                    bars, dept_df['formatted_walking'], dept_df['formatted_total'], dept_df['employee_count']
                ):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height + 1,
                        f"{height:.1f}%\n{walking_time}\n({emp_count} emps)",
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold'
                    )
                
                ax.set_title(get_text('Walking Time Percentage by Department', language), fontsize=16)
                ax.set_xlabel(get_text('Department', language), fontsize=14)
                ax.set_ylabel(get_text('Walking Time (%)', language), fontsize=14)
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                save_figure(fig, vis_dir / 'walking_percentage_by_department.png')
        
        # Average walking duration by employee
        if employee_metrics:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Sort by average duration
            emp_df = emp_df.sort_values('avg_walking_duration', ascending=False)
            
            # Create bar chart
            bars = ax.bar(
                emp_df['employee_id'],
                emp_df['avg_walking_duration'],
                color=[ACTIVITY_COLORS['Walk']] * len(emp_df)
            )
            
            # Add data labels
            for bar, avg_duration, walking_count in zip(
                bars, emp_df['formatted_avg_duration'], emp_df['walking_count']
            ):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.1,
                    f"{avg_duration}\n({walking_count:,} walks)",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.set_title(get_text('Average Walking Duration by Employee', language), fontsize=16)
            ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
            ax.set_ylabel(get_text('Average Duration (seconds)', language), fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, vis_dir / 'average_walking_duration_by_employee.png')
    
    return results