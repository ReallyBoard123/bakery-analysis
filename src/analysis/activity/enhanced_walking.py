"""
Enhanced Walking Analysis Module

Analyzes walking patterns of bakery employees with focus on:
1. Walks within same region vs region transitions
2. Employee-specific walking patterns
3. Walking sequences and paths
4. Time of day analysis
5. Purposeful vs transitory visits
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import networkx as nx
from datetime import datetime, timedelta

from src.utils.time_utils import format_seconds_to_hms
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import get_text, set_visualization_style, get_employee_colors

# Constants for analysis
TRANSITORY_VISIT_THRESHOLD = 60  # seconds - visits shorter than this are considered "passing through"
PURPOSEFUL_VISIT_THRESHOLD = 300  # seconds - visits longer than this are considered purposeful

def analyze_walking_patterns(data, output_dir=None, language='en', focus_employees=None):
    """
    Analyze walking patterns in the bakery dataset with detailed breakdowns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    focus_employees : list, optional
        List of specific employee IDs to focus on (e.g., ['32-D', '32-G'])
    
    Returns:
    --------
    dict
        Dictionary with detailed walking pattern analysis results
    """
    # Create subdirectories
    if output_dir:
        stats_dir = ensure_dir_exists(output_dir / 'statistics')
        vis_dir = ensure_dir_exists(output_dir / 'visualizations')
        employee_dir = ensure_dir_exists(output_dir / 'employee_analysis')
    
    # Filter for walking activities only
    walk_data = data[data['activity'] == 'Walk']
    
    if walk_data.empty:
        print("No walking data found in the dataset.")
        return {}
    
    # If focus employees specified, create subdirectories for them
    if focus_employees:
        for emp_id in focus_employees:
            if emp_id in data['id'].unique():
                ensure_dir_exists(employee_dir / emp_id)
    
    results = {}
    
    # 1. Analyze walks within regions vs between regions
    print("  Analyzing walks within regions vs between regions...")
    region_results = analyze_region_vs_transition_walks(data, stats_dir, vis_dir, language)
    results['region_analysis'] = region_results
    
    # 2. Analyze walking sequences and movement paths
    print("  Analyzing walking sequences and paths...")
    sequence_results = analyze_walking_sequences(data, stats_dir, vis_dir, language)
    results['sequences'] = sequence_results
    
    # 3. Analyze walking by time of day for each employee
    print("  Analyzing walking patterns by time of day...")
    # Check if startTimeClock column exists
    has_time_data = 'startTimeClock' in walk_data.columns
    if has_time_data:
        time_results = analyze_walking_by_time_of_day(walk_data, stats_dir, vis_dir, language)
        results['time_analysis'] = time_results
    else:
        print("  Time information not available in the dataset.")
        results['time_analysis'] = {}
    
    # 4. Analyze purposeful vs. transitory visits
    print("  Analyzing purposeful vs. transitory visits...")
    visit_results = analyze_visit_patterns(data, stats_dir, vis_dir, language)
    results['visit_patterns'] = visit_results
    
    # 5. Detailed employee-specific analysis (focus on requested employees)
    if focus_employees:
        employee_results = {}
        for emp_id in focus_employees:
            if emp_id in data['id'].unique():
                print(f"  Performing detailed analysis for employee {emp_id}...")
                emp_data = data[data['id'] == emp_id]
                emp_walk_data = walk_data[walk_data['id'] == emp_id]
                
                emp_results = analyze_employee_walking(
                    emp_id, emp_data, emp_walk_data, 
                    employee_dir / emp_id, stats_dir, 
                    vis_dir, language, 
                    has_time_data=has_time_data
                )
                employee_results[emp_id] = emp_results
        
        results['employee_analysis'] = employee_results
    
    print(f"Enhanced walking analysis complete. Results saved to {output_dir}")
    return results

def analyze_region_vs_transition_walks(data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze walks that occur within a region vs walks that involve region transitions
    """
    # Sort data by employee and time
    sorted_data = data.sort_values(['id', 'startTime'])
    
    # Track different types of walks
    within_region_walks = []
    transition_walks = []
    
    # Variables to track the current context
    current_employee = None
    current_activity = None
    current_region = None
    current_start_time = None
    region_transition_pending = False
    
    for _, row in sorted_data.iterrows():
        # If employee changes, reset tracking
        if current_employee != row['id']:
            current_employee = row['id']
            current_region = row['region']
            current_activity = row['activity']
            current_start_time = row['startTime']
            region_transition_pending = False
            continue
        
        # Capture the end of a walking activity
        if current_activity == 'Walk' and row['activity'] != 'Walk':
            # Check if region changed during this walk
            if region_transition_pending:
                walk_info = {
                    'employee_id': current_employee,
                    'start_time': current_start_time,
                    'end_time': row['startTime'],
                    'duration': row['startTime'] - current_start_time,
                    'from_region': current_region,
                    'to_region': row['region'],
                    'formatted_duration': format_seconds_to_hms(row['startTime'] - current_start_time)
                }
                transition_walks.append(walk_info)
                region_transition_pending = False
            else:
                walk_info = {
                    'employee_id': current_employee,
                    'region': current_region,
                    'start_time': current_start_time,
                    'end_time': row['startTime'],
                    'duration': row['startTime'] - current_start_time,
                    'formatted_duration': format_seconds_to_hms(row['startTime'] - current_start_time)
                }
                within_region_walks.append(walk_info)
        
        # Check for region transitions during walking
        if current_activity == 'Walk' and row['activity'] == 'Walk' and current_region != row['region']:
            region_transition_pending = True
        
        # Update tracking variables
        current_region = row['region']
        current_activity = row['activity']
        if row['activity'] == 'Walk' and current_start_time is None:
            current_start_time = row['startTime']
        elif row['activity'] != 'Walk':
            current_start_time = None
    
    # Calculate summary statistics
    results = {
        'within_region_walks': len(within_region_walks),
        'transition_walks': len(transition_walks),
        'total_walks': len(within_region_walks) + len(transition_walks),
        'within_region_pct': len(within_region_walks) / (len(within_region_walks) + len(transition_walks)) * 100 if (len(within_region_walks) + len(transition_walks)) > 0 else 0,
        'transition_pct': len(transition_walks) / (len(within_region_walks) + len(transition_walks)) * 100 if (len(within_region_walks) + len(transition_walks)) > 0 else 0,
    }
    
    # Calculate total duration for each type
    within_region_duration = sum(walk['duration'] for walk in within_region_walks)
    transition_duration = sum(walk['duration'] for walk in transition_walks)
    total_duration = within_region_duration + transition_duration
    
    results['within_region_duration'] = within_region_duration
    results['transition_duration'] = transition_duration
    results['total_walk_duration'] = total_duration
    results['within_region_duration_pct'] = (within_region_duration / total_duration * 100) if total_duration > 0 else 0
    results['transition_duration_pct'] = (transition_duration / total_duration * 100) if total_duration > 0 else 0
    results['within_region_duration_formatted'] = format_seconds_to_hms(within_region_duration)
    results['transition_duration_formatted'] = format_seconds_to_hms(transition_duration)
    results['total_duration_formatted'] = format_seconds_to_hms(total_duration)
    
    # Get employee-specific statistics
    employee_stats = {}
    for emp_id in data['id'].unique():
        emp_within = [w for w in within_region_walks if w['employee_id'] == emp_id]
        emp_transition = [w for w in transition_walks if w['employee_id'] == emp_id]
        
        within_duration = sum(w['duration'] for w in emp_within)
        transition_duration = sum(w['duration'] for w in emp_transition)
        total_duration = within_duration + transition_duration
        
        employee_stats[emp_id] = {
            'within_region_walks': len(emp_within),
            'transition_walks': len(emp_transition),
            'total_walks': len(emp_within) + len(emp_transition),
            'within_region_pct': len(emp_within) / (len(emp_within) + len(emp_transition)) * 100 if (len(emp_within) + len(emp_transition)) > 0 else 0,
            'transition_pct': len(emp_transition) / (len(emp_within) + len(emp_transition)) * 100 if (len(emp_within) + len(emp_transition)) > 0 else 0,
            'within_region_duration': within_duration,
            'transition_duration': transition_duration,
            'total_walk_duration': total_duration,
            'within_region_duration_pct': (within_duration / total_duration * 100) if total_duration > 0 else 0,
            'transition_duration_pct': (transition_duration / total_duration * 100) if total_duration > 0 else 0,
            'within_region_duration_formatted': format_seconds_to_hms(within_duration),
            'transition_duration_formatted': format_seconds_to_hms(transition_duration),
            'total_duration_formatted': format_seconds_to_hms(total_duration)
        }
    
    results['employee_stats'] = employee_stats
    
    # Find most common regions for within-region walks
    region_counts = Counter([w['region'] for w in within_region_walks])
    results['common_regions_for_walking'] = region_counts.most_common(10)
    
    # Find most common transition paths
    transition_paths = Counter([(w['from_region'], w['to_region']) for w in transition_walks])
    results['common_transition_paths'] = transition_paths.most_common(10)
    
    # Save statistics
    if stats_dir:
        # Overall summary
        summary_df = pd.DataFrame([{
            'walk_type': 'Within Region',
            'count': results['within_region_walks'],
            'percentage': results['within_region_pct'],
            'total_duration': results['within_region_duration'],
            'duration_percentage': results['within_region_duration_pct'],
            'formatted_duration': results['within_region_duration_formatted']
        }, {
            'walk_type': 'Region Transition',
            'count': results['transition_walks'],
            'percentage': results['transition_pct'],
            'total_duration': results['transition_duration'],
            'duration_percentage': results['transition_duration_pct'],
            'formatted_duration': results['transition_duration_formatted']
        }])
        summary_df.to_csv(stats_dir / 'walk_types_summary.csv', index=False)
        
        # Employee-specific statistics
        emp_df = pd.DataFrame([
            {
                'employee_id': emp_id,
                'within_region_walks': stats['within_region_walks'],
                'transition_walks': stats['transition_walks'],
                'total_walks': stats['total_walks'],
                'within_region_pct': stats['within_region_pct'],
                'transition_pct': stats['transition_pct'],
                'within_region_duration': stats['within_region_duration'],
                'transition_duration': stats['transition_duration'],
                'total_walk_duration': stats['total_walk_duration'],
                'within_duration_pct': stats['within_region_duration_pct'],
                'transition_duration_pct': stats['transition_duration_pct']
            }
            for emp_id, stats in employee_stats.items()
        ])
        emp_df.to_csv(stats_dir / 'employee_walk_types.csv', index=False)
        
        # Common regions
        if results['common_regions_for_walking']:
            region_df = pd.DataFrame([
                {'region': region, 'walk_count': count}
                for region, count in results['common_regions_for_walking']
            ])
            region_df.to_csv(stats_dir / 'common_walking_regions.csv', index=False)
        
        # Common transition paths
        if results['common_transition_paths']:
            path_df = pd.DataFrame([
                {'from_region': from_r, 'to_region': to_r, 'count': count}
                for (from_r, to_r), count in results['common_transition_paths']
            ])
            path_df.to_csv(stats_dir / 'common_transition_paths.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Pie chart of walk types
        set_visualization_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Chart 1: Walk counts
        ax1.pie(
            [results['within_region_walks'], results['transition_walks']],
            labels=['Within Region', 'Region Transitions'],
            autopct='%1.1f%%',
            colors=['#6495ED', '#FFA07A'],
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        ax1.set_title('Distribution of Walk Instances', fontsize=14)
        
        # Chart 2: Walk durations
        ax2.pie(
            [results['within_region_duration'], results['transition_duration']],
            labels=['Within Region', 'Region Transitions'],
            autopct='%1.1f%%',
            colors=['#6495ED', '#FFA07A'],
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        ax2.set_title('Distribution of Walking Time', fontsize=14)
        
        plt.suptitle('Walking Patterns: Within Regions vs. Region Transitions', fontsize=16)
        plt.tight_layout()
        plt.savefig(vis_dir / 'walk_types_distribution.png', dpi=300)
        plt.close()
        
        # Create employee comparison chart
        if employee_stats:
            # Extract data for visualization
            emp_ids = list(employee_stats.keys())
            within_pcts = [employee_stats[emp]['within_region_duration_pct'] for emp in emp_ids]
            transition_pcts = [employee_stats[emp]['transition_duration_pct'] for emp in emp_ids]
            
            # Sort employees by transition percentage for better visualization
            sorted_indices = np.argsort(transition_pcts)
            emp_ids = [emp_ids[i] for i in sorted_indices]
            within_pcts = [within_pcts[i] for i in sorted_indices]
            transition_pcts = [transition_pcts[i] for i in sorted_indices]
            
            # Create the stacked bar chart
            plt.figure(figsize=(12, 8))
            
            bar_width = 0.8
            ind = np.arange(len(emp_ids))
            
            # Create stacked bars
            plt.bar(ind, within_pcts, bar_width, label='Within Region', color='#6495ED')
            plt.bar(ind, transition_pcts, bar_width, bottom=within_pcts, label='Region Transitions', color='#FFA07A')
            
            # Add total walking time labels
            for i, emp_id in enumerate(emp_ids):
                total_duration = employee_stats[emp_id]['total_duration_formatted']
                plt.text(i, 102, f"{total_duration}", ha='center', va='bottom', fontsize=10, rotation=0)
            
            plt.xlabel('Employee ID', fontsize=12)
            plt.ylabel('Percentage of Total Walking Time (%)', fontsize=12)
            plt.title('Walking Pattern Comparison by Employee', fontsize=14)
            plt.xticks(ind, emp_ids)
            plt.yticks(np.arange(0, 101, 10))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
            
            plt.tight_layout()
            plt.savefig(vis_dir / 'employee_walking_patterns.png', dpi=300)
            plt.close()
            
            # Create transition path visualization
            if results['common_transition_paths']:
                # Create a graph visualization of common walking paths
                G = nx.DiGraph()
                
                # Add edges for the most common transitions
                for (from_r, to_r), count in results['common_transition_paths']:
                    if count >= 5:  # Only show transitions with significant count
                        G.add_edge(from_r, to_r, weight=count)
                
                if G.number_of_edges() > 0:
                    plt.figure(figsize=(14, 12))
                    
                    # Use spring layout for graph visualization
                    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
                    
                    # Draw the graph
                    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
                    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
                    nx.draw_networkx_labels(G, pos, font_size=10)
                    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', 
                                          arrowsize=15, connectionstyle='arc3,rad=0.1')
                    
                    # Add edge labels (transition counts)
                    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
                    
                    plt.title('Common Walking Transition Paths', fontsize=16)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(vis_dir / 'walking_transition_paths.png', dpi=300)
                    plt.close()
    
    return results

def analyze_walking_sequences(data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze walking sequences to identify complete movement paths
    """
    # Sort data by employee and time
    sorted_data = data.sort_values(['id', 'startTime'])
    
    # Track walking sequences
    walking_sequences = []
    current_sequence = None
    
    # Process data to identify continuous walking sequences
    for emp_id, emp_data in sorted_data.groupby('id'):
        emp_data = emp_data.sort_values('startTime')
        
        # Reset for new employee
        current_sequence = None
        
        for _, row in emp_data.iterrows():
            if row['activity'] == 'Walk':
                # Start or continue a sequence
                if current_sequence is None:
                    current_sequence = {
                        'employee_id': emp_id,
                        'regions': [row['region']],
                        'durations': [row['duration']],
                        'start_time': row['startTime'],
                        'end_time': row['endTime'],
                        'start_time_clock': row['startTimeClock'] if 'startTimeClock' in row else None,
                        'total_duration': row['duration']
                    }
                else:
                    # Check if this is continuous with the previous walk
                    time_difference = row['startTime'] - current_sequence['end_time']
                    
                    # If the gap is small (less than 5 seconds), consider it part of the same sequence
                    if time_difference < 5:
                        current_sequence['regions'].append(row['region'])
                        current_sequence['durations'].append(row['duration'])
                        current_sequence['end_time'] = row['endTime']
                        current_sequence['total_duration'] += row['duration']
                    else:
                        # Gap too large, finish the current sequence and start new one
                        current_sequence['formatted_duration'] = format_seconds_to_hms(current_sequence['total_duration'])
                        walking_sequences.append(current_sequence)
                        
                        current_sequence = {
                            'employee_id': emp_id,
                            'regions': [row['region']],
                            'durations': [row['duration']],
                            'start_time': row['startTime'],
                            'end_time': row['endTime'],
                            'start_time_clock': row['startTimeClock'] if 'startTimeClock' in row else None,
                            'total_duration': row['duration']
                        }
            else:
                # End current sequence if exists
                if current_sequence is not None:
                    current_sequence['formatted_duration'] = format_seconds_to_hms(current_sequence['total_duration'])
                    walking_sequences.append(current_sequence)
                    current_sequence = None
    
    # Ensure the last sequence is added
    if current_sequence is not None:
        current_sequence['formatted_duration'] = format_seconds_to_hms(current_sequence['total_duration'])
        walking_sequences.append(current_sequence)
    
    # Process walking sequences to extract paths and sequence data
    paths = []
    sequence_lengths = []
    sequence_durations = []
    
    for seq in walking_sequences:
        # Extract unique sequential regions (remove consecutive duplicates)
        regions = seq['regions']
        unique_path = [regions[0]]
        
        for region in regions[1:]:
            if region != unique_path[-1]:
                unique_path.append(region)
        
        # Only include paths with at least 2 unique regions
        if len(unique_path) >= 2:
            path_str = ' → '.join(unique_path)
            
            paths.append({
                'employee_id': seq['employee_id'],
                'path': path_str,
                'regions': unique_path,
                'region_count': len(unique_path),
                'duration': seq['total_duration'],
                'formatted_duration': seq['formatted_duration'],
                'start_time': seq['start_time'],
                'start_time_clock': seq['start_time_clock']
            })
        
        # Record sequence statistics
        sequence_lengths.append(len(seq['regions']))
        sequence_durations.append(seq['total_duration'])
    
    # Calculate statistics
    avg_sequence_length = np.mean(sequence_lengths) if sequence_lengths else 0
    avg_sequence_duration = np.mean(sequence_durations) if sequence_durations else 0
    
    # Count common paths
    path_counts = Counter([p['path'] for p in paths])
    common_paths = path_counts.most_common(20)
    
    # Calculate employee-specific statistics
    employee_stats = {}
    for emp_id in data['id'].unique():
        emp_sequences = [seq for seq in walking_sequences if seq['employee_id'] == emp_id]
        emp_paths = [p for p in paths if p['employee_id'] == emp_id]
        
        emp_seq_lengths = [len(seq['regions']) for seq in emp_sequences]
        emp_seq_durations = [seq['total_duration'] for seq in emp_sequences]
        
        avg_length = np.mean(emp_seq_lengths) if emp_seq_lengths else 0
        avg_duration = np.mean(emp_seq_durations) if emp_seq_durations else 0
        
        emp_path_counts = Counter([p['path'] for p in emp_paths])
        
        employee_stats[emp_id] = {
            'total_sequences': len(emp_sequences),
            'total_paths': len(emp_paths),
            'avg_sequence_length': avg_length,
            'avg_sequence_duration': avg_duration,
            'formatted_avg_duration': format_seconds_to_hms(avg_duration),
            'common_paths': emp_path_counts.most_common(5)
        }
    
    results = {
        'total_sequences': len(walking_sequences),
        'total_paths': len(paths),
        'unique_paths': len(path_counts),
        'avg_sequence_length': avg_sequence_length,
        'avg_sequence_duration': avg_sequence_duration,
        'formatted_avg_duration': format_seconds_to_hms(avg_sequence_duration),
        'common_paths': common_paths,
        'employee_stats': employee_stats,
        'sequence_data': walking_sequences,
        'path_data': paths
    }
    
    # Save statistics
    if stats_dir:
        # Save sequence summary
        summary_df = pd.DataFrame([{
            'total_sequences': results['total_sequences'],
            'total_paths': results['total_paths'],
            'unique_paths': results['unique_paths'],
            'avg_sequence_length': results['avg_sequence_length'],
            'avg_sequence_duration': results['avg_sequence_duration'],
            'formatted_avg_duration': results['formatted_avg_duration']
        }])
        summary_df.to_csv(stats_dir / 'walking_sequence_summary.csv', index=False)
        
        # Save common paths
        if common_paths:
            paths_df = pd.DataFrame([
                {'path': path, 'count': count, 'percentage': round(count / len(paths) * 100, 2) if paths else 0}
                for path, count in common_paths
            ])
            paths_df.to_csv(stats_dir / 'common_walking_paths.csv', index=False)
        
        # Save employee statistics
        emp_stats_df = pd.DataFrame([
            {
                'employee_id': emp_id,
                'total_sequences': stats['total_sequences'],
                'total_paths': stats['total_paths'],
                'avg_sequence_length': stats['avg_sequence_length'],
                'avg_sequence_duration': stats['avg_sequence_duration'],
                'formatted_avg_duration': stats['formatted_avg_duration']
            }
            for emp_id, stats in employee_stats.items()
        ])
        emp_stats_df.to_csv(stats_dir / 'employee_walking_sequences.csv', index=False)
        
        # Save full path data
        if paths:
            path_data_df = pd.DataFrame([
                {
                    'employee_id': p['employee_id'],
                    'path': p['path'],
                    'region_count': p['region_count'],
                    'duration': p['duration'],
                    'formatted_duration': p['formatted_duration'],
                    'start_time_clock': p['start_time_clock'] if p['start_time_clock'] else 'Unknown'
                }
                for p in paths
            ])
            path_data_df.to_csv(stats_dir / 'walking_paths_full.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Create a horizontal bar chart of most common paths
        if common_paths:
            plt.figure(figsize=(14, 10))
            
            # Use only top 15 paths at most for better visualization
            display_paths = common_paths[:15]
            
            # Create horizontal bars
            paths_labels = [path for path, _ in display_paths]
            counts = [count for _, count in display_paths]
            
            # Reverse lists to show most common at the top
            paths_labels.reverse()
            counts.reverse()
            
            bars = plt.barh(paths_labels, counts, color='#6495ED')
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f"{width}", ha='left', va='center')
            
            plt.title('Most Common Walking Paths', fontsize=16)
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Path', fontsize=12)
            plt.tight_layout()
            plt.savefig(vis_dir / 'common_walking_paths.png', dpi=300)
            plt.close()
        
        # Create histogram of walking sequence lengths
        plt.figure(figsize=(12, 8))
        plt.hist(sequence_lengths, bins=range(1, max(sequence_lengths) + 2) if sequence_lengths else range(1, 10), 
                alpha=0.7, color='#6495ED', edgecolor='black')
        plt.title('Distribution of Walking Sequence Lengths', fontsize=16)
        plt.xlabel('Number of Regions in Sequence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / 'walking_sequence_lengths.png', dpi=300)
        plt.close()
        
        # Create comparison of avg sequence length by employee
        plt.figure(figsize=(12, 8))
        
        # Get employee IDs and values
        emp_ids = list(employee_stats.keys())
        avg_lengths = [employee_stats[emp]['avg_sequence_length'] for emp in emp_ids]
        
        # Sort by average length
        sorted_indices = np.argsort(avg_lengths)[::-1]  # Descending
        emp_ids = [emp_ids[i] for i in sorted_indices]
        avg_lengths = [avg_lengths[i] for i in sorted_indices]
        
        # Create bars
        bars = plt.bar(emp_ids, avg_lengths, color='#6495ED')
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height:.1f}", ha='center', va='bottom')
        
        plt.title('Average Walking Sequence Length by Employee', fontsize=16)
        plt.xlabel('Employee ID', fontsize=12)
        plt.ylabel('Average Number of Regions', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / 'employee_avg_sequence_length.png', dpi=300)
        plt.close()
    
    return results

def analyze_walking_by_time_of_day(walk_data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze walking patterns by time of day for each employee
    """
    # Check if time information is available
    if 'startTimeClock' not in walk_data.columns:
        print("  Time information not available in the dataset.")
        return {}
    
    # Extract hour from startTimeClock
    walk_data['hour'] = walk_data['startTimeClock'].str.split(':', expand=True)[0].astype(int)
    
    # Calculate stats for all employees
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
    
    # Calculate employee-specific hourly stats
    employee_hourly_stats = {}
    
    for emp_id in walk_data['id'].unique():
        emp_walk_data = walk_data[walk_data['id'] == emp_id]
        
        # Calculate hourly stats for this employee
        emp_hourly = emp_walk_data.groupby('hour')['duration'].agg(['sum', 'count', 'mean']).reset_index()
        
        # Add formatted time
        emp_hourly['formatted_sum'] = emp_hourly['sum'].apply(format_seconds_to_hms)
        
        # Calculate percentage
        emp_total = emp_hourly['sum'].sum()
        emp_hourly['percentage'] = (emp_hourly['sum'] / emp_total * 100).round(2) if emp_total > 0 else 0
        
        # Fill in missing hours
        all_hours = pd.DataFrame({'hour': range(24)})
        emp_hourly = pd.merge(all_hours, emp_hourly, on='hour', how='left').fillna(0)
        
        employee_hourly_stats[emp_id] = emp_hourly.to_dict('records')
    
    # Organize results
    results = {
        'overall_hourly': hourly_stats.to_dict('records'),
        'employee_hourly': employee_hourly_stats
    }
    
    # Save statistics
    if stats_dir:
        # Save overall hourly stats
        hourly_stats.to_csv(stats_dir / 'walking_by_hour_overall.csv', index=False)
        
        # Save employee hourly stats
        for emp_id, hourly_data in employee_hourly_stats.items():
            emp_df = pd.DataFrame(hourly_data)
            emp_df.to_csv(stats_dir / f'walking_by_hour_{emp_id}.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Overall hourly walking chart
        plt.figure(figsize=(14, 8))
        
        # Convert to minutes for better visualization
        minutes = hourly_stats['total_duration'] / 60
        
        plt.bar(hourly_stats['hour'], minutes, color='#6495ED')
        
        plt.title('Walking Time by Hour of Day (All Employees)', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Duration (minutes)', fontsize=12)
        plt.xticks(range(24))
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'walking_by_hour_overall.png', dpi=300)
        plt.close()
        
        # Create a heatmap of walking time by hour and employee
        # First, prepare the data
        heatmap_data = []
        
        for emp_id in employee_hourly_stats.keys():
            for hour_data in employee_hourly_stats[emp_id]:
                heatmap_data.append({
                    'employee_id': emp_id,
                    'hour': hour_data['hour'],
                    'duration': hour_data['sum'] / 60  # Convert to minutes
                })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Create pivot table for heatmap
            pivot_data = heatmap_df.pivot(index='employee_id', columns='hour', values='duration')
            
            # Fill NaN with 0
            pivot_data = pivot_data.fillna(0)
            
            # Create heatmap
            plt.figure(figsize=(16, 8))
            sns.heatmap(pivot_data, cmap='YlGnBu', annot=False, fmt='.1f', linewidths=0.5)
            
            plt.title('Walking Time by Hour and Employee (minutes)', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Employee ID', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(vis_dir / 'walking_heatmap_by_hour_employee.png', dpi=300)
            plt.close()
        
        # Create individual hourly charts for each employee
        for emp_id, hourly_data in employee_hourly_stats.items():
            plt.figure(figsize=(12, 6))
            
            # Convert to minutes
            emp_df = pd.DataFrame(hourly_data)
            minutes = emp_df['sum'] / 60
            
            plt.bar(emp_df['hour'], minutes, color='#6495ED')
            
            plt.title(f'Walking Time by Hour of Day - Employee {emp_id}', fontsize=14)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Duration (minutes)', fontsize=12)
            plt.xticks(range(24))
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(vis_dir / f'walking_by_hour_{emp_id}.png', dpi=300)
            plt.close()
    
    return results

def analyze_visit_patterns(data, stats_dir=None, vis_dir=None, language='en'):
    """
    Analyze region visit patterns to distinguish between transitory and purposeful visits
    """
    # Sort data by employee and time
    sorted_data = data.sort_values(['id', 'startTime'])
    
    # Track visits to regions
    visits = []
    
    # Variables to track the current visit
    current_employee = None
    current_region = None
    visit_start_time = None
    visit_activities = []
    prev_row = None
    
    for _, row in sorted_data.iterrows():
        # If employee changes, reset tracking
        if current_employee != row['id']:
            # Save the last visit if exists
            if current_employee and current_region and visit_start_time and prev_row is not None:
                visit_end_time = prev_row['endTime']
                visit_duration = visit_end_time - visit_start_time
                
                visits.append({
                    'employee_id': current_employee,
                    'region': current_region,
                    'start_time': visit_start_time,
                    'end_time': visit_end_time,
                    'duration': visit_duration,
                    'formatted_duration': format_seconds_to_hms(visit_duration),
                    'activities': visit_activities
                })
            
            # Reset for new employee
            current_employee = row['id']
            current_region = row['region']
            visit_start_time = row['startTime']
            visit_activities = [row['activity']]
            prev_row = row
            continue
        
        # Check if region changed
        if row['region'] != current_region:
            # Save the previous visit
            if prev_row is not None:
                visit_end_time = prev_row['endTime']
                visit_duration = visit_end_time - visit_start_time
                
                visits.append({
                    'employee_id': current_employee,
                    'region': current_region,
                    'start_time': visit_start_time,
                    'end_time': visit_end_time,
                    'duration': visit_duration,
                    'formatted_duration': format_seconds_to_hms(visit_duration),
                    'activities': visit_activities
                })
            
            # Start a new visit
            current_region = row['region']
            visit_start_time = row['startTime']
            visit_activities = [row['activity']]
        else:
            # Continue the current visit
            if row['activity'] not in visit_activities:
                visit_activities.append(row['activity'])
        
        # Update previous row
        prev_row = row
    
    # Add the final visit
    if current_employee and current_region and visit_start_time and prev_row is not None:
        visit_end_time = prev_row['endTime']
        visit_duration = visit_end_time - visit_start_time
        
        visits.append({
            'employee_id': current_employee,
            'region': current_region,
            'start_time': visit_start_time,
            'end_time': visit_end_time,
            'duration': visit_duration,
            'formatted_duration': format_seconds_to_hms(visit_duration),
            'activities': visit_activities
        })
    
    # Classify visits as transitory or purposeful
    for visit in visits:
        if visit['duration'] < TRANSITORY_VISIT_THRESHOLD:
            visit['visit_type'] = 'Transitory'
        elif visit['duration'] >= PURPOSEFUL_VISIT_THRESHOLD:
            visit['visit_type'] = 'Purposeful'
        else:
            visit['visit_type'] = 'Brief'
        
        # Check if visit was just walking
        if visit['activities'] == ['Walk']:
            visit['activity_type'] = 'Walking only'
        elif 'Walk' in visit['activities']:
            visit['activity_type'] = 'Mixed with walking'
        else:
            visit['activity_type'] = 'No walking'
    
    # Calculate statistics
    transitory_visits = [v for v in visits if v['visit_type'] == 'Transitory']
    brief_visits = [v for v in visits if v['visit_type'] == 'Brief']
    purposeful_visits = [v for v in visits if v['visit_type'] == 'Purposeful']
    
    # Calculate statistics by visit type
    total_visits = len(visits)
    
    results = {
        'total_visits': total_visits,
        'transitory_visits': len(transitory_visits),
        'brief_visits': len(brief_visits),
        'purposeful_visits': len(purposeful_visits),
        'transitory_pct': (len(transitory_visits) / total_visits * 100) if total_visits > 0 else 0,
        'brief_pct': (len(brief_visits) / total_visits * 100) if total_visits > 0 else 0,
        'purposeful_pct': (len(purposeful_visits) / total_visits * 100) if total_visits > 0 else 0,
        'visits': visits
    }
    
    # Calculate employee-specific statistics
    employee_stats = {}
    
    for emp_id in data['id'].unique():
        emp_visits = [v for v in visits if v['employee_id'] == emp_id]
        emp_transitory = [v for v in emp_visits if v['visit_type'] == 'Transitory']
        emp_brief = [v for v in emp_visits if v['visit_type'] == 'Brief']
        emp_purposeful = [v for v in emp_visits if v['visit_type'] == 'Purposeful']
        
        emp_total = len(emp_visits)
        
        employee_stats[emp_id] = {
            'total_visits': emp_total,
            'transitory_visits': len(emp_transitory),
            'brief_visits': len(emp_brief),
            'purposeful_visits': len(emp_purposeful),
            'transitory_pct': (len(emp_transitory) / emp_total * 100) if emp_total > 0 else 0,
            'brief_pct': (len(emp_brief) / emp_total * 100) if emp_total > 0 else 0,
            'purposeful_pct': (len(emp_purposeful) / emp_total * 100) if emp_total > 0 else 0
        }
    
    results['employee_stats'] = employee_stats
    
    # Find most common transitory paths (region pairs)
    transitory_paths = []
    
    for i in range(len(visits) - 1):
        visit1 = visits[i]
        visit2 = visits[i + 1]
        
        # Check if visits are by the same employee and sequential
        if (visit1['employee_id'] == visit2['employee_id'] and 
            visit1['end_time'] == visit2['start_time']):
            
            # Check if at least one visit is transitory
            if visit1['visit_type'] == 'Transitory' or visit2['visit_type'] == 'Transitory':
                transitory_paths.append((visit1['region'], visit2['region']))
    
    # Count common transitory paths
    common_transitory_paths = Counter(transitory_paths).most_common(10)
    results['common_transitory_paths'] = common_transitory_paths
    
    # Save statistics
    if stats_dir:
        # Overall summary
        summary_df = pd.DataFrame([{
            'visit_type': 'Transitory (<1 min)',
            'count': results['transitory_visits'],
            'percentage': results['transitory_pct']
        }, {
            'visit_type': 'Brief (1-5 min)',
            'count': results['brief_visits'],
            'percentage': results['brief_pct']
        }, {
            'visit_type': 'Purposeful (>5 min)',
            'count': results['purposeful_visits'],
            'percentage': results['purposeful_pct']
        }])
        summary_df.to_csv(stats_dir / 'visit_types_summary.csv', index=False)
        
        # Employee-specific summary
        emp_summary_df = pd.DataFrame([
            {
                'employee_id': emp_id,
                'total_visits': stats['total_visits'],
                'transitory_visits': stats['transitory_visits'],
                'brief_visits': stats['brief_visits'],
                'purposeful_visits': stats['purposeful_visits'],
                'transitory_pct': stats['transitory_pct'],
                'brief_pct': stats['brief_pct'],
                'purposeful_pct': stats['purposeful_pct']
            }
            for emp_id, stats in employee_stats.items()
        ])
        emp_summary_df.to_csv(stats_dir / 'employee_visit_types.csv', index=False)
        
        # Save full visit data
        visit_df = pd.DataFrame([
            {
                'employee_id': v['employee_id'],
                'region': v['region'],
                'duration': v['duration'],
                'formatted_duration': v['formatted_duration'],
                'visit_type': v['visit_type'],
                'activity_type': v['activity_type'],
                'activities': ','.join(v['activities'])
            }
            for v in visits
        ])
        visit_df.to_csv(stats_dir / 'region_visits_full.csv', index=False)
        
        # Save common transitory paths
        if common_transitory_paths:
            path_df = pd.DataFrame([
                {'from_region': from_r, 'to_region': to_r, 'count': count}
                for (from_r, to_r), count in common_transitory_paths
            ])
            path_df.to_csv(stats_dir / 'common_transitory_paths.csv', index=False)
    
    # Create visualizations
    if vis_dir:
        # Pie chart of visit types
        plt.figure(figsize=(10, 8))
        
        plt.pie(
            [results['transitory_visits'], results['brief_visits'], results['purposeful_visits']],
            labels=['Transitory (<1 min)', 'Brief (1-5 min)', 'Purposeful (>5 min)'],
            autopct='%1.1f%%',
            colors=['#FFA07A', '#FFD700', '#90EE90'],
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        plt.title('Distribution of Region Visit Types', fontsize=16)
        plt.tight_layout()
        plt.savefig(vis_dir / 'visit_types_distribution.png', dpi=300)
        plt.close()
        
        # Employee comparison chart
        plt.figure(figsize=(14, 8))
        
        # Extract data for visualization
        emp_ids = list(employee_stats.keys())
        transitory_pcts = [employee_stats[emp]['transitory_pct'] for emp in emp_ids]
        brief_pcts = [employee_stats[emp]['brief_pct'] for emp in emp_ids]
        purposeful_pcts = [employee_stats[emp]['purposeful_pct'] for emp in emp_ids]
        
        # Sort employees for better visualization
        sorted_indices = np.argsort(purposeful_pcts)
        emp_ids = [emp_ids[i] for i in sorted_indices]
        transitory_pcts = [transitory_pcts[i] for i in sorted_indices]
        brief_pcts = [brief_pcts[i] for i in sorted_indices]
        purposeful_pcts = [purposeful_pcts[i] for i in sorted_indices]
        
        # Create stacked bar chart
        ind = np.arange(len(emp_ids))
        width = 0.8
        
        plt.bar(ind, transitory_pcts, width, label='Transitory (<1 min)', color='#FFA07A')
        plt.bar(ind, brief_pcts, width, bottom=transitory_pcts, label='Brief (1-5 min)', color='#FFD700')
        plt.bar(ind, purposeful_pcts, width, bottom=[sum(x) for x in zip(transitory_pcts, brief_pcts)],
               label='Purposeful (>5 min)', color='#90EE90')
        
        plt.xlabel('Employee ID', fontsize=12)
        plt.ylabel('Percentage of Visits (%)', fontsize=12)
        plt.title('Visit Type Distribution by Employee', fontsize=16)
        plt.xticks(ind, emp_ids)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)
        
        # Add visit count labels
        for i, emp_id in enumerate(emp_ids):
            plt.text(i, 102, f"n={employee_stats[emp_id]['total_visits']}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'employee_visit_types.png', dpi=300)
        plt.close()
    
    return results

def analyze_employee_walking(emp_id, emp_data, emp_walk_data, output_dir, stats_dir, vis_dir, language='en', has_time_data=True):
    """
    Perform a detailed walking analysis for a specific employee
    
    Parameters:
    -----------
    emp_id : str
        Employee ID to analyze
    emp_data : pandas.DataFrame
        Filtered dataframe for this employee
    emp_walk_data : pandas.DataFrame
        Filtered walking data for this employee
    output_dir : Path
        Output directory for employee-specific analysis
    stats_dir : Path
        Directory for saving statistics
    vis_dir : Path
        Directory for saving visualizations
    language : str
        Language code ('en' or 'de')
    has_time_data : bool, optional
        Whether time data (startTimeClock) is available
    
    Returns:
    --------
    dict
        Dictionary with detailed employee walking analysis
    """
    results = {}
    
    # 1. Basic walking statistics
    total_time = emp_data['duration'].sum()
    walking_time = emp_walk_data['duration'].sum()
    walking_percentage = (walking_time / total_time * 100) if total_time > 0 else 0
    
    results['basic_stats'] = {
        'employee_id': emp_id,
        'total_time': total_time,
        'walking_time': walking_time,
        'walking_percentage': walking_percentage,
        'total_time_formatted': format_seconds_to_hms(total_time),
        'walking_time_formatted': format_seconds_to_hms(walking_time),
        'walk_count': len(emp_walk_data)
    }
    
    # 2. Analyze walks within regions vs between regions
    within_region_walks = []
    transition_walks = []
    
    # Sort by time
    sorted_data = emp_data.sort_values('startTime')
    
    # Variables to track the current context
    current_activity = None
    current_region = None
    current_start_time = None
    region_transition_pending = False
    
    for _, row in sorted_data.iterrows():
        # Capture the end of a walking activity
        if current_activity == 'Walk' and row['activity'] != 'Walk':
            # Check if region changed during this walk
            if region_transition_pending:
                walk_info = {
                    'start_time': current_start_time,
                    'end_time': row['startTime'],
                    'duration': row['startTime'] - current_start_time,
                    'from_region': current_region,
                    'to_region': row['region'],
                    'formatted_duration': format_seconds_to_hms(row['startTime'] - current_start_time)
                }
                transition_walks.append(walk_info)
                region_transition_pending = False
            else:
                walk_info = {
                    'region': current_region,
                    'start_time': current_start_time,
                    'end_time': row['startTime'],
                    'duration': row['startTime'] - current_start_time,
                    'formatted_duration': format_seconds_to_hms(row['startTime'] - current_start_time)
                }
                within_region_walks.append(walk_info)
        
        # Check for region transitions during walking
        if current_activity == 'Walk' and row['activity'] == 'Walk' and current_region != row['region']:
            region_transition_pending = True
        
        # Update tracking variables
        current_region = row['region']
        current_activity = row['activity']
        if row['activity'] == 'Walk' and current_start_time is None:
            current_start_time = row['startTime']
        elif row['activity'] != 'Walk':
            current_start_time = None
    
    # Calculate region walk stats
    within_region_duration = sum(walk['duration'] for walk in within_region_walks)
    transition_duration = sum(walk['duration'] for walk in transition_walks)
    
    results['walk_types'] = {
        'within_region_walks': len(within_region_walks),
        'transition_walks': len(transition_walks),
        'within_region_duration': within_region_duration,
        'transition_duration': transition_duration,
        'within_region_pct': (len(within_region_walks) / (len(within_region_walks) + len(transition_walks)) * 100) 
                            if (len(within_region_walks) + len(transition_walks)) > 0 else 0,
        'transition_pct': (len(transition_walks) / (len(within_region_walks) + len(transition_walks)) * 100)
                         if (len(within_region_walks) + len(transition_walks)) > 0 else 0,
        'within_duration_pct': (within_region_duration / (within_region_duration + transition_duration) * 100)
                              if (within_region_duration + transition_duration) > 0 else 0,
        'transition_duration_pct': (transition_duration / (within_region_duration + transition_duration) * 100)
                                 if (within_region_duration + transition_duration) > 0 else 0,
        'within_region_formatted': format_seconds_to_hms(within_region_duration),
        'transition_formatted': format_seconds_to_hms(transition_duration)
    }
    
    # Find regions where employee spends most time walking
    walking_by_region = emp_walk_data.groupby('region')['duration'].sum().reset_index()
    walking_by_region['percentage'] = (walking_by_region['duration'] / walking_time * 100) if walking_time > 0 else 0
    walking_by_region['formatted_time'] = walking_by_region['duration'].apply(format_seconds_to_hms)
    walking_by_region = walking_by_region.sort_values('duration', ascending=False)
    
    results['walking_by_region'] = walking_by_region.to_dict('records')
    
    # Find most common transition paths
    if transition_walks:
        transition_paths = Counter([(w['from_region'], w['to_region']) for w in transition_walks])
        results['common_transition_paths'] = transition_paths.most_common(10)
    else:
        results['common_transition_paths'] = []
    
    # 3. Analyze walking by time of day if time data is available
    if has_time_data and 'startTimeClock' in emp_walk_data.columns:
        emp_walk_data['hour'] = emp_walk_data['startTimeClock'].str.split(':', expand=True)[0].astype(int)
        
        # Calculate hourly stats
        hourly_stats = emp_walk_data.groupby('hour')['duration'].agg(['sum', 'count', 'mean']).reset_index()
        hourly_stats['formatted_sum'] = hourly_stats['sum'].apply(format_seconds_to_hms)
        
        # Fill in missing hours
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_stats = pd.merge(all_hours, hourly_stats, on='hour', how='left').fillna(0)
        
        results['walking_by_hour'] = hourly_stats.to_dict('records')
        
        # Find peak walking hours
        if not hourly_stats.empty:
            peak_hours = hourly_stats.sort_values('sum', ascending=False).head(3)
            results['peak_walking_hours'] = peak_hours.to_dict('records')
        
        # Identify if employee has more walking at specific times
        hourly_pct = hourly_stats['sum'] / hourly_stats['sum'].sum() * 100 if hourly_stats['sum'].sum() > 0 else 0
        results['hourly_distribution'] = hourly_pct.tolist()
        
        # Check for time periods with high walking concentration
        morning_hours = range(5, 12)
        afternoon_hours = range(12, 18)
        evening_hours = list(range(18, 24)) + list(range(0, 5))
        
        morning_walking = hourly_stats[hourly_stats['hour'].isin(morning_hours)]['sum'].sum()
        afternoon_walking = hourly_stats[hourly_stats['hour'].isin(afternoon_hours)]['sum'].sum()
        evening_walking = hourly_stats[hourly_stats['hour'].isin(evening_hours)]['sum'].sum()
        
        results['time_period_walking'] = {
            'morning': morning_walking,
            'afternoon': afternoon_walking,
            'evening': evening_walking,
            'morning_pct': (morning_walking / walking_time * 100) if walking_time > 0 else 0,
            'afternoon_pct': (afternoon_walking / walking_time * 100) if walking_time > 0 else 0,
            'evening_pct': (evening_walking / walking_time * 100) if walking_time > 0 else 0,
            'morning_formatted': format_seconds_to_hms(morning_walking),
            'afternoon_formatted': format_seconds_to_hms(afternoon_walking),
            'evening_formatted': format_seconds_to_hms(evening_walking)
        }
    
    # 4. Analyze walking sequences and patterns
    # Track walking sequences
    walking_sequences = []
    current_sequence = None
    
    # Extract walking sequences
    for _, row in sorted_data.iterrows():
        if row['activity'] == 'Walk':
            # Start or continue a sequence
            if current_sequence is None:
                current_sequence = {
                    'regions': [row['region']],
                    'durations': [row['duration']],
                    'start_time': row['startTime'],
                    'end_time': row['endTime'],
                    'start_time_clock': row['startTimeClock'] if 'startTimeClock' in row else None,
                    'total_duration': row['duration']
                }
            else:
                # Check if this is continuous with the previous walk
                time_difference = row['startTime'] - current_sequence['end_time']
                
                # If the gap is small (less than 5 seconds), consider it part of the same sequence
                if time_difference < 5:
                    current_sequence['regions'].append(row['region'])
                    current_sequence['durations'].append(row['duration'])
                    current_sequence['end_time'] = row['endTime']
                    current_sequence['total_duration'] += row['duration']
                else:
                    # Gap too large, finish the current sequence and start new one
                    current_sequence['formatted_duration'] = format_seconds_to_hms(current_sequence['total_duration'])
                    walking_sequences.append(current_sequence)
                    
                    current_sequence = {
                        'regions': [row['region']],
                        'durations': [row['duration']],
                        'start_time': row['startTime'],
                        'end_time': row['endTime'],
                        'start_time_clock': row['startTimeClock'] if 'startTimeClock' in row else None,
                        'total_duration': row['duration']
                    }
        else:
            # End current sequence if exists
            if current_sequence is not None:
                current_sequence['formatted_duration'] = format_seconds_to_hms(current_sequence['total_duration'])
                walking_sequences.append(current_sequence)
                current_sequence = None
    
    # Ensure the last sequence is added
    if current_sequence is not None:
        current_sequence['formatted_duration'] = format_seconds_to_hms(current_sequence['total_duration'])
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
                    'path': path_str,
                    'regions': unique_path,
                    'region_count': len(unique_path),
                    'duration': total_duration,
                    'formatted_duration': format_seconds_to_hms(total_duration),
                    'start_time': seq['start_time'],
                    'start_time_clock': seq['start_time_clock']
                })
    
    # Calculate sequence statistics
    results['walking_sequences'] = {
        'total_sequences': len(walking_sequences),
        'avg_sequence_length': np.mean([len(seq['regions']) for seq in walking_sequences]) if walking_sequences else 0,
        'avg_sequence_duration': np.mean([seq['total_duration'] for seq in walking_sequences]) if walking_sequences else 0,
        'formatted_avg_duration': format_seconds_to_hms(np.mean([seq['total_duration'] for seq in walking_sequences]) if walking_sequences else 0),
        'max_sequence_length': max([len(seq['regions']) for seq in walking_sequences]) if walking_sequences else 0,
        'sequences': walking_sequences[:10]  # Include only first 10 for brevity
    }
    
    # Count common paths
    if paths:
        path_counts = Counter([p['path'] for p in paths])
        results['common_paths'] = path_counts.most_common(10)
    else:
        results['common_paths'] = []
    
    # Save statistics for this employee
    if output_dir:
        # Basic stats
        pd.DataFrame([results['basic_stats']]).to_csv(output_dir / 'basic_walking_stats.csv', index=False)
        
        # Walk types
        pd.DataFrame([results['walk_types']]).to_csv(output_dir / 'walk_types.csv', index=False)
        
        # Walking by region
        if results['walking_by_region']:
            pd.DataFrame(results['walking_by_region']).to_csv(output_dir / 'walking_by_region.csv', index=False)
        
        # Common transition paths
        if results['common_transition_paths']:
            pd.DataFrame([
                {'from_region': from_r, 'to_region': to_r, 'count': count}
                for (from_r, to_r), count in results['common_transition_paths']
            ]).to_csv(output_dir / 'common_transition_paths.csv', index=False)
        
        # Walking by hour (if available)
        if 'walking_by_hour' in results:
            pd.DataFrame(results['walking_by_hour']).to_csv(output_dir / 'walking_by_hour.csv', index=False)
        
        # Common walking paths
        if results['common_paths']:
            pd.DataFrame([
                {'path': path, 'count': count}
                for path, count in results['common_paths']
            ]).to_csv(output_dir / 'common_walking_paths.csv', index=False)
    
    # Create visualizations
    if output_dir:
        # Create a pie chart of walk types
        plt.figure(figsize=(10, 8))
        
        plt.pie(
            [results['walk_types']['within_region_duration'], results['walk_types']['transition_duration']],
            labels=['Within Region', 'Region Transitions'],
            autopct='%1.1f%%',
            colors=['#6495ED', '#FFA07A'],
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        plt.title(f'Walking Types - Employee {emp_id}', fontsize=16)
        plt.figtext(0.5, 0.01, f"Total walking time: {results['basic_stats']['walking_time_formatted']}", 
                   ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'walk_types_pie.png', dpi=300)
        plt.close()
        
        # Create a bar chart of walking by region
        if results['walking_by_region']:
            # Get top regions for visualization
            top_regions = pd.DataFrame(results['walking_by_region']).head(10)
            
            plt.figure(figsize=(12, 8))
            
            bars = plt.bar(top_regions['region'], top_regions['duration'] / 60, color='#6495ED')  # Convert to minutes
            
            # Add duration labels
            for bar, time in zip(bars, top_regions['formatted_time']):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        time, ha='center', va='bottom', fontsize=10)
            
            plt.title(f'Top Regions by Walking Time - Employee {emp_id}', fontsize=16)
            plt.xlabel('Region', fontsize=12)
            plt.ylabel('Duration (minutes)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'walking_by_region.png', dpi=300)
            plt.close()
        
        # Create hourly walking chart if time data is available
        if 'walking_by_hour' in results:
            plt.figure(figsize=(12, 6))
            
            # Get hourly data
            hourly_data = pd.DataFrame(results['walking_by_hour'])
            
            # Convert to minutes for better visualization
            bars = plt.bar(hourly_data['hour'], hourly_data['sum'] / 60, color='#6495ED')
            
            plt.title(f'Walking Time by Hour of Day - Employee {emp_id}', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Duration (minutes)', fontsize=12)
            plt.xticks(range(24))
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'walking_by_hour.png', dpi=300)
            plt.close()
            
            # Create time period pie chart if available
            if 'time_period_walking' in results:
                plt.figure(figsize=(10, 8))
                
                time_data = results['time_period_walking']
                
                plt.pie(
                    [time_data['morning'], time_data['afternoon'], time_data['evening']],
                    labels=['Morning (5-12)', 'Afternoon (12-18)', 'Evening/Night (18-5)'],
                    autopct='%1.1f%%',
                    colors=['#FFD700', '#6495ED', '#8A2BE2'],
                    startangle=90,
                    wedgeprops={'edgecolor': 'w', 'linewidth': 1}
                )
                
                plt.title(f'Walking Time by Period of Day - Employee {emp_id}', fontsize=16)
                plt.tight_layout()
                plt.savefig(output_dir / 'walking_by_time_period.png', dpi=300)
                plt.close()
        
        # Create common paths visualization
        if results['common_paths']:
            plt.figure(figsize=(14, 10))
            
            # Use only the top 8 paths for better visualization
            display_paths = results['common_paths'][:8]
            
            # Extract labels and counts
            paths_labels = [path for path, _ in display_paths]
            counts = [count for _, count in display_paths]
            
            # Reverse lists to show most common at the top
            paths_labels.reverse()
            counts.reverse()
            
            bars = plt.barh(paths_labels, counts, color='#6495ED')
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                        f"{width}", ha='left', va='center')
            
            plt.title(f'Most Common Walking Paths - Employee {emp_id}', fontsize=16)
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Path', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / 'common_walking_paths.png', dpi=300)
            plt.close()
    
    return results