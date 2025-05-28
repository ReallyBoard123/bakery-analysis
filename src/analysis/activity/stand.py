"""
Standing Analysis Module

Analyzes standing patterns of bakery employees to identify bottlenecks and optimization opportunities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

from src.utils.time_utils import format_seconds_to_hms, format_seconds_to_minutes
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import (save_figure, set_visualization_style, 
                                 get_employee_colors, get_department_colors,
                                 get_text)

def analyze_standing_patterns(data, output_dir=None, language='en'):
    """
    Analyze standing patterns in the bakery dataset
    
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
        Dictionary with standing pattern analysis results
    """
    # Create subdirectories
    if output_dir:
        stats_dir = ensure_dir_exists(output_dir / 'statistics')
        vis_dir = ensure_dir_exists(output_dir / 'visualizations')
    
    # Filter for standing activities only
    stand_data = data[data['activity'] == 'Stand']
    
    if stand_data.empty:
        print("No standing data found in the dataset.")
        return {}
    
    results = {}
    
    # 1. Standing Duration Analysis
    print("  Analyzing standing durations...")
    duration_results = analyze_standing_durations(stand_data, stats_dir, language)
    results['durations'] = duration_results
    
    # 2. Standing by Region
    print("  Analyzing standing time by region...")
    region_results = analyze_standing_by_region(stand_data, vis_dir, language)
    results['by_region'] = region_results
    
    # 3. Standing by Employee
    print("  Analyzing standing time by employee...")
    employee_results = analyze_standing_by_employee(data, stand_data, vis_dir, language)
    results['by_employee'] = employee_results
    
    # 4. Activity Context Analysis
    print("  Analyzing activities before and after standing...")
    context_results = analyze_standing_context(data, stats_dir, language)
    results['context'] = context_results
    
    # 5. Standing by Department
    print("  Comparing standing patterns between departments...")
    department_results = compare_department_standing(data, stand_data, vis_dir, language)
    results['by_department'] = department_results
    
    print(f"Standing analysis complete. Results saved to {output_dir}")
    return results

def analyze_standing_durations(stand_data, output_dir=None, language='en'):
    """
    Analyze the distribution of standing durations
    
    Parameters:
    -----------
    stand_data : pandas.DataFrame
        Input dataframe filtered for standing activities
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with standing duration statistics
    """
    # Calculate overall statistics
    results = {
        'overall': {
            'count': len(stand_data),
            'min': stand_data['duration'].min(),
            'max': stand_data['duration'].max(),
            'mean': stand_data['duration'].mean(),
            'median': stand_data['duration'].median(),
            'std': stand_data['duration'].std(),
            'total_duration': stand_data['duration'].sum(),
        }
    }
    
    # Add formatted times for readability
    results['overall']['formatted_total'] = format_seconds_to_hms(results['overall']['total_duration'])
    results['overall']['formatted_mean'] = format_seconds_to_hms(results['overall']['mean'])
    results['overall']['formatted_median'] = format_seconds_to_hms(results['overall']['median'])
    
    # Calculate duration distribution
    duration_bins = [0, 5, 15, 30, 60, 120, 300, 600, float('inf')]
    bin_labels = ['0-5s', '5-15s', '15-30s', '30-60s', '1-2min', '2-5min', '5-10min', '>10min']
    stand_data['duration_category'] = pd.cut(stand_data['duration'], bins=duration_bins, labels=bin_labels)
    
    duration_dist = stand_data['duration_category'].value_counts().sort_index()
    results['duration_distribution'] = duration_dist.to_dict()
    
    # Calculate statistics by employee
    employee_stats = {}
    
    for emp_id, emp_data in stand_data.groupby('id'):
        total_duration = emp_data['duration'].sum()
        
        stats = {
            'count': len(emp_data),
            'min': emp_data['duration'].min(),
            'max': emp_data['duration'].max(),
            'mean': emp_data['duration'].mean(),
            'median': emp_data['duration'].median(),
            'std': emp_data['duration'].std(),
            'total_duration': total_duration,
            'formatted_total': format_seconds_to_hms(total_duration),
            'formatted_mean': format_seconds_to_hms(emp_data['duration'].mean()),
            'formatted_median': format_seconds_to_hms(emp_data['duration'].median())
        }
        
        # Duration distribution for this employee
        emp_dist = emp_data['duration_category'].value_counts().sort_index()
        stats['duration_distribution'] = emp_dist.to_dict()
        
        employee_stats[emp_id] = stats
    
    results['by_employee'] = employee_stats
    
    # Save statistics
    if output_dir:
        # Create DataFrames for saving statistics
        overall_df = pd.DataFrame([results['overall']])
        overall_df.to_csv(output_dir / 'standing_duration_overall.csv', index=False)
        
        emp_df = pd.DataFrame([
            {
                'employee_id': emp_id,
                'count': stats['count'],
                'min_seconds': stats['min'],
                'max_seconds': stats['max'],
                'mean_seconds': stats['mean'],
                'median_seconds': stats['median'],
                'std': stats['std'],
                'total_seconds': stats['total_duration'],
                'formatted_total': stats['formatted_total'],
                'formatted_mean': stats['formatted_mean'],
                'formatted_median': stats['formatted_median']
            }
            for emp_id, stats in employee_stats.items()
        ])
        
        if not emp_df.empty:
            emp_df.to_csv(output_dir / 'standing_duration_by_employee.csv', index=False)
        
        # Create distribution data for saving
        dist_df = pd.DataFrame({
            'duration_category': results['duration_distribution'].keys(),
            'count': results['duration_distribution'].values()
        })
        
        if not dist_df.empty:
            dist_df.to_csv(output_dir / 'standing_duration_distribution.csv', index=False)
    
    return results

def analyze_standing_by_region(stand_data, output_dir=None, language='en'):
    """
    Analyze standing time by region
    
    Parameters:
    -----------
    stand_data : pandas.DataFrame
        Input dataframe filtered for standing activities
    output_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with standing statistics by region
    """
    results = {}
    
    # Calculate total standing time by region
    region_stats = []
    
    for region, region_data in stand_data.groupby('region'):
        total_duration = region_data['duration'].sum()
        
        stats = {
            'region': region,
            'total_duration': total_duration,
            'formatted_total': format_seconds_to_hms(total_duration),
            'count': len(region_data),
            'mean_duration': region_data['duration'].mean(),
            'formatted_mean': format_seconds_to_hms(region_data['duration'].mean()),
            'median_duration': region_data['duration'].median(),
            'min_duration': region_data['duration'].min(),
            'max_duration': region_data['duration'].max()
        }
        
        region_stats.append(stats)
        results[region] = stats
    
    # Sort by total duration (descending)
    region_stats_df = pd.DataFrame(region_stats)
    if not region_stats_df.empty:
        region_stats_df = region_stats_df.sort_values('total_duration', ascending=False)
    
    # Create visualization of standing time by region
    if output_dir and not region_stats_df.empty:
        # Create visualization for top regions
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Focus on top 15 regions for better visualization
        top_regions = region_stats_df.head(15)
        
        # Calculate standing time in hours for better visualization
        top_regions['hours'] = top_regions['total_duration'] / 3600
        
        # Plot standing time by region
        bars = ax.barh(
            top_regions['region'],
            top_regions['hours'],
            color='#FFAA00'  # Using the color for Standing from the codebase
        )
        
        # Add labels
        for bar, total in zip(bars, top_regions['formatted_total']):
            width = bar.get_width()
            ax.text(
                width + 0.1,
                bar.get_y() + bar.get_height()/2,
                f"{width:.2f} hrs\n{total}",
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Top Regions by Standing Time', language), fontsize=16)
        ax.set_xlabel(get_text('Duration (hours)', language), fontsize=14)
        ax.set_ylabel(get_text('Region', language), fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'standing_by_region.png')
        
        # Create visualization for standing time percentage by region
        total_standing_time = stand_data['duration'].sum()
        top_regions['percentage'] = (top_regions['total_duration'] / total_standing_time * 100)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot standing percentage by region
        bars = ax.barh(
            top_regions['region'],
            top_regions['percentage'],
            color='#FFAA00'
        )
        
        # Add labels
        for bar, total in zip(bars, top_regions['formatted_total']):
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%\n{total}",
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Top Regions by Standing Time Percentage', language), fontsize=16)
        ax.set_xlabel(get_text('Percentage of Total Standing Time (%)', language), fontsize=14)
        ax.set_ylabel(get_text('Region', language), fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'standing_percentage_by_region.png')
    
    return results

def analyze_standing_by_employee(data, stand_data, output_dir=None, language='en'):
    """
    Analyze standing time by employee
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    stand_data : pandas.DataFrame
        Input dataframe filtered for standing activities
    output_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with standing statistics by employee
    """
    results = {}
    
    # Calculate total time and standing time by employee
    employee_stats = []
    
    for emp_id, emp_data in data.groupby('id'):
        total_time = emp_data['duration'].sum()
        
        # Get standing data for this employee
        emp_standing = stand_data[stand_data['id'] == emp_id]
        standing_time = emp_standing['duration'].sum() if not emp_standing.empty else 0
        standing_percentage = (standing_time / total_time * 100) if total_time > 0 else 0
        
        # Get department information
        department = emp_data['department'].iloc[0] if 'department' in emp_data.columns else 'Unknown'
        
        # Calculate most common regions for standing
        if not emp_standing.empty:
            region_time = emp_standing.groupby('region')['duration'].sum()
            top_regions = region_time.sort_values(ascending=False).head(3).index.tolist()
        else:
            top_regions = []
        
        stats = {
            'employee_id': emp_id,
            'department': department,
            'total_time': total_time,
            'standing_time': standing_time,
            'standing_percentage': standing_percentage,
            'formatted_total': format_seconds_to_hms(total_time),
            'formatted_standing': format_seconds_to_hms(standing_time),
            'top_standing_regions': top_regions
        }
        
        employee_stats.append(stats)
        results[emp_id] = stats
    
    # Sort by standing percentage (descending)
    employee_stats_df = pd.DataFrame(employee_stats)
    if not employee_stats_df.empty:
        employee_stats_df = employee_stats_df.sort_values('standing_percentage', ascending=False)
    
    # Create visualization of standing time by employee
    if output_dir and not employee_stats_df.empty:
        # Create visualization
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get employee colors
        employee_colors = get_employee_colors()
        
        # Plot standing percentage by employee
        bars = ax.bar(
            employee_stats_df['employee_id'],
            employee_stats_df['standing_percentage'],
            color=[employee_colors.get(emp, '#333333') for emp in employee_stats_df['employee_id']]
        )
        
        # Add labels
        for bar, total, standing in zip(
            bars, 
            employee_stats_df['formatted_total'], 
            employee_stats_df['formatted_standing']
        ):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 1,
                f"{height:.1f}%\n{standing}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Standing Time Percentage by Employee', language), fontsize=16)
        ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
        ax.set_ylabel(get_text('Standing Time (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'standing_by_employee.png')
        
        # Create visualization of top standing regions by employee
        if 'top_standing_regions' in employee_stats_df.columns and any(employee_stats_df['top_standing_regions']):
            # Create a matrix of region usage by employee
            top_regions_all = []
            for regions in employee_stats_df['top_standing_regions']:
                if regions:
                    top_regions_all.extend(regions)
            
            top_regions_all = list(set(top_regions_all))
            if top_regions_all:
                # Create heatmap data
                heatmap_data = pd.DataFrame(index=employee_stats_df['employee_id'], columns=top_regions_all)
                
                for emp_id, emp_stats in results.items():
                    emp_standing = stand_data[stand_data['id'] == emp_id]
                    if not emp_standing.empty:
                        region_time = emp_standing.groupby('region')['duration'].sum()
                        for region in top_regions_all:
                            if region in region_time.index:
                                heatmap_data.at[emp_id, region] = region_time[region]
                            else:
                                heatmap_data.at[emp_id, region] = 0
                
                # Fill NaN with 0
                heatmap_data = heatmap_data.fillna(0)
                
                # Convert to percentage for better visualization
                heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
                
                # Create heatmap
                plt.figure(figsize=(14, 10))
                sns.heatmap(
                    heatmap_pct, 
                    annot=True, 
                    fmt=".1f",
                    cmap="YlOrRd",
                    linewidths=0.5,
                    cbar_kws={'label': get_text('Percentage of Standing Time (%)', language)}
                )
                
                plt.title(get_text('Distribution of Standing Time by Region and Employee', language), fontsize=16)
                plt.tight_layout()
                
                save_figure(plt.gcf(), output_dir / 'standing_region_heatmap.png')
    
    return results

def analyze_standing_context(data, output_dir=None, language='en'):
    """
    Analyze activities before and after standing
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path, optional
        Directory to save statistics
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with standing context analysis results
    """
    results = {'overall': {}, 'by_employee': {}}
    
    # Analyze activity sequences
    before_standing = []
    after_standing = []
    employee_context = defaultdict(lambda: {'before': [], 'after': []})
    
    for emp_id, emp_data in data.groupby('id'):
        # Sort by time
        emp_data = emp_data.sort_values('startTime')
        
        # Create sequence of activities
        activities = list(emp_data['activity'])
        
        # Look for activities before and after standing
        for i in range(1, len(activities) - 1):
            if activities[i] == 'Stand':
                before_standing.append(activities[i-1])
                after_standing.append(activities[i+1])
                
                employee_context[emp_id]['before'].append(activities[i-1])
                employee_context[emp_id]['after'].append(activities[i+1])
    
    # Calculate overall statistics
    before_counts = Counter(before_standing)
    after_counts = Counter(after_standing)
    
    results['overall'] = {
        'total_sequences': len(before_standing),
        'before_standing': dict(before_counts.most_common()),
        'after_standing': dict(after_counts.most_common())
    }
    
    # Calculate employee statistics
    for emp_id, context in employee_context.items():
        before_counts = Counter(context['before'])
        after_counts = Counter(context['after'])
        
        results['by_employee'][emp_id] = {
            'total_sequences': len(context['before']),
            'before_standing': dict(before_counts.most_common()),
            'after_standing': dict(after_counts.most_common())
        }
    
    # Save statistics
    if output_dir:
        # Overall before/after counts
        before_df = pd.DataFrame([
            {
                'activity': activity,
                'count': count,
                'percentage': (count / len(before_standing) * 100) if before_standing else 0
            }
            for activity, count in before_counts.most_common()
        ])
        
        if not before_df.empty:
            before_df.to_csv(output_dir / 'activities_before_standing.csv', index=False)
        
        after_df = pd.DataFrame([
            {
                'activity': activity,
                'count': count,
                'percentage': (count / len(after_standing) * 100) if after_standing else 0
            }
            for activity, count in after_counts.most_common()
        ])
        
        if not after_df.empty:
            after_df.to_csv(output_dir / 'activities_after_standing.csv', index=False)
        
        # Employee context summary
        employee_summary = []
        for emp_id, context in results['by_employee'].items():
            top_before = next(iter(context['before_standing'].items())) if context['before_standing'] else ('', 0)
            top_after = next(iter(context['after_standing'].items())) if context['after_standing'] else ('', 0)
            
            employee_summary.append({
                'employee_id': emp_id,
                'total_standing_sequences': context['total_sequences'],
                'top_before_activity': top_before[0],
                'top_before_count': top_before[1],
                'top_after_activity': top_after[0],
                'top_after_count': top_after[1]
            })
        
        if employee_summary:
            pd.DataFrame(employee_summary).to_csv(output_dir / 'standing_context_by_employee.csv', index=False)
    
    return results

def compare_department_standing(data, stand_data, output_dir=None, language='en'):
    """
    Compare standing patterns between departments
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    stand_data : pandas.DataFrame
        Input dataframe filtered for standing activities
    output_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with department standing comparison results
    """
    results = {}
    
    # Check if department column exists
    if 'department' not in data.columns:
        print("  Department information not available in the dataset.")
        return results
    
    # Calculate standing statistics by department
    dept_stats = []
    
    for dept, dept_data in data.groupby('department'):
        total_time = dept_data['duration'].sum()
        
        # Get standing data for this department
        dept_standing = stand_data[stand_data['id'].isin(dept_data['id'].unique())]
        standing_time = dept_standing['duration'].sum() if not dept_standing.empty else 0
        standing_percentage = (standing_time / total_time * 100) if total_time > 0 else 0
        
        stats = {
            'department': dept,
            'total_time': total_time,
            'standing_time': standing_time,
            'standing_percentage': standing_percentage,
            'formatted_total': format_seconds_to_hms(total_time),
            'formatted_standing': format_seconds_to_hms(standing_time),
            'employee_count': dept_data['id'].nunique()
        }
        
        # Top regions for standing in this department
        if not dept_standing.empty:
            region_time = dept_standing.groupby('region')['duration'].sum()
            top_regions = region_time.sort_values(ascending=False).head(5)
            stats['top_regions'] = {
                region: {
                    'duration': duration,
                    'percentage': (duration / standing_time * 100) if standing_time > 0 else 0,
                    'formatted_duration': format_seconds_to_hms(duration)
                }
                for region, duration in top_regions.items()
            }
        else:
            stats['top_regions'] = {}
        
        dept_stats.append(stats)
        results[dept] = stats
    
    # Create visualizations
    if output_dir and dept_stats:
        # Department comparison
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get department colors
        department_colors = get_department_colors()
        
        # Sort by department name
        dept_stats_df = pd.DataFrame([
            {
                'department': dept['department'],
                'standing_percentage': dept['standing_percentage'],
                'formatted_standing': dept['formatted_standing']
            }
            for dept in dept_stats
        ]).sort_values('department')
        
        # Plot standing percentage by department
        bars = ax.bar(
            dept_stats_df['department'],
            dept_stats_df['standing_percentage'],
            color=[department_colors.get(dept, '#333333') for dept in dept_stats_df['department']]
        )
        
        # Add labels
        for bar, standing in zip(bars, dept_stats_df['formatted_standing']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 1,
                f"{height:.1f}%\n{standing}",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Standing Time Percentage by Department', language), fontsize=16)
        ax.set_xlabel(get_text('Department', language), fontsize=14)
        ax.set_ylabel(get_text('Standing Time (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'standing_by_department.png')
        
        # Top regions by department
        for dept in dept_stats:
            dept_name = dept['department']
            if not dept['top_regions']:
                continue
                
            # Create visualization of top regions for this department
            fig, ax = plt.subplots(figsize=(12, 8))
            
            regions = list(dept['top_regions'].keys())
            durations = [region_data['duration'] / 3600 for region_data in dept['top_regions'].values()]
            percentages = [region_data['percentage'] for region_data in dept['top_regions'].values()]
            formatted_durations = [region_data['formatted_duration'] for region_data in dept['top_regions'].values()]
            
            # Plot standing time by region
            bars = ax.bar(
                regions,
                durations,
                color='#FFAA00'
            )
            
            # Add labels
            for bar, percentage, formatted in zip(bars, percentages, formatted_durations):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.1,
                    f"{percentage:.1f}%\n{formatted}",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            ax.set_title(get_text('Top Standing Regions - {0} Department', language).format(dept_name), fontsize=16)
            ax.set_xlabel(get_text('Region', language), fontsize=14)
            ax.set_ylabel(get_text('Standing Time (hours)', language), fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, output_dir / f'standing_regions_{dept_name.lower()}.png')
    
    return results