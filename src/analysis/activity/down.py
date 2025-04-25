"""
Handle Down Analysis Module

Analyzes 'Handle down' patterns of bakery employees to identify ergonomic concerns and optimization opportunities.
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

def analyze_handling_down(data, output_dir=None, language='en'):
    """
    Analyze 'Handle down' patterns in the bakery dataset
    
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
        Dictionary with 'Handle down' pattern analysis results
    """
    # Create subdirectories
    if output_dir:
        stats_dir = ensure_dir_exists(output_dir / 'statistics')
        vis_dir = ensure_dir_exists(output_dir / 'visualizations')
    
    # Filter for 'Handle down' activities only
    handle_down_data = data[data['activity'] == 'Handle down']
    
    if handle_down_data.empty:
        print("No 'Handle down' data found in the dataset.")
        return {}
    
    results = {}
    
    # 1. Handle Down Duration Analysis
    print("  Analyzing 'Handle down' durations...")
    duration_results = analyze_handle_down_durations(handle_down_data, stats_dir, language)
    results['durations'] = duration_results
    
    # 2. Handle Down by Region
    print("  Analyzing 'Handle down' time by region...")
    region_results = analyze_handle_down_by_region(handle_down_data, vis_dir, language)
    results['by_region'] = region_results
    
    # 3. Handle Down by Employee
    print("  Analyzing 'Handle down' time by employee...")
    employee_results = analyze_handle_down_by_employee(data, handle_down_data, vis_dir, language)
    results['by_employee'] = employee_results
    
    # 4. Department Comparison
    print("  Comparing 'Handle down' patterns between departments...")
    department_results = compare_department_handle_down(data, handle_down_data, vis_dir, language)
    results['by_department'] = department_results
    
    # 5. Alternative Assessment
    print("  Analyzing potential alternatives to bending...")
    alternative_results = analyze_bending_alternatives(data, handle_down_data, stats_dir, language)
    results['alternatives'] = alternative_results
    
    # 6. Ergonomic Risk Assessment
    print("  Assessing ergonomic risks from 'Handle down' activities...")
    risk_results = assess_ergonomic_risks_handle_down(data, handle_down_data, stats_dir, language)
    results['ergonomic_risks'] = risk_results
    
    print(f"'Handle down' analysis complete. Results saved to {output_dir}")
    return results

def analyze_handle_down_durations(handle_down_data, output_dir=None, language='en'):
    """
    Analyze the distribution of 'Handle down' durations
    
    Parameters:
    -----------
    handle_down_data : pandas.DataFrame
        Input dataframe filtered for 'Handle down' activities
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with 'Handle down' duration statistics
    """
    # Calculate overall statistics
    results = {
        'overall': {
            'count': len(handle_down_data),
            'min': handle_down_data['duration'].min(),
            'max': handle_down_data['duration'].max(),
            'mean': handle_down_data['duration'].mean(),
            'median': handle_down_data['duration'].median(),
            'std': handle_down_data['duration'].std(),
            'total_duration': handle_down_data['duration'].sum(),
        }
    }
    
    # Add formatted times for readability
    results['overall']['formatted_total'] = format_seconds_to_hms(results['overall']['total_duration'])
    results['overall']['formatted_mean'] = format_seconds_to_hms(results['overall']['mean'])
    results['overall']['formatted_median'] = format_seconds_to_hms(results['overall']['median'])
    
    # Calculate duration distribution
    duration_bins = [0, 5, 15, 30, 60, 120, 300, 600, float('inf')]
    bin_labels = ['0-5s', '5-15s', '15-30s', '30-60s', '1-2min', '2-5min', '5-10min', '>10min']
    handle_down_data['duration_category'] = pd.cut(handle_down_data['duration'], bins=duration_bins, labels=bin_labels)
    
    duration_dist = handle_down_data['duration_category'].value_counts().sort_index()
    results['duration_distribution'] = duration_dist.to_dict()
    
    # Calculate statistics by employee
    employee_stats = {}
    
    for emp_id, emp_data in handle_down_data.groupby('id'):
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
        overall_df.to_csv(output_dir / 'handle_down_duration_overall.csv', index=False)
        
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
            emp_df.to_csv(output_dir / 'handle_down_duration_by_employee.csv', index=False)
        
        # Create distribution data for saving
        dist_df = pd.DataFrame({
            'duration_category': results['duration_distribution'].keys(),
            'count': results['duration_distribution'].values()
        })
        
        if not dist_df.empty:
            dist_df.to_csv(output_dir / 'handle_down_duration_distribution.csv', index=False)
    
    return results

def analyze_handle_down_by_region(handle_down_data, output_dir=None, language='en'):
    """
    Analyze 'Handle down' time by region
    
    Parameters:
    -----------
    handle_down_data : pandas.DataFrame
        Input dataframe filtered for 'Handle down' activities
    output_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with 'Handle down' statistics by region
    """
    results = {}
    
    # Calculate total 'Handle down' time by region
    region_stats = []
    
    for region, region_data in handle_down_data.groupby('region'):
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
    
    # Create visualization of 'Handle down' time by region
    if output_dir and not region_stats_df.empty:
        # Create visualization for top regions
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Focus on top 15 regions for better visualization
        top_regions = region_stats_df.head(15)
        
        # Calculate 'Handle down' time in hours for better visualization
        top_regions['hours'] = top_regions['total_duration'] / 3600
        
        # Plot 'Handle down' time by region
        bars = ax.barh(
            top_regions['region'],
            top_regions['hours'],
            color='#96BEE6'  # Using the color for Handle down from the codebase
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
        
        ax.set_title(get_text('Top Regions by Handle Down Time', language), fontsize=16)
        ax.set_xlabel(get_text('Duration (hours)', language), fontsize=14)
        ax.set_ylabel(get_text('Region', language), fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'handle_down_by_region.png')
        
        # Create visualization for 'Handle down' time percentage by region
        total_handle_down_time = handle_down_data['duration'].sum()
        top_regions['percentage'] = (top_regions['total_duration'] / total_handle_down_time * 100)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot 'Handle down' percentage by region
        bars = ax.barh(
            top_regions['region'],
            top_regions['percentage'],
            color='#96BEE6'
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
        
        ax.set_title(get_text('Top Regions by Handle Down Time Percentage', language), fontsize=16)
        ax.set_xlabel(get_text('Percentage of Total Handle Down Time (%)', language), fontsize=14)
        ax.set_ylabel(get_text('Region', language), fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'handle_down_percentage_by_region.png')
    
    return results

def analyze_handle_down_by_employee(data, handle_down_data, output_dir=None, language='en'):
    """
    Analyze 'Handle down' time by employee
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    handle_down_data : pandas.DataFrame
        Input dataframe filtered for 'Handle down' activities
    output_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with 'Handle down' statistics by employee
    """
    results = {}
    
    # Calculate total time and 'Handle down' time by employee
    employee_stats = []
    
    for emp_id, emp_data in data.groupby('id'):
        total_time = emp_data['duration'].sum()
        
        # Get 'Handle down' data for this employee
        emp_handle_down = handle_down_data[handle_down_data['id'] == emp_id]
        handle_down_time = emp_handle_down['duration'].sum() if not emp_handle_down.empty else 0
        handle_down_percentage = (handle_down_time / total_time * 100) if total_time > 0 else 0
        
        # Get department information
        department = emp_data['department'].iloc[0] if 'department' in emp_data.columns else 'Unknown'
        
        # Calculate most common regions for 'Handle down'
        if not emp_handle_down.empty:
            region_time = emp_handle_down.groupby('region')['duration'].sum()
            top_regions = region_time.sort_values(ascending=False).head(3).index.tolist()
        else:
            top_regions = []
        
        stats = {
            'employee_id': emp_id,
            'department': department,
            'total_time': total_time,
            'handle_down_time': handle_down_time,
            'handle_down_percentage': handle_down_percentage,
            'formatted_total': format_seconds_to_hms(total_time),
            'formatted_handle_down': format_seconds_to_hms(handle_down_time),
            'top_handle_down_regions': top_regions
        }
        
        employee_stats.append(stats)
        results[emp_id] = stats
    
    # Sort by 'Handle down' percentage (descending)
    employee_stats_df = pd.DataFrame(employee_stats)
    if not employee_stats_df.empty:
        employee_stats_df = employee_stats_df.sort_values('handle_down_percentage', ascending=False)
    
    # Create visualization of 'Handle down' time by employee
    if output_dir and not employee_stats_df.empty:
        # Create visualization
        set_visualization_style()
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get employee colors
        employee_colors = get_employee_colors()
        
        # Plot 'Handle down' percentage by employee
        bars = ax.bar(
            employee_stats_df['employee_id'],
            employee_stats_df['handle_down_percentage'],
            color=[employee_colors.get(emp, '#333333') for emp in employee_stats_df['employee_id']]
        )
        
        # Add labels
        for bar, total, handle_down in zip(
            bars, 
            employee_stats_df['formatted_total'], 
            employee_stats_df['formatted_handle_down']
        ):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 1,
                f"{height:.1f}%\n{handle_down}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Handle Down Time Percentage by Employee', language), fontsize=16)
        ax.set_xlabel(get_text('Employee ID', language), fontsize=14)
        ax.set_ylabel(get_text('Handle Down Time (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'handle_down_by_employee.png')
        
        # Create visualization of top 'Handle down' regions by employee
        if 'top_handle_down_regions' in employee_stats_df.columns and any(employee_stats_df['top_handle_down_regions']):
            # Create a matrix of region usage by employee
            top_regions_all = []
            for regions in employee_stats_df['top_handle_down_regions']:
                if regions:
                    top_regions_all.extend(regions)
            
            top_regions_all = list(set(top_regions_all))
            if top_regions_all:
                # Create heatmap data
                heatmap_data = pd.DataFrame(index=employee_stats_df['employee_id'], columns=top_regions_all)
                
                for emp_id, emp_stats in results.items():
                    emp_handle_down = handle_down_data[handle_down_data['id'] == emp_id]
                    if not emp_handle_down.empty:
                        region_time = emp_handle_down.groupby('region')['duration'].sum()
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
                    cbar_kws={'label': get_text('Percentage of Handle Down Time (%)', language)}
                )
                
                plt.title(get_text('Distribution of Handle Down Time by Region and Employee', language), fontsize=16)
                plt.tight_layout()
                
                save_figure(plt.gcf(), output_dir / 'handle_down_region_heatmap.png')
    
    return results

def compare_department_handle_down(data, handle_down_data, output_dir=None, language='en'):
    """
    Compare 'Handle down' patterns between departments
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    handle_down_data : pandas.DataFrame
        Input dataframe filtered for 'Handle down' activities
    output_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with department 'Handle down' comparison results
    """
    results = {}
    
    # Check if department column exists
    if 'department' not in data.columns:
        print("  Department information not available in the dataset.")
        return results
    
    # Calculate 'Handle down' statistics by department
    dept_stats = []
    
    for dept, dept_data in data.groupby('department'):
        total_time = dept_data['duration'].sum()
        
        # Get 'Handle down' data for this department
        dept_handle_down = handle_down_data[handle_down_data['id'].isin(dept_data['id'].unique())]
        handle_down_time = dept_handle_down['duration'].sum() if not dept_handle_down.empty else 0
        handle_down_percentage = (handle_down_time / total_time * 100) if total_time > 0 else 0
        
        stats = {
            'department': dept,
            'total_time': total_time,
            'handle_down_time': handle_down_time,
            'handle_down_percentage': handle_down_percentage,
            'formatted_total': format_seconds_to_hms(total_time),
            'formatted_handle_down': format_seconds_to_hms(handle_down_time),
            'employee_count': dept_data['id'].nunique()
        }
        
        # Top regions for 'Handle down' in this department
        if not dept_handle_down.empty:
            region_time = dept_handle_down.groupby('region')['duration'].sum()
            top_regions = region_time.sort_values(ascending=False).head(5)
            stats['top_regions'] = {
                region: {
                    'duration': duration,
                    'percentage': (duration / handle_down_time * 100) if handle_down_time > 0 else 0,
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
                'handle_down_percentage': dept['handle_down_percentage'],
                'formatted_handle_down': dept['formatted_handle_down']
            }
            for dept in dept_stats
        ]).sort_values('department')
        
        # Plot 'Handle down' percentage by department
        bars = ax.bar(
            dept_stats_df['department'],
            dept_stats_df['handle_down_percentage'],
            color=[department_colors.get(dept, '#333333') for dept in dept_stats_df['department']]
        )
        
        # Add labels
        for bar, handle_down in zip(bars, dept_stats_df['formatted_handle_down']):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 1,
                f"{height:.1f}%\n{handle_down}",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        ax.set_title(get_text('Handle Down Time Percentage by Department', language), fontsize=16)
        ax.set_xlabel(get_text('Department', language), fontsize=14)
        ax.set_ylabel(get_text('Handle Down Time (%)', language), fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, output_dir / 'handle_down_by_department.png')
        
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
            
            # Plot 'Handle down' time by region
            bars = ax.bar(
                regions,
                durations,
                color='#96BEE6'
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
            
            ax.set_title(get_text('Top Handle Down Regions - {0} Department', language).format(dept_name), fontsize=16)
            ax.set_xlabel(get_text('Region', language), fontsize=14)
            ax.set_ylabel(get_text('Handle Down Time (hours)', language), fontsize=14)
            ax.grid(axis='y', alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_figure(fig, output_dir / f'handle_down_regions_{dept_name.lower()}.png')
    
    return results

def analyze_bending_alternatives(data, handle_down_data, output_dir=None, language='en'):
    """
    Analyze potential alternatives to bending (Handle down)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    handle_down_data : pandas.DataFrame
        Input dataframe filtered for 'Handle down' activities
    output_dir : Path, optional
        Directory to save statistics
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with alternative assessment results
    """
    results = {'regions': {}, 'employees': {}}
    
    # Find regions with most 'Handle down' time
    region_handle_down = handle_down_data.groupby('region')['duration'].sum().sort_values(ascending=False)
    
    # Analyze where some employees bend less than others for the same tasks/regions
    region_emp_stats = []
    
    for region in region_handle_down.index:
        # Get data for this region
        region_data = data[data['region'] == region]
        region_handle_down_data = handle_down_data[handle_down_data['region'] == region]
        
        if region_data.empty or region_handle_down_data.empty:
            continue
        
        # Calculate total time and 'Handle down' time by employee in this region
        employee_stats = []
        
        for emp_id, emp_data in region_data.groupby('id'):
            total_time = emp_data['duration'].sum()
            
            # Get 'Handle down' data for this employee in this region
            emp_handle_down = region_handle_down_data[region_handle_down_data['id'] == emp_id]
            handle_down_time = emp_handle_down['duration'].sum() if not emp_handle_down.empty else 0
            handle_down_percentage = (handle_down_time / total_time * 100) if total_time > 0 else 0
            
            employee_stats.append({
                'employee_id': emp_id,
                'region': region,
                'total_time': total_time,
                'handle_down_time': handle_down_time,
                'handle_down_percentage': handle_down_percentage,
                'formatted_total': format_seconds_to_hms(total_time),
                'formatted_handle_down': format_seconds_to_hms(handle_down_time)
            })
        
        # Only include regions with multiple employees
        if len(employee_stats) < 2:
            continue
        
        # Check if there's significant variation in bending percentage
        handle_down_percentages = [emp['handle_down_percentage'] for emp in employee_stats]
        min_pct = min(handle_down_percentages)
        max_pct = max(handle_down_percentages)
        pct_range = max_pct - min_pct
        
        # Only include regions with significant variation (>10% difference)
        if pct_range < 10:
            continue
        
        # Sort employees by 'Handle down' percentage
        employee_stats.sort(key=lambda x: x['handle_down_percentage'])
        
        # Store results
        results['regions'][region] = {
            'employee_stats': employee_stats,
            'min_percentage': min_pct,
            'max_percentage': max_pct,
            'percentage_range': pct_range,
            'best_employee': employee_stats[0]['employee_id'],
            'worst_employee': employee_stats[-1]['employee_id']
        }
        
        # Add to list for saving to CSV
        for emp_stat in employee_stats:
            region_emp_stats.append(emp_stat)
    
    # Find employees who consistently bend less than others
    employee_rankings = defaultdict(int)
    
    for region, region_data in results['regions'].items():
        emp_stats = region_data['employee_stats']
        
        # Rank employees by 'Handle down' percentage (lower is better)
        for rank, emp_stat in enumerate(emp_stats):
            employee_rankings[emp_stat['employee_id']] += rank
    
    # Normalize rankings
    num_regions = len(results['regions'])
    if num_regions > 0:
        normalized_rankings = {emp_id: rank / num_regions for emp_id, rank in employee_rankings.items()}
        
        # Sort employees by normalized ranking
        sorted_employees = sorted(normalized_rankings.items(), key=lambda x: x[1])
        
        # Store best and worst overall employees
        results['employees'] = {
            'rankings': normalized_rankings,
            'sorted_rankings': sorted_employees,
            'best_employees': [emp_id for emp_id, _ in sorted_employees[:3]] if len(sorted_employees) >= 3 else [emp_id for emp_id, _ in sorted_employees],
            'worst_employees': [emp_id for emp_id, _ in sorted_employees[-3:]] if len(sorted_employees) >= 3 else [emp_id for emp_id, _ in sorted_employees[::-1]]
        }
    
    # Save statistics
    if output_dir and region_emp_stats:
        # Save region-employee stats
        region_emp_df = pd.DataFrame(region_emp_stats)
        region_emp_df.to_csv(output_dir / 'handle_down_variation_by_region.csv', index=False)
        
        # Save employee rankings
        if 'rankings' in results['employees']:
            rankings_df = pd.DataFrame([
                {
                    'employee_id': emp_id,
                    'normalized_ranking': rank,
                    'is_best': emp_id in results['employees']['best_employees'],
                    'is_worst': emp_id in results['employees']['worst_employees']
                }
                for emp_id, rank in results['employees']['rankings'].items()
            ])
            
            rankings_df = rankings_df.sort_values('normalized_ranking')
            rankings_df.to_csv(output_dir / 'handle_down_employee_rankings.csv', index=False)
    
    return results

def assess_ergonomic_risks_handle_down(data, handle_down_data, output_dir=None, language='en'):
    """
    Assess ergonomic risks from 'Handle down' activities
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete input dataframe
    handle_down_data : pandas.DataFrame
        Input dataframe filtered for 'Handle down' activities
    output_dir : Path, optional
        Directory to save statistics
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with ergonomic risk assessment results
    """
    results = {'overall': {}, 'by_employee': {}}
    
    # Define risk thresholds for 'Handle down' (more strict than 'Handle up')
    risk_thresholds = {
        'low': 4,       # < 4% of time is low risk
        'medium': 12,   # 4-12% is medium risk
        'high': 25,     # 12-25% is high risk
        'very_high': 100  # > 25% is very high risk
    }
    
    # Calculate overall risk metrics
    total_time = data['duration'].sum()
    handle_down_time = handle_down_data['duration'].sum()
    handle_down_percentage = (handle_down_time / total_time * 100) if total_time > 0 else 0
    
    # Determine overall risk level
    overall_risk_level = 'low'
    for level, threshold in sorted(risk_thresholds.items(), key=lambda x: x[1]):
        if handle_down_percentage <= threshold:
            overall_risk_level = level
            break
    
    # Calculate longest continuous 'Handle down' periods
    continuous_periods = []
    
    for emp_id, emp_data in handle_down_data.groupby('id'):
        # Sort by time
        emp_data = emp_data.sort_values('startTime')
        
        # Get continuous periods
        for region, region_data in emp_data.groupby('region'):
            region_data = region_data.sort_values('startTime')
            
            for i, row in region_data.iterrows():
                continuous_periods.append({
                    'employee_id': emp_id,
                    'region': region,
                    'duration': row['duration'],
                    'start_time': row['startTime'],
                    'end_time': row['endTime']
                })
    
    # Sort by duration (descending)
    continuous_periods.sort(key=lambda x: x['duration'], reverse=True)
    
    # Store overall results
    results['overall'] = {
        'total_time': total_time,
        'handle_down_time': handle_down_time,
        'handle_down_percentage': handle_down_percentage,
        'risk_level': overall_risk_level,
        'formatted_total': format_seconds_to_hms(total_time),
        'formatted_handle_down': format_seconds_to_hms(handle_down_time),
        'longest_periods': continuous_periods[:10] if continuous_periods else []
    }
    
    # Calculate risk metrics by employee
    employee_risks = []
    
    for emp_id, emp_data in data.groupby('id'):
        total_emp_time = emp_data['duration'].sum()
        
        # Get 'Handle down' data for this employee
        emp_handle_down = handle_down_data[handle_down_data['id'] == emp_id]
        handle_down_emp_time = emp_handle_down['duration'].sum() if not emp_handle_down.empty else 0
        handle_down_emp_percentage = (handle_down_emp_time / total_emp_time * 100) if total_emp_time > 0 else 0
        
        # Determine risk level for this employee
        emp_risk_level = 'low'
        for level, threshold in sorted(risk_thresholds.items(), key=lambda x: x[1]):
            if handle_down_emp_percentage <= threshold:
                emp_risk_level = level
                break
        
        # Get longest continuous period for this employee
        emp_longest_period = next((p for p in continuous_periods if p['employee_id'] == emp_id), {})
        
        # Get department
        department = emp_data['department'].iloc[0] if 'department' in emp_data.columns else 'Unknown'
        
        risk_data = {
            'employee_id': emp_id,
            'department': department,
            'total_time': total_emp_time,
            'handle_down_time': handle_down_emp_time,
            'handle_down_percentage': handle_down_emp_percentage,
            'risk_level': emp_risk_level,
            'formatted_total': format_seconds_to_hms(total_emp_time),
            'formatted_handle_down': format_seconds_to_hms(handle_down_emp_time)
        }
        
        if emp_longest_period:
            risk_data['longest_period'] = emp_longest_period['duration']
            risk_data['longest_region'] = emp_longest_period['region']
            risk_data['formatted_longest'] = format_seconds_to_hms(emp_longest_period['duration'])
        
        employee_risks.append(risk_data)
        results['by_employee'][emp_id] = risk_data
    
    # Save risk assessment results
    if output_dir:
        # Save overall summary
        overall_df = pd.DataFrame([{
            'total_time': results['overall']['total_time'],
            'handle_down_time': results['overall']['handle_down_time'],
            'handle_down_percentage': results['overall']['handle_down_percentage'],
            'risk_level': results['overall']['risk_level'],
            'formatted_total': results['overall']['formatted_total'],
            'formatted_handle_down': results['overall']['formatted_handle_down']
        }])
        
        overall_df.to_csv(output_dir / 'ergonomic_risk_handle_down_overall.csv', index=False)
        
        # Save employee risks
        emp_risk_df = pd.DataFrame(employee_risks)
        
        if not emp_risk_df.empty:
            emp_risk_df = emp_risk_df.sort_values('handle_down_percentage', ascending=False)
            emp_risk_df.to_csv(output_dir / 'ergonomic_risk_handle_down_by_employee.csv', index=False)
        
        # Save longest continuous periods
        if continuous_periods:
            longest_df = pd.DataFrame(continuous_periods[:20])  # Top 20 longest periods
            longest_df['formatted_duration'] = longest_df['duration'].apply(format_seconds_to_hms)
            longest_df.to_csv(output_dir / 'longest_handle_down_periods.csv', index=False)
    
    return results