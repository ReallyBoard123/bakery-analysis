#!/usr/bin/env python3
"""
Enhanced Bakery Walking Pattern Analysis

This script analyzes walking patterns of bakery employees based on sensor data, with focus on:

1. Dynamic time range detection rather than fixed 24-hour periods
2. Region analysis during peak walking times  
3. Statistical correlation between walking patterns, regions, and time periods
4. Detection of repetitive movement patterns and their significance
5. Deep statistical analysis with walking frequency, duration, and employee comparisons
"""

import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind, f_oneway
import seaborn as sns

from src.analysis.activity.walking import (
    analyze_walking_patterns, 
    analyze_walking_by_shift,
    determine_employee_time_range,
    create_region_focused_plot,
    detect_repetitive_walking,
    analyze_region_walking_patterns
)
from src.utils.file_utils import ensure_dir_exists, check_required_files
from src.utils.data_utils import load_sensor_data, classify_departments
from src.utils.time_utils import format_seconds_to_hms
from src.visualization.base import set_visualization_style, get_employee_colors, save_figure

def parse_arguments():
    """Parse command line arguments with enhanced options"""
    parser = argparse.ArgumentParser(description='Enhanced Bakery Employee Walking Pattern Analysis')
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                      help='Path to the data file')
    parser.add_argument('--output', type=str, default='output',
                      help='Directory to save output files')
    parser.add_argument('--german', action='store_true', help='Generate visualizations and reports in German')
    parser.add_argument('--walk', action='store_true', help='Analyze walking patterns')
    parser.add_argument('--employee', type=str, help='Specific employee ID to analyze (e.g., 32-A)')
    parser.add_argument('--time-slot', type=int, default=30, 
                        help='Time slot size in minutes (default: 30)')
    
    # New options for enhanced analysis
    parser.add_argument('--region-focus', action='store_true', 
                        help='Focus analysis on regions with highest walking time')
    parser.add_argument('--repetitive-patterns', action='store_true', 
                        help='Detect repetitive walking patterns')
    parser.add_argument('--statistical', action='store_true', 
                        help='Perform detailed statistical analysis')
    parser.add_argument('--region', type=str,
                        help='Focus analysis on a specific region')
    parser.add_argument('--compare-employees', action='store_true',
                        help='Compare walking patterns between employees')
    
    return parser.parse_args()

def perform_statistical_analysis(data, walking_data, output_dir=None, language='en'):
    """
    Perform detailed statistical analysis of walking patterns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete dataset
    walking_data : pandas.DataFrame
        Filtered walking data
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with statistical analysis results
    """
    print("  Performing detailed statistical analysis of walking patterns...")
    
    # Create directory for statistical analysis
    if output_dir:
        stats_dir = ensure_dir_exists(output_dir / 'statistics')
    
    # Results dictionary
    results = {}
    
    # 1. Compare walking proportions between employees
    # Calculate total time and walking time per employee
    emp_total_time = data.groupby('id')['duration'].sum().reset_index()
    emp_walking_time = walking_data.groupby('id')['duration'].sum().reset_index()
    
    # Merge to calculate walking percentage
    emp_merge = pd.merge(emp_total_time, emp_walking_time, on='id', suffixes=('_total', '_walking'))
    emp_merge['walking_percentage'] = (emp_merge['duration_walking'] / emp_merge['duration_total'] * 100).round(2)
    emp_merge['walking_hours'] = emp_merge['duration_walking'] / 3600
    emp_merge['total_hours'] = emp_merge['duration_total'] / 3600
    
    # Sort by walking percentage
    emp_merge = emp_merge.sort_values('walking_percentage', ascending=False)
    results['walking_percentage'] = emp_merge
    
    if output_dir:
        emp_merge.to_csv(stats_dir / 'employee_walking_percentage.csv', index=False)
        
        # Create visualization
        create_employee_percentage_visualization(emp_merge, stats_dir, language)
    
    # 2. Group employees by walking patterns for statistical comparison
    high_walkers = emp_merge[emp_merge['walking_percentage'] > emp_merge['walking_percentage'].median()]['id'].tolist()
    low_walkers = emp_merge[emp_merge['walking_percentage'] <= emp_merge['walking_percentage'].median()]['id'].tolist()
    
    # 3. Compare regions between high and low walkers
    if high_walkers and low_walkers:
        high_walker_data = walking_data[walking_data['id'].isin(high_walkers)]
        low_walker_data = walking_data[walking_data['id'].isin(low_walkers)]
        
        # Calculate region distribution
        high_regions = high_walker_data.groupby('region')['duration'].sum().reset_index()
        high_regions['percentage'] = (high_regions['duration'] / high_regions['duration'].sum() * 100).round(2)
        high_regions['group'] = 'High Walkers'
        
        low_regions = low_walker_data.groupby('region')['duration'].sum().reset_index()
        low_regions['percentage'] = (low_regions['duration'] / low_regions['duration'].sum() * 100).round(2)
        low_regions['group'] = 'Low Walkers'
        
        # Combine for comparison
        region_comparison = pd.concat([high_regions, low_regions])
        
        results['region_comparison'] = region_comparison
        
        if output_dir:
            region_comparison.to_csv(stats_dir / 'region_comparison_by_walker_type.csv', index=False)
            
            # Create visualization
            create_region_comparison_visualization(region_comparison, stats_dir, language)
            
        # 4. Statistical test for significant differences in regions
        region_stats = {}
        all_regions = data['region'].unique()
        
        for region in all_regions:
            # Get walking durations in this region for both groups
            high_durations = high_walker_data[high_walker_data['region'] == region]['duration'].values
            low_durations = low_walker_data[low_walker_data['region'] == region]['duration'].values
            
            if len(high_durations) > 0 and len(low_durations) > 0:
                # Perform t-test
                t_stat, p_value = ttest_ind(high_durations, low_durations, equal_var=False)
                
                region_stats[region] = {
                    'high_mean': np.mean(high_durations),
                    'low_mean': np.mean(low_durations),
                    'high_count': len(high_durations),
                    'low_count': len(low_durations),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Convert to DataFrame
        region_stats_df = pd.DataFrame.from_dict(region_stats, orient='index').reset_index()
        region_stats_df = region_stats_df.rename(columns={'index': 'region'})
        
        # Calculate absolute difference and sort
        region_stats_df['mean_difference'] = (region_stats_df['high_mean'] - region_stats_df['low_mean']).abs()
        region_stats_df = region_stats_df.sort_values(['significant', 'mean_difference'], ascending=[False, False])
        
        results['region_statistics'] = region_stats_df
        
        if output_dir:
            region_stats_df.to_csv(stats_dir / 'region_statistical_tests.csv', index=False)
            
            # Create visualization of significant differences
            create_significant_region_visualization(region_stats_df, stats_dir, language)
    
    # 5. Analyze walk durations by time of day
    walking_data['hour_of_day'] = (walking_data['startTime'] / 3600) % 24
    walking_data['time_category'] = pd.cut(
        walking_data['hour_of_day'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
    )
    
    # Walking duration by time category
    time_stats = walking_data.groupby('time_category').agg({
        'duration': ['mean', 'median', 'std', 'count']
    }).reset_index()
    
    # Flatten multi-index columns
    time_stats.columns = ['time_category', 'mean_duration', 'median_duration', 'std_duration', 'walk_count']
    
    # Check for significant differences between time categories
    if len(walking_data['time_category'].unique()) > 1:
        time_groups = []
        for cat in walking_data['time_category'].unique():
            time_groups.append(walking_data[walking_data['time_category'] == cat]['duration'].values)
        
        # ANOVA test
        f_stat, p_value = f_oneway(*time_groups)
        
        time_stats_dict = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'time_stats': time_stats
        }
        
        results['time_of_day_statistics'] = time_stats_dict
        
        if output_dir:
            time_stats.to_csv(stats_dir / 'walking_by_time_of_day.csv', index=False)
            
            # Create visualization
            create_time_category_visualization(walking_data, time_stats, time_stats_dict, stats_dir, language)
    
    # 6. Region transition analysis
    employee_transition_stats = {}
    
    for emp_id in walking_data['id'].unique():
        emp_data = data[data['id'] == emp_id].sort_values('startTime')
        
        # Count transitions between regions
        transitions = []
        prev_region = None
        
        for _, row in emp_data.iterrows():
            curr_region = row['region']
            if prev_region and prev_region != curr_region:
                transitions.append((prev_region, curr_region))
            prev_region = curr_region
        
        # Count unique transitions and total transitions
        unique_transitions = set(transitions)
        
        employee_transition_stats[emp_id] = {
            'total_transitions': len(transitions),
            'unique_transitions': len(unique_transitions),
            'transition_ratio': len(unique_transitions) / len(transitions) if transitions else 0
        }
    
    # Convert to DataFrame
    transition_stats_df = pd.DataFrame.from_dict(employee_transition_stats, orient='index').reset_index()
    transition_stats_df = transition_stats_df.rename(columns={'index': 'id'})
    
    # Merge with walking percentage for correlation
    transition_stats_df = pd.merge(transition_stats_df, emp_merge[['id', 'walking_percentage']], on='id')
    
    # Calculate correlation between transition metrics and walking percentage
    trans_ratio_corr, trans_ratio_p = pearsonr(
        transition_stats_df['transition_ratio'], 
        transition_stats_df['walking_percentage']
    )
    
    transition_corr = {
        'transition_ratio_correlation': trans_ratio_corr,
        'transition_ratio_p_value': trans_ratio_p,
        'significant': trans_ratio_p < 0.05
    }
    
    results['transition_statistics'] = {
        'employee_stats': transition_stats_df,
        'correlation': transition_corr
    }
    
    if output_dir:
        transition_stats_df.to_csv(stats_dir / 'transition_statistics.csv', index=False)
        
        # Create visualization
        create_transition_stats_visualization(transition_stats_df, transition_corr, stats_dir, language)
    
    # Create summary report
    if output_dir:
        create_statistical_summary_report(results, stats_dir, language)
    
    return results


def create_employee_percentage_visualization(emp_data, output_dir, language='en'):
    """
    Create visualization showing walking percentage by employee
    
    Parameters:
    -----------
    emp_data : pandas.DataFrame
        Employee walking percentage data
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Create figure with horizontal bar chart
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar chart sorted by walking percentage
    bars = plt.barh(
        emp_data['id'],
        emp_data['walking_percentage'],
        color=[employee_colors.get(emp, '#1f77b4') for emp in emp_data['id']]
    )
    
    # Add percentage and time labels
    for bar, pct, total_h, walk_h in zip(
        bars, 
        emp_data['walking_percentage'],
        emp_data['total_hours'],
        emp_data['walking_hours']
    ):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}% ({walk_h:.1f}h of {total_h:.1f}h)",
            va='center',
            fontweight='bold'
        )
    
    # Add median line
    median_pct = emp_data['walking_percentage'].median()
    plt.axvline(median_pct, color='red', linestyle='--', alpha=0.7)
    plt.text(
        median_pct + 0.5, 
        0.02, 
        f"Median: {median_pct:.1f}%", 
        transform=plt.gca().get_xaxis_transform(),
        color='red',
        fontweight='bold'
    )
    
    plt.title("Walking Percentage by Employee", fontsize=16, fontweight='bold')
    plt.xlabel("Percentage of Total Time (%)", fontsize=14)
    plt.ylabel("Employee ID", fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / 'employee_walking_percentage.png')
    plt.close()


def create_region_comparison_visualization(region_data, output_dir, language='en'):
    """
    Create visualization comparing region usage between high and low walkers
    
    Parameters:
    -----------
    region_data : pandas.DataFrame
        Region comparison data
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Get top 10 regions by total duration
    region_totals = region_data.groupby('region')['duration'].sum().sort_values(ascending=False)
    top_regions = region_totals.head(10).index.tolist()
    
    # Filter to top regions
    top_region_data = region_data[region_data['region'].isin(top_regions)]
    
    # Pivot data for side-by-side comparison
    pivot_data = top_region_data.pivot(index='region', columns='group', values='percentage')
    
    # Sort by total percentage
    pivot_data['total'] = pivot_data.sum(axis=1)
    pivot_data = pivot_data.sort_values('total', ascending=False)
    pivot_data = pivot_data.drop('total', axis=1)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    ax = pivot_data.plot(
        kind='bar',
        color=['#1f77b4', '#ff7f0e'],
        width=0.8
    )
    
    # Iterate through bars and add percentage labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontweight='bold')
    
    plt.title("Region Distribution: High vs. Low Walkers", fontsize=16, fontweight='bold')
    plt.xlabel("Region", fontsize=14)
    plt.ylabel("Percentage of Walking Time (%)", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(title="Group")
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / 'region_comparison_high_vs_low.png')
    plt.close()


def create_significant_region_visualization(region_stats, output_dir, language='en'):
    """
    Create visualization showing statistically significant differences in region usage
    
    Parameters:
    -----------
    region_stats : pandas.DataFrame
        Region statistical test results
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    # Filter to significant regions only
    sig_regions = region_stats[region_stats['significant']]
    
    if len(sig_regions) == 0:
        print("  No statistically significant differences in region usage found")
        return
    
    set_visualization_style()
    
    # Take top 10 most significant regions by mean difference
    top_sig_regions = sig_regions.head(10)
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot high and low walker means side by side
    width = 0.35
    ind = np.arange(len(top_sig_regions))
    
    plt.bar(
        ind - width/2, 
        top_sig_regions['high_mean'] / 60,  # Convert to minutes
        width,
        label='High Walkers',
        color='#1f77b4'
    )
    
    plt.bar(
        ind + width/2, 
        top_sig_regions['low_mean'] / 60,  # Convert to minutes 
        width,
        label='Low Walkers',
        color='#ff7f0e'
    )
    
    # Add region labels and p-value annotations
    plt.xticks(ind, top_sig_regions['region'], rotation=45, ha='right')
    
    for i, (_, row) in enumerate(top_sig_regions.iterrows()):
        # Calculate the height for the annotation (use the taller bar plus a bit)
        y_pos = max(row['high_mean'], row['low_mean']) / 60 + 0.5
        
        # Format p-value with scientific notation for very small values
        if row['p_value'] < 0.001:
            p_text = "p<0.001"
        else:
            p_text = f"p={row['p_value']:.3f}"
        
        plt.annotate(
            p_text,
            xy=(i, y_pos),
            ha='center',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    plt.title("Regions with Significant Differences Between Walker Types", fontsize=16, fontweight='bold')
    plt.xlabel("Region", fontsize=14)
    plt.ylabel("Average Walking Duration (minutes)", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(title="Group")
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / 'significant_region_differences.png')
    plt.close()


def create_time_category_visualization(walking_data, time_stats, time_stats_dict, output_dir, language='en'):
    """
    Create visualization comparing walking patterns by time of day
    
    Parameters:
    -----------
    walking_data : pandas.DataFrame
        Walking data with time categories
    time_stats : pandas.DataFrame
        Statistics by time category
    time_stats_dict : dict
        Dictionary with ANOVA results
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Box plot of walking durations by time category
    sns.boxplot(
        x='time_category', 
        y='duration',
        data=walking_data,
        ax=ax1,
        palette='viridis'
    )
    
    # Add ANOVA results if significant
    if time_stats_dict['significant']:
        ax1.set_title(
            f"Walking Duration by Time of Day (ANOVA: p={time_stats_dict['p_value']:.4f}, significant)",
            fontsize=16,
            fontweight='bold'
        )
    else:
        ax1.set_title(
            f"Walking Duration by Time of Day (ANOVA: p={time_stats_dict['p_value']:.4f}, not significant)",
            fontsize=16,
            fontweight='bold'
        )
    
    ax1.set_ylabel("Duration (seconds)", fontsize=14)
    ax1.set_xlabel("", fontsize=14)  # Hide x-label as it's duplicated in the subplot below
    
    # 2. Bar chart of walk counts by time category
    bars = ax2.bar(
        time_stats['time_category'],
        time_stats['walk_count'],
        color=plt.cm.viridis(np.linspace(0, 0.8, len(time_stats)))
    )
    
    # Add count labels
    for bar, count, mean_dur in zip(bars, time_stats['walk_count'], time_stats['mean_duration']):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f"{count}\n({mean_dur:.1f}s avg)",
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    ax2.set_title("Number of Walks by Time of Day", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Time of Day", fontsize=14)
    ax2.set_ylabel("Number of Walks", fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, output_dir / 'walking_by_time_of_day.png')
    plt.close()


def create_transition_stats_visualization(transition_stats, transition_corr, output_dir, language='en'):
    """
    Create visualization of transition statistics and correlation with walking percentage
    
    Parameters:
    -----------
    transition_stats : pandas.DataFrame
        Transition statistics by employee
    transition_corr : dict
        Correlation statistics
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    set_visualization_style()
    
    # Create figure with scatter plot
    plt.figure(figsize=(12, 8))
    
    # Get employee colors
    employee_colors = get_employee_colors()
    
    # Scatter plot of transition ratio vs walking percentage
    for _, row in transition_stats.iterrows():
        emp_id = row['id']
        x = row['transition_ratio']
        y = row['walking_percentage']
        
        plt.scatter(
            x, y,
            s=100,
            color=employee_colors.get(emp_id, '#1f77b4'),
            alpha=0.8,
            edgecolor='black'
        )
        
        # Add employee label
        plt.annotate(
            emp_id,
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontweight='bold'
        )
    
    # Add correlation information
    corr_text = f"Correlation: {transition_corr['transition_ratio_correlation']:.2f}"
    if transition_corr['significant']:
        corr_text += f" (p={transition_corr['transition_ratio_p_value']:.4f}, significant)"
    else:
        corr_text += f" (p={transition_corr['transition_ratio_p_value']:.4f}, not significant)"
    
    plt.figtext(
        0.5, 0.01,
        corr_text,
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )
    
    # Add trend line if correlation is significant
    if transition_corr['significant']:
        # Calculate trend line
        z = np.polyfit(transition_stats['transition_ratio'], transition_stats['walking_percentage'], 1)
        p = np.poly1d(z)
        
        # Add trend line to plot
        x_range = np.linspace(transition_stats['transition_ratio'].min(), transition_stats['transition_ratio'].max(), 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.title("Relationship Between Transition Ratio and Walking Percentage", fontsize=16, fontweight='bold')
    plt.xlabel("Transition Ratio (Unique/Total)", fontsize=14)
    plt.ylabel("Walking Percentage (%)", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), output_dir / 'transition_ratio_correlation.png')
    plt.close()


def create_statistical_summary_report(results, output_dir, language='en'):
    """
    Create a summary report of statistical findings
    
    Parameters:
    -----------
    results : dict
        Dictionary with statistical analysis results
    output_dir : Path
        Directory to save report
    language : str, optional
        Language code ('en' or 'de')
    """
    # Create a summary report in markdown format
    report_path = output_dir / 'statistical_analysis_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# Statistical Analysis of Walking Patterns\n\n")
        
        # Walking percentage summary
        f.write("## Employee Walking Percentages\n\n")
        f.write("Employee | Walking % | Walking Hours | Total Hours\n")
        f.write("--- | --- | --- | ---\n")
        
        for _, row in results['walking_percentage'].iterrows():
            f.write(f"{row['id']} | {row['walking_percentage']:.2f}% | {row['walking_hours']:.2f}h | {row['total_hours']:.2f}h\n")
        
        f.write(f"\nMedian walking percentage: {results['walking_percentage']['walking_percentage'].median():.2f}%\n\n")
        
        # Time of day analysis
        if 'time_of_day_statistics' in results:
            time_stats = results['time_of_day_statistics']
            
            f.write("## Walking Patterns by Time of Day\n\n")
            
            if time_stats['significant']:
                f.write(f"ANOVA test shows **significant differences** between time periods (p={time_stats['p_value']:.4f}).\n\n")
            else:
                f.write(f"ANOVA test shows no significant differences between time periods (p={time_stats['p_value']:.4f}).\n\n")
            
            f.write("Time Period | Mean Duration (s) | Median Duration (s) | # of Walks\n")
            f.write("--- | --- | --- | ---\n")
            
            for _, row in time_stats['time_stats'].iterrows():
                f.write(f"{row['time_category']} | {row['mean_duration']:.2f} | {row['median_duration']:.2f} | {row['walk_count']}\n")
        
        # Region comparison
        if 'region_statistics' in results:
            region_stats = results['region_statistics']
            sig_regions = region_stats[region_stats['significant']]
            
            f.write("\n## Significant Region Differences\n\n")
            
            if len(sig_regions) > 0:
                f.write(f"Found {len(sig_regions)} regions with statistically significant differences between high and low walkers.\n\n")
                
                f.write("Region | High Walker Mean (s) | Low Walker Mean (s) | p-value\n")
                f.write("--- | --- | --- | ---\n")
                
                for _, row in sig_regions.head(10).iterrows():
                    f.write(f"{row['region']} | {row['high_mean']:.2f} | {row['low_mean']:.2f} | {row['p_value']:.4f}\n")
            else:
                f.write("No statistically significant differences in region usage found.\n")
        
        # Transition statistics
        if 'transition_statistics' in results:
            trans_stats = results['transition_statistics']
            
            f.write("\n## Transition Analysis\n\n")
            
            corr = trans_stats['correlation']['transition_ratio_correlation']
            p_val = trans_stats['correlation']['transition_ratio_p_value']
            
            if trans_stats['correlation']['significant']:
                f.write(f"Found **significant correlation** between transition ratio and walking percentage (r={corr:.2f}, p={p_val:.4f}).\n\n")
                
                if corr > 0:
                    f.write("Employees with more diverse movement patterns (higher unique-to-total transition ratio) tend to have higher walking percentages.\n\n")
                else:
                    f.write("Employees with more repetitive movement patterns (lower unique-to-total transition ratio) tend to have higher walking percentages.\n\n")
            else:
                f.write(f"No significant correlation between transition ratio and walking percentage (r={corr:.2f}, p={p_val:.4f}).\n\n")
            
            f.write("Employee | Total Transitions | Unique Transitions | Transition Ratio | Walking %\n")
            f.write("--- | --- | --- | --- | ---\n")
            
            for _, row in trans_stats['employee_stats'].iterrows():
                f.write(f"{row['id']} | {row['total_transitions']} | {row['unique_transitions']} | {row['transition_ratio']:.2f} | {row['walking_percentage']:.2f}%\n")
        
        f.write("\n---\n\nReport generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"  Created statistical summary report: {report_path}")


def analyze_specific_region(data, region_name, output_dir=None, language='en'):
    """
    Perform focused analysis on a specific region
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete dataset
    region_name : str
        Name of the region to analyze
    output_dir : Path, optional
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with region analysis results
    """
    print(f"  Performing focused analysis on region: {region_name}")
    
    # Create directory for region analysis
    if output_dir:
        region_dir = ensure_dir_exists(output_dir / 'region_focus')
    
    # Check if region exists in data
    if region_name not in data['region'].unique():
        print(f"  Error: Region '{region_name}' not found in the dataset")
        return {}
    
    # Filter data for this region
    region_data = data[data['region'] == region_name]
    
    # Filter walking data for this region
    walking_data = region_data[region_data['activity'] == 'Walk']
    
    # Results dictionary
    results = {
        'region_name': region_name,
        'total_duration': region_data['duration'].sum(),
        'walking_duration': walking_data['duration'].sum() if not walking_data.empty else 0,
        'employee_count': region_data['id'].nunique(),
        'activity_counts': region_data.groupby('activity')['duration'].agg(['sum', 'count']).reset_index()
    }
    
    # Calculate percentages
    total_duration = results['total_duration']
    results['formatted_total'] = format_seconds_to_hms(total_duration)
    results['total_hours'] = total_duration / 3600
    
    if not walking_data.empty:
        results['walking_percentage'] = (results['walking_duration'] / total_duration * 100).round(2)
        results['formatted_walking'] = format_seconds_to_hms(results['walking_duration'])
        results['walking_hours'] = results['walking_duration'] / 3600
    
    # Activity breakdown
    activity_breakdown = region_data.groupby('activity')['duration'].sum().reset_index()
    activity_breakdown['percentage'] = (activity_breakdown['duration'] / total_duration * 100).round(2)
    activity_breakdown['formatted_duration'] = activity_breakdown['duration'].apply(format_seconds_to_hms)
    activity_breakdown = activity_breakdown.sort_values('duration', ascending=False)
    
    results['activity_breakdown'] = activity_breakdown
    
    # Employee analysis in this region
    employee_stats = []
    
    for emp_id in region_data['id'].unique():
        emp_region_data = region_data[region_data['id'] == emp_id]
        emp_walking = emp_region_data[emp_region_data['activity'] == 'Walk']
        
        emp_total = emp_region_data['duration'].sum()
        emp_walking_duration = emp_walking['duration'].sum() if not emp_walking.empty else 0
        
        employee_stats.append({
            'id': emp_id,
            'total_duration': emp_total,
            'walking_duration': emp_walking_duration,
            'walking_percentage': (emp_walking_duration / emp_total * 100).round(2) if emp_total > 0 else 0,
            'formatted_total': format_seconds_to_hms(emp_total),
            'formatted_walking': format_seconds_to_hms(emp_walking_duration),
            'visit_count': len(emp_region_data),
            'activities': emp_region_data['activity'].nunique()
        })
    
    # Convert to DataFrame and sort by total duration
    employee_df = pd.DataFrame(employee_stats)
    if not employee_df.empty:
        employee_df = employee_df.sort_values('total_duration', ascending=False)
        results['employee_stats'] = employee_df
    
    # Temporal analysis
    if not region_data.empty:
        region_data['hour'] = (region_data['startTime'] / 3600) % 24
        hourly_stats = region_data.groupby('hour').agg({
            'duration': 'sum',
            'id': 'nunique'
        }).reset_index()
        
        hourly_stats['percentage'] = (hourly_stats['duration'] / total_duration * 100).round(2)
        hourly_stats = hourly_stats.sort_values('hour')
        
        results['hourly_stats'] = hourly_stats
    
    # Save results to CSV if output directory is provided
    if output_dir:
        if not activity_breakdown.empty:
            activity_breakdown.to_csv(region_dir / f"{region_name}_activity_breakdown.csv", index=False)
        
        if 'employee_stats' in results and not results['employee_stats'].empty:
            results['employee_stats'].to_csv(region_dir / f"{region_name}_employee_stats.csv", index=False)
        
        if 'hourly_stats' in results and not results['hourly_stats'].empty:
            results['hourly_stats'].to_csv(region_dir / f"{region_name}_hourly_stats.csv", index=False)
        
        # Create visualizations
        create_region_focus_visualizations(results, region_dir, language)
    
    return results


def create_region_focus_visualizations(region_results, output_dir, language='en'):
    """
    Create visualizations for region focus analysis
    
    Parameters:
    -----------
    region_results : dict
        Dictionary with region analysis results
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    region_name = region_results['region_name']
    
    # 1. Activity breakdown pie chart
    if 'activity_breakdown' in region_results and not region_results['activity_breakdown'].empty:
        set_visualization_style()
        
        activity_data = region_results['activity_breakdown']
        
        plt.figure(figsize=(10, 8))
        
        # Create pie chart
        patches, texts, autotexts = plt.pie(
            activity_data['percentage'],
            labels=activity_data['activity'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Customize text
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        plt.axis('equal')
        plt.title(f"Activity Breakdown in {region_name}", fontsize=16, fontweight='bold')
        
        # Add total duration
        total_hours = region_results['total_hours']
        formatted_total = region_results['formatted_total']
        
        plt.figtext(
            0.5, 0.01,
            f"Total time: {formatted_total} ({total_hours:.1f} hours)",
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
        
        # Save figure
        save_figure(plt.gcf(), output_dir / f"{region_name}_activity_breakdown.png")
        plt.close()
    
    # 2. Employee comparison
    if 'employee_stats' in region_results and not region_results['employee_stats'].empty:
        employee_data = region_results['employee_stats']
        
        set_visualization_style()
        
        # Get employee colors
        employee_colors = get_employee_colors()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(
            employee_data['id'],
            employee_data['total_duration'] / 60,  # Convert to minutes
            color=[employee_colors.get(emp, '#1f77b4') for emp in employee_data['id']]
        )
        
        # Add labels
        for bar, total, walking, visit_count in zip(
            bars, 
            employee_data['formatted_total'],
            employee_data['formatted_walking'],
            employee_data['visit_count']
        ):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{total} (Walk: {walking})\n{visit_count} visits",
                va='center',
                fontweight='bold'
            )
        
        plt.title(f"Time Spent in {region_name} by Employee", fontsize=16, fontweight='bold')
        plt.xlabel("Duration (minutes)", fontsize=14)
        plt.ylabel("Employee ID", fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), output_dir / f"{region_name}_employee_comparison.png")
        plt.close()
    
    # 3. Hourly pattern
    if 'hourly_stats' in region_results and not region_results['hourly_stats'].empty:
        hourly_data = region_results['hourly_stats']
        
        set_visualization_style()
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create bar chart for duration
        plt.bar(
            hourly_data['hour'],
            hourly_data['duration'] / 60,  # Convert to minutes
            color=plt.cm.viridis(np.linspace(0, 0.8, 24)),
            alpha=0.7
        )
        
        # Overlay line for employee count
        plt.twinx()
        plt.plot(
            hourly_data['hour'],
            hourly_data['id'],
            'r-',
            linewidth=2,
            marker='o',
            markersize=6
        )
        
        plt.title(f"Activity in {region_name} by Hour of Day", fontsize=16, fontweight='bold')
        plt.xlabel("Hour of Day", fontsize=14)
        plt.ylabel("Number of Employees", fontsize=14, color='r')
        plt.grid(True, alpha=0.3)
        
        # First y-axis label
        plt.gca().get_yaxis().set_label_coords(1.1, 0.5)
        plt.gca().get_shared_x_axes().join(plt.gca(), plt.gcf().axes[0])
        plt.gcf().axes[0].set_ylabel("Duration (minutes)", fontsize=14)
        
        # Set x-ticks to whole hours
        plt.xticks(range(24))
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), output_dir / f"{region_name}_hourly_pattern.png")
        plt.close()


def compare_employees(data, walking_data, employees, output_dir=None, language='en'):
    """
    Perform direct comparison between specific employees
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Complete dataset
    walking_data : pandas.DataFrame
        Filtered walking data
    employees : list
        List of employee IDs to compare
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    dict
        Dictionary with comparison results
    """
    print(f"  Comparing employees: {', '.join(employees)}")
    
    # Create directory for employee comparison
    if output_dir:
        compare_dir = ensure_dir_exists(output_dir / 'employee_comparison')
    
    # Results dictionary
    results = {}
    
    # Filter data for selected employees
    filtered_data = data[data['id'].isin(employees)]
    filtered_walking = walking_data[walking_data['id'].isin(employees)]
    
    if filtered_data.empty:
        print("  No data found for specified employees")
        return {}
    
    # 1. Overall activity comparison
    activity_comparison = []
    
    for emp_id in employees:
        emp_data = data[data['id'] == emp_id]
        
        if emp_data.empty:
            print(f"  Warning: No data found for employee {emp_id}")
            continue
        
        # Calculate activity breakdown
        emp_activities = emp_data.groupby('activity')['duration'].sum().reset_index()
        total_duration = emp_activities['duration'].sum()
        
        for _, row in emp_activities.iterrows():
            activity_comparison.append({
                'id': emp_id,
                'activity': row['activity'],
                'duration': row['duration'],
                'percentage': (row['duration'] / total_duration * 100).round(2),
                'formatted_duration': format_seconds_to_hms(row['duration'])
            })
    
    # Convert to DataFrame
    activity_df = pd.DataFrame(activity_comparison)
    results['activity_comparison'] = activity_df
    
    # 2. Region comparison
    region_comparison = []
    
    for emp_id in employees:
        emp_data = data[data['id'] == emp_id]
        
        if emp_data.empty:
            continue
        
        # Calculate region breakdown
        emp_regions = emp_data.groupby('region')['duration'].sum().reset_index()
        total_duration = emp_regions['duration'].sum()
        
        for _, row in emp_regions.iterrows():
            region_comparison.append({
                'id': emp_id,
                'region': row['region'],
                'duration': row['duration'],
                'percentage': (row['duration'] / total_duration * 100).round(2),
                'formatted_duration': format_seconds_to_hms(row['duration'])
            })
    
    # Convert to DataFrame
    region_df = pd.DataFrame(region_comparison)
    results['region_comparison'] = region_df
    
    # 3. Walking pattern comparison
    walking_patterns = []
    
    for emp_id in employees:
        emp_walking = walking_data[walking_data['id'] == emp_id]
        
        if emp_walking.empty:
            continue
        
        # Calculate walking stats
        total_walking = emp_walking['duration'].sum()
        walk_count = len(emp_walking)
        
        # Add hour of day for temporal analysis
        emp_walking['hour'] = (emp_walking['startTime'] / 3600) % 24
        
        # Calculate hourly distribution
        hourly = emp_walking.groupby('hour')['duration'].sum().reset_index()
        
        walking_patterns.append({
            'id': emp_id,
            'total_walking': total_walking,
            'formatted_walking': format_seconds_to_hms(total_walking),
            'walking_hours': total_walking / 3600,
            'walk_count': walk_count,
            'avg_duration': total_walking / walk_count if walk_count > 0 else 0,
            'hourly_distribution': hourly
        })
    
    results['walking_patterns'] = walking_patterns
    
    # 4. Calculate time spent in the same regions
    if len(employees) > 1:
        common_regions = set()
        
        for emp_id in employees:
            emp_regions = set(data[data['id'] == emp_id]['region'].unique())
            if not common_regions:
                common_regions = emp_regions
            else:
                common_regions = common_regions.intersection(emp_regions)
        
        results['common_regions'] = list(common_regions)
        
        # Calculate time in common regions
        common_region_data = []
        
        for region in common_regions:
            region_stats = []
            
            for emp_id in employees:
                emp_region_data = data[(data['id'] == emp_id) & (data['region'] == region)]
                
                if not emp_region_data.empty:
                    total_time = emp_region_data['duration'].sum()
                    
                    region_stats.append({
                        'id': emp_id,
                        'duration': total_time,
                        'formatted_duration': format_seconds_to_hms(total_time)
                    })
            
            if region_stats:
                common_region_data.append({
                    'region': region,
                    'employee_stats': region_stats
                })
        
        results['common_region_data'] = common_region_data
    
    # Save results to CSV if output directory is provided
    if output_dir:
        if not activity_df.empty:
            activity_df.to_csv(compare_dir / 'activity_comparison.csv', index=False)
        
        if not region_df.empty:
            region_df.to_csv(compare_dir / 'region_comparison.csv', index=False)
        
        # Create visualizations
        create_employee_comparison_visualizations(results, compare_dir, language)
    
    return results


def create_employee_comparison_visualizations(results, output_dir, language='en'):
    """
    Create visualizations for employee comparison
    
    Parameters:
    -----------
    results : dict
        Dictionary with comparison results
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    # 1. Activity comparison visualization
    if 'activity_comparison' in results and not results['activity_comparison'].empty:
        activity_data = results['activity_comparison']
        
        set_visualization_style()
        
        # Create pivot table
        pivot_data = activity_data.pivot_table(
            index='activity',
            columns='id',
            values='percentage',
            fill_value=0
        )
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        ax = pivot_data.plot(kind='bar', figsize=(12, 8))
        
        # Iterate through bars and add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', fontweight='bold')
        
        plt.title("Activity Distribution by Employee", fontsize=16, fontweight='bold')
        plt.xlabel("Activity", fontsize=14)
        plt.ylabel("Percentage (%)", fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(title="Employee")
        
        plt.tight_layout()
        
        # Save figure
        save_figure(plt.gcf(), output_dir / 'activity_comparison.png')
        plt.close()
    
    # 2. Top regions comparison
    if 'region_comparison' in results and not results['region_comparison'].empty:
        region_data = results['region_comparison']
        
        # Get top 10 regions by total time across all employees
        region_totals = region_data.groupby('region')['duration'].sum().sort_values(ascending=False)
        top_regions = region_totals.head(10).index.tolist()
        
        # Filter to top regions
        top_region_data = region_data[region_data['region'].isin(top_regions)]
        
        if not top_region_data.empty:
            set_visualization_style()
            
            # Create pivot table
            pivot_data = top_region_data.pivot_table(
                index='region',
                columns='id',
                values='percentage',
                fill_value=0
            )
            
            # Sort by total percentage
            pivot_data['total'] = pivot_data.sum(axis=1)
            pivot_data = pivot_data.sort_values('total', ascending=False)
            pivot_data = pivot_data.drop('total', axis=1)
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Create grouped bar chart
            ax = pivot_data.plot(kind='bar', figsize=(14, 10))
            
            # Iterate through bars and add percentage labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%')
            
            plt.title("Top Regions by Employee", fontsize=16, fontweight='bold')
            plt.xlabel("Region", fontsize=14)
            plt.ylabel("Percentage of Employee's Time (%)", fontsize=14)
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend(title="Employee")
            
            plt.tight_layout()
            
            # Save figure
            save_figure(plt.gcf(), output_dir / 'top_regions_comparison.png')
            plt.close()
    
    # 3. Walking pattern comparison
    if 'walking_patterns' in results and results['walking_patterns']:
        walking_data = results['walking_patterns']
        
        set_visualization_style()
        
        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 2]})
        
        # Get employee colors
        employee_colors = get_employee_colors()
        
        # 1. Bar chart of total walking time
        employee_ids = [w['id'] for w in walking_data]
        walking_hours = [w['walking_hours'] for w in walking_data]
        walk_counts = [w['walk_count'] for w in walking_data]
        
        bars = ax1.bar(
            employee_ids,
            walking_hours,
            color=[employee_colors.get(emp, '#1f77b4') for emp in employee_ids]
        )
        
        # Add labels
        for bar, hours, count, formatted_time in zip(
            bars, 
            walking_hours,
            walk_counts,
            [w['formatted_walking'] for w in walking_data]
        ):
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{formatted_time} ({hours:.1f}h)\n{count} walks",
                ha='center',
                fontweight='bold'
            )
        
        ax1.set_title("Total Walking Time by Employee", fontsize=16, fontweight='bold')
        ax1.set_ylabel("Hours", fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Hourly distribution comparison
        for employee_data in walking_data:
            emp_id = employee_data['id']
            hourly = employee_data['hourly_distribution']
            
            # Create full time series with all hours
            full_hours = pd.DataFrame({'hour': range(24)})
            full_hourly = pd.merge(full_hours, hourly, on='hour', how='left')
            full_hourly = full_hourly.fillna(0)
            
            # Convert to minutes for better readability
            ax2.plot(
                full_hourly['hour'],
                full_hourly['duration'] / 60,
                '-',
                label=emp_id,
                linewidth=2.5,
                color=employee_colors.get(emp_id, None),
                marker='o',
                markersize=6
            )
        
        ax2.set_title("Walking Pattern Throughout the Day", fontsize=16, fontweight='bold')
        ax2.set_xlabel("Hour of Day", fontsize=14)
        ax2.set_ylabel("Minutes Spent Walking", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Employee")
        
        # Set x-ticks to whole hours
        ax2.set_xticks(range(24))
        
        plt.tight_layout()
        
        # Save figure
        save_figure(fig, output_dir / 'walking_pattern_comparison.png')
        plt.close()
    
    # 4. Common regions comparison
    if 'common_region_data' in results and results['common_region_data']:
        common_regions = results['common_region_data']
        
        # Extract data for visualization
        region_names = []
        employee_data = {}
        
        for region_info in common_regions:
            region_name = region_info['region']
            region_names.append(region_name)
            
            for stat in region_info['employee_stats']:
                emp_id = stat['id']
                if emp_id not in employee_data:
                    employee_data[emp_id] = {}
                
                employee_data[emp_id][region_name] = stat['duration'] / 60  # Convert to minutes
        
        if region_names and employee_data:
            set_visualization_style()
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Set up bar positions
            bar_width = 0.8 / len(employee_data)
            x = np.arange(len(region_names))
            
            # Get employee colors
            employee_colors = get_employee_colors()
            
            # Plot bars for each employee
            for i, (emp_id, regions) in enumerate(employee_data.items()):
                # Get values for this employee across all common regions
                values = [regions.get(region, 0) for region in region_names]
                
                # Calculate bar positions
                offset = bar_width * i - bar_width * (len(employee_data) - 1) / 2
                
                # Plot bars
                bars = plt.bar(
                    x + offset,
                    values,
                    bar_width,
                    label=emp_id,
                    color=employee_colors.get(emp_id, None)
                )
                
                # Add value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        plt.text(
                            bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.5,
                            f"{val:.1f}m",
                            ha='center',
                            fontsize=8,
                            fontweight='bold'
                        )
            
            plt.title("Time Spent in Common Regions", fontsize=16, fontweight='bold')
            plt.xlabel("Region", fontsize=14)
            plt.ylabel("Minutes", fontsize=14)
            plt.xticks(x, region_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend(title="Employee")
            
            plt.tight_layout()
            
            # Save figure
            save_figure(plt.gcf(), output_dir / 'common_regions_comparison.png')
            plt.close()


def main():
    """Main function to orchestrate the analysis"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine language based on --german flag
    language = 'de' if args.german else 'en'
    
    # Validate time slot input
    if args.time_slot <= 0 or args.time_slot > 120:
        print("Error: Time slot size must be between 1 and 120 minutes")
        return 1
    
    # Check if 24 hours is divisible by the time slot to avoid partial slots
    if 24 * 60 % args.time_slot != 0:
        print(f"Warning: 24 hours is not evenly divisible by {args.time_slot} minutes.")
        print(f"Consider using one of these values: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, or 120 minutes.")
    
    # Check required files
    required_files = {
        'Data file': args.data
    }
    
    files_exist, missing_files = check_required_files(required_files)
    if not files_exist:
        print("The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return 1
    
    print("=" * 80)
    print("=== Enhanced Bakery Walking Pattern Analysis ===")
    print("=" * 80)
    
    # Create output directory structure
    output_path = ensure_dir_exists(args.output)
    walking_dir = ensure_dir_exists(output_path / 'walking_analysis')
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = load_sensor_data(args.data)
    
    # Add department classification if the function exists
    try:
        data = classify_departments(data)
        print("Added department classification")
    except Exception as e:
        print(f"Note: Could not classify departments: {e}")
    
    # Filter by employee if specified
    if args.employee:
        if args.employee not in data['id'].unique():
            print(f"Error: Employee {args.employee} not found in the data")
            return 1
        
        data = data[data['id'] == args.employee]
        print(f"Filtered data for employee {args.employee}")
    
    # Filter walking data once for all analyses
    walking_data = data[data['activity'] == 'Walk']
    if walking_data.empty:
        print("Warning: No walking data found in the dataset.")
    
    # Determine time ranges for each employee
    time_ranges = determine_employee_time_range(data)
    
    # Print time range info
    print("\nDetected time ranges for employees:")
    for emp_id, range_info in time_ranges.items():
        print(f"  {emp_id}: {range_info['formatted_start']} to {range_info['formatted_end']} "
              f"({range_info['total_hours']:.1f} hours)")
    
    # Run the appropriate analyses based on flags
    
    # Basic walking analysis
    if args.walk or (not any([args.walk, args.region_focus, args.repetitive_patterns, 
                             args.statistical, args.compare_employees]) and not args.region):
        print("\n" + "=" * 40)
        print(f"=== Walking Pattern Analysis ({args.time_slot}-minute intervals) ===")
        
        print(f"Analyzing walking patterns throughout the day...")
        walking_results = analyze_walking_patterns(data, walking_dir, args.time_slot, language)
        
        print(f"Analyzing walking patterns by shift...")
        if 'shift' in data.columns:
            shift_results = analyze_walking_by_shift(walking_data, walking_dir, args.time_slot, language)
            print(f"Created shift comparison visualizations for {len(shift_results) if shift_results else 0} employees")
        else:
            print("Shift information not found in dataset - skipping shift analysis")

        print(f"\nSaved walking analysis results to {walking_dir}/")
        print(f"Timeline visualizations saved to {walking_dir}/timeline/")
        print(f"Shift comparison visualizations saved to {walking_dir}/shifts/")
    
    # Region focus analysis
    if args.region_focus:
        print("\n" + "=" * 40)
        print("=== Region Focus Analysis ===")
        
        region_results = analyze_region_walking_patterns(data, walking_data, walking_dir, language)
        print(f"Generated region walking pattern analysis with {len(region_results['region_stats']) if 'region_stats' in region_results else 0} regions")
        print(f"Saved region analysis to {walking_dir}/regions/")
    
    # Specific region analysis
    if args.region:
        print("\n" + "=" * 40)
        print(f"=== Specific Region Analysis: {args.region} ===")
        
        region_results = analyze_specific_region(data, args.region, walking_dir, language)
        if region_results:
            print(f"Generated detailed analysis for region {args.region}")
            print(f"Saved region focus analysis to {walking_dir}/region_focus/")
    
    # Repetitive pattern analysis
    if args.repetitive_patterns:
        print("\n" + "=" * 40)
        print("=== Repetitive Pattern Analysis ===")
        
        pattern_results = detect_repetitive_walking(walking_data, walking_dir, language)
        print(f"Detected {len(pattern_results['patterns']) if 'patterns' in pattern_results and not pattern_results['patterns'].empty else 0} repetitive walking patterns")
        print(f"Saved pattern analysis to {walking_dir}/patterns/")
    
    # Statistical analysis
    if args.statistical:
        print("\n" + "=" * 40)
        print("=== Statistical Analysis ===")
        
        stat_results = perform_statistical_analysis(data, walking_data, walking_dir, language)
        print(f"Performed detailed statistical analysis of walking patterns")
        print(f"Saved statistical analysis to {walking_dir}/statistics/")
    
    # Employee comparison
    if args.compare_employees:
        print("\n" + "=" * 40)
        print("=== Employee Comparison ===")
        
        # Get all employees if none specified
        if args.employee:
            employees = [args.employee]
            # Add another employee for comparison
            all_employees = sorted(data['id'].unique())
            for emp in all_employees:
                if emp != args.employee:
                    employees.append(emp)
                    break
        else:
            employees = sorted(data['id'].unique())
        
        if len(employees) > 1:
            comp_results = compare_employees(data, walking_data, employees, walking_dir, language)
            print(f"Compared walking patterns for {len(employees)} employees")
            print(f"Saved comparison results to {walking_dir}/employee_comparison/")
        else:
            print("Need multiple employees for comparison - skipping")
    
    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "=" * 40)
    print("=== Analysis Complete ===")
    print("=" * 40)
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"\nResults saved to: {output_path}")
    
    # Display key visualizations to check
    print("\nKey visualizations to check:")
    
    if args.walk or (not any([args.walk, args.region_focus, args.repetitive_patterns, 
                             args.statistical, args.compare_employees]) and not args.region):
        print(f"1. {walking_dir}/timeline/all_employees_walking_patterns_{args.time_slot}min.png - Combined walking patterns")
        
        employees = data['id'].unique()
        if len(employees) <= 5:  # Only list individual files if there aren't too many
            for emp_id in employees:
                print(f"2. {walking_dir}/timeline/employee_{emp_id}_walking_pattern_{args.time_slot}min.png - Walking pattern for employee {emp_id}")
        else:
            print(f"2. {walking_dir}/timeline/ - Individual employee walking patterns")
    
    if args.region_focus:
        print(f"3. {walking_dir}/regions/top_walking_regions.png - Top regions by walking time")
        print(f"4. {walking_dir}/regions/top_walking_transitions.png - Top walking transitions")
    
    if args.repetitive_patterns:
        print(f"5. {walking_dir}/patterns/pattern_summary.png - Summary of repetitive patterns")
    
    if args.statistical:
        print(f"6. {walking_dir}/statistics/employee_walking_percentage.png - Walking percentages by employee")
        print(f"7. {walking_dir}/statistics/region_comparison_high_vs_low.png - Region usage differences")
        print(f"8. {walking_dir}/statistics/statistical_analysis_summary.md - Detailed statistical findings")
    
    if args.region:
        print(f"9. {walking_dir}/region_focus/{args.region}_activity_breakdown.png - Activity breakdown in specific region")
        print(f"10. {walking_dir}/region_focus/{args.region}_employee_comparison.png - Employee comparison in specific region")
    
    if args.compare_employees:
        print(f"11. {walking_dir}/employee_comparison/walking_pattern_comparison.png - Employee walking pattern comparison")
        print(f"12. {walking_dir}/employee_comparison/top_regions_comparison.png - Top regions by employee")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())