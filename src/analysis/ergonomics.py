"""
Ergonomics Analysis Module

Analyzes ergonomic patterns and generates employee ergonomic scores with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Import visualization utilities from base
from src.visualization.base import (
    save_figure, set_visualization_style, 
    get_activity_colors, get_employee_colors
)

def create_region_clusters():
    """Group similar regions into functional clusters"""
    return {
        'Bread_Stations': ['1_Brotstation', '2_Brotstation', '3_Brotstation', '4_Brotstation'],
        'Pastry_Stations': ['1_Konditorei_station', '2_Konditorei_station', '3_Konditorei_station', 
                           'konditorei_deco', 'konditorei_deco_1', 'konditorei_deco_2'],
        'Ovens': ['Etagenofen', '1_Stickenofen', '2_Stickenofen', 'Gärraum', 'Fettbacken', 
                 'g\u00e4rraum_handle', 'etagenofen'],
        'Storage': ['Lager', '1_lager', '2_lager', '3_lager', 'Silo', 'Lager_eingang', '2_lager_eingang'],
        'Hallways': ['flur_staircase_up', 'flur_eingang_umkleidung_WC', '1_corridor', '2_flur_aufzug', 
                    '1_flur_aufzug', '1_corridor_a', '1_corridor_b', 'flur_staircase']
    }

def analyze_activity_durations(data):
    """
    Analyze the distribution of activity durations to establish data-driven thresholds
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    
    Returns:
    --------
    dict
        Dictionary with activity duration statistics
    """
    # Analyze durations by activity type
    activity_stats = {}
    
    for activity in data['activity'].unique():
        act_data = data[data['activity'] == activity]['duration']
        
        # Calculate percentiles
        percentiles = np.percentile(act_data, [25, 50, 75, 90, 95])
        
        activity_stats[activity] = {
            'count': len(act_data),
            'mean': act_data.mean(),
            'median': act_data.median(),
            'p25': percentiles[0],
            'p50': percentiles[1],
            'p75': percentiles[2],
            'p90': percentiles[3],
            'p95': percentiles[4],
            'min': act_data.min(),
            'max': act_data.max()
        }
    
    return activity_stats

def calculate_ergonomic_score(data, min_duration=5):
    """
    Calculate ergonomic scores out of 100 for each employee
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    min_duration : int, optional
        Minimum duration in seconds to filter out potential false readings
    
    Returns:
    --------
    dict
        Dictionary containing ergonomic scores and detailed factor breakdown
    """
    # First analyze duration distributions to set data-driven thresholds
    activity_stats = analyze_activity_durations(data)
    
    # Filter out very short activities that might be false readings
    filtered_data = data[data['duration'] >= min_duration]
    
    # Starting score is 100, deductions will be made for ergonomic issues
    base_score = 100
    
    # Define ergonomic deduction factors based on position types
    position_deductions = {
        'Handle up': 2.0,     # High deduction for overhead work
        'Handle down': 1.5,   # Medium-high deduction for low level work 
        'Handle center': 0.2, # Small deduction for neutral position
        'Stand': 0.5,         # Medium deduction for static standing
        'Walk': 0.3           # Small deduction for excessive walking
    }
    
    # Get median durations for scaling
    if 'Handle up' in activity_stats:
        handle_up_median = activity_stats['Handle up']['median']
    else:
        handle_up_median = 60  # Default if no data
    
    # Define duration factor scaling based on the data
    def duration_factor(activity, seconds):
        if activity == 'Handle up':
            # Scale based on percentage of median duration
            if seconds < handle_up_median * 0.5:
                return 0.5  # Short durations
            elif seconds < handle_up_median:
                return 1.0  # Medium durations
            elif seconds < handle_up_median * 2:
                return 1.5  # Long durations
            else:
                return 2.0  # Very long durations
        elif activity == 'Stand':
            # Static standing becomes problematic after longer durations
            if seconds < 60:
                return 0.5  # Short standing is less problematic
            elif seconds < 300:
                return 1.0  # 1-5 minutes
            elif seconds < 900:
                return 1.5  # 5-15 minutes
            else:
                return 2.0  # > 15 minutes
        else:
            # Default scaling for other activities
            if seconds < 60:
                return 0.7
            elif seconds < 300:
                return 1.0
            elif seconds < 600:
                return 1.3
            else:
                return 1.5
    
    # Add region cluster column
    region_mapping = create_region_clusters()
    filtered_data['region_cluster'] = filtered_data['region'].apply(
        lambda r: next((cluster for cluster, regions in region_mapping.items() 
                      if r in regions), 'Other'))
    
    # Calculate employee-level ergonomic scores
    employee_scores = []
    
    for emp_id, emp_data in filtered_data.groupby('id'):
        total_duration = emp_data['duration'].sum()
        
        # Initialize score components
        score_components = {
            'base_score': base_score,
            'deductions': {},
            'final_score': 0,
            'activity_breakdown': {},
            'region_breakdown': {},
            'duration_factors': {}
        }
        
        # Calculate deductions by activity
        for activity in position_deductions.keys():
            act_data = emp_data[emp_data['activity'] == activity]
            if act_data.empty:
                continue
                
            activity_duration = act_data['duration'].sum()
            activity_percentage = (activity_duration / total_duration) * 100
            
            # Calculate weighted deduction based on duration
            weighted_deduction = 0
            for _, row in act_data.iterrows():
                dur_factor = duration_factor(activity, row['duration'])
                deduction = position_deductions[activity] * dur_factor
                weighted_deduction += (row['duration'] / activity_duration) * deduction
            
            # Total deduction is proportional to percentage of time in this activity
            total_deduction = weighted_deduction * (activity_percentage / 100) * 10
            score_components['deductions'][activity] = total_deduction
            
            # Store activity breakdown
            score_components['activity_breakdown'][activity] = {
                'percentage': activity_percentage,
                'duration': activity_duration,
                'deduction': total_deduction
            }
            
            # Store duration factors
            duration_buckets = act_data.groupby(
                lambda idx: duration_factor(activity, act_data.loc[idx, 'duration']))['duration'].sum()
            
            score_components['duration_factors'][activity] = {
                str(factor): {
                    'duration': dur,
                    'percentage': (dur / activity_duration) * 100
                } for factor, dur in duration_buckets.items()
            }
            
        # Calculate region-specific deductions
        region_deductions = {}
        for region, region_data in emp_data.groupby('region'):
            region_duration = region_data['duration'].sum()
            region_percentage = (region_duration / total_duration) * 100
            
            # Calculate weighted deduction for this region
            region_deduction = 0
            for activity, act_data in region_data.groupby('activity'):
                if activity not in position_deductions:
                    continue
                    
                activity_duration = act_data['duration'].sum()
                for _, row in act_data.iterrows():
                    dur_factor = duration_factor(activity, row['duration'])
                    deduction = position_deductions[activity] * dur_factor
                    region_deduction += (row['duration'] / region_duration) * deduction
            
            # Store region breakdown
            region_deductions[region] = region_deduction * (region_percentage / 100)
            score_components['region_breakdown'][region] = {
                'percentage': region_percentage,
                'duration': region_duration,
                'deduction': region_deductions[region]
            }
        
        # Calculate final score
        total_deduction = sum(score_components['deductions'].values())
        final_score = max(base_score - total_deduction, 0)
        score_components['final_score'] = final_score
        
        employee_scores.append({
            'employee_id': emp_id,
            'final_score': final_score,
            'total_duration': total_duration,
            'components': score_components
        })
    
    return employee_scores

def export_duration_factor_thresholds(data, output_dir):
    """
    Export the duration factor thresholds used in ergonomic scoring
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path
        Directory to save the threshold information
    
    Returns:
    --------
    None
    """
    activity_stats = analyze_activity_durations(data)
    output_path = Path(output_dir)
    
    with open(output_path / 'duration_factor_thresholds.txt', 'w') as f:
        f.write("Duration Factor Thresholds Used in Ergonomic Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        # Handle up thresholds
        if 'Handle up' in activity_stats:
            handle_up_median = activity_stats['Handle up']['median']
            f.write("Handle up activity:\n")
            f.write(f"  Median duration: {handle_up_median:.1f} seconds\n")
            f.write(f"  Short  (factor 0.5): < {handle_up_median * 0.5:.1f} seconds\n")
            f.write(f"  Medium (factor 1.0): < {handle_up_median:.1f} seconds\n")
            f.write(f"  Long   (factor 1.5): < {handle_up_median * 2:.1f} seconds\n")
            f.write(f"  Very long (factor 2.0): >= {handle_up_median * 2:.1f} seconds\n\n")
        
        # Stand thresholds
        f.write("Stand activity:\n")
        f.write("  Short  (factor 0.5): < 60 seconds\n")
        f.write("  Medium (factor 1.0): < 300 seconds (5 minutes)\n")
        f.write("  Long   (factor 1.5): < 900 seconds (15 minutes)\n")
        f.write("  Very long (factor 2.0): >= 900 seconds\n\n")
        
        # Default thresholds
        f.write("Default scaling for other activities:\n")
        f.write("  Very short (factor 0.7): < 60 seconds\n")
        f.write("  Short      (factor 1.0): < 300 seconds (5 minutes)\n")
        f.write("  Medium     (factor 1.3): < 600 seconds (10 minutes)\n")
        f.write("  Long       (factor 1.5): >= 600 seconds\n")

def generate_ergonomic_report(employee_scores, output_dir):
    """
    Generate detailed ergonomic report visualizations for each employee
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries from calculate_ergonomic_score
    output_dir : Path
        Directory to save the reports
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard activity and employee colors
    activity_colors = get_activity_colors()
    employee_colors = get_employee_colors()
    
    # Calculate benchmark values for comparison
    avg_score = np.mean([emp['final_score'] for emp in employee_scores])
    max_score = np.max([emp['final_score'] for emp in employee_scores])
    min_score = np.min([emp['final_score'] for emp in employee_scores])
    
    # Generate report for each employee
    for emp_data in employee_scores:
        emp_id = emp_data['employee_id']
        score = emp_data['final_score']
        components = emp_data['components']
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. Main score display
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.axis('off')
        
        # Get employee-specific color
        emp_color = employee_colors.get(emp_id, '#333333')
        
        # Display the score prominently
        score_color = 'green' if score >= 75 else ('orange' if score >= 60 else 'red')
        ax_main.text(0.5, 0.7, f"Ergonomic Score: {score:.1f}/100", 
                    fontsize=28, ha='center', color=score_color, fontweight='bold')
        
        # Add comparison to other employees
        comparison_text = f"Average: {avg_score:.1f} | Min: {min_score:.1f} | Max: {max_score:.1f}"
        ax_main.text(0.5, 0.5, comparison_text, fontsize=14, ha='center')
        
        # Add employee ID and total duration
        hours = emp_data['total_duration'] / 3600
        ax_main.text(0.5, 0.3, f"Employee: {emp_id} | Hours Analyzed: {hours:.1f}", 
                    fontsize=16, ha='center', fontweight='bold', color=emp_color)
        
        # 2. Deductions breakdown
        ax_deduct = fig.add_subplot(gs[1, 0:2])
        
        # Extract deductions
        deductions = components['deductions']
        activities = list(deductions.keys())
        values = list(deductions.values())
        
        # Sort by deduction value
        sorted_idx = np.argsort(values)
        activities = [activities[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Color bars based on deduction severity
        colors = ['green' if v < 3 else ('orange' if v < 7 else 'red') for v in values]
        
        # Create horizontal bar chart
        bars = ax_deduct.barh(activities, values, color=colors)
        ax_deduct.set_xlabel('Deduction Points')
        ax_deduct.set_title('Ergonomic Deductions by Activity', fontsize=14)
        
        # Add labels
        for bar in bars:
            width = bar.get_width()
            ax_deduct.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                          f"{width:.1f}", va='center')
        
        # 3. Activity time breakdown
        ax_activity = fig.add_subplot(gs[1, 2])
        
        # Extract activity percentages
        act_breakdown = components['activity_breakdown']
        activities = list(act_breakdown.keys())
        percentages = [act_breakdown[act]['percentage'] for act in activities]
        
        # Use standard activity colors
        colors = [activity_colors.get(act, '#CCCCCC') for act in activities]
        
        ax_activity.pie(percentages, labels=activities, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        ax_activity.set_title('Activity Time Distribution', fontsize=14)
        
        # 4. Region breakdown
        ax_region = fig.add_subplot(gs[2, 0])
        
        # Extract region data
        region_breakdown = components['region_breakdown']
        top_regions = sorted(region_breakdown.items(), 
                           key=lambda x: x[1]['duration'], reverse=True)[:5]
        
        region_names = [r[0] for r in top_regions]
        region_pcts = [r[1]['percentage'] for r in top_regions]
        
        # Create bar chart
        ax_region.bar(region_names, region_pcts)
        ax_region.set_ylabel('% of Total Time')
        ax_region.set_title('Top 5 Regions by Time Spent', fontsize=14)
        plt.setp(ax_region.get_xticklabels(), rotation=45, ha='right')
        
        # 5. Recommendations
        ax_rec = fig.add_subplot(gs[2, 1:])
        ax_rec.axis('off')
        
        # Generate recommendations based on deductions
        recommendations = []
        
        if 'Handle up' in deductions and deductions['Handle up'] > 5:
            recommendations.append("• Reduce overhead work or use platforms/tools to minimize arm elevation")
            
        if 'Handle down' in deductions and deductions['Handle down'] > 5:
            recommendations.append("• Raise working surfaces to reduce bending and stooping postures")
            
        if 'Stand' in deductions and deductions['Stand'] > 3:
            recommendations.append("• Use anti-fatigue mats and alternate between sitting and standing")
            
        if len(recommendations) == 0:
            recommendations.append("• Current ergonomic practices are good - maintain awareness of posture")
            recommendations.append("• Take short breaks to stretch throughout shift")
        
        ax_rec.text(0.05, 0.95, "Recommendations for Improvement:", fontsize=14, fontweight='bold')
        y_pos = 0.85
        for rec in recommendations:
            ax_rec.text(0.05, y_pos, rec, fontsize=12)
            y_pos -= 0.1
        
        # Add explanatory text
        ax_rec.text(0.05, 0.3, 
                   "Scoring Methodology:", fontsize=12, fontweight='bold')
        ax_rec.text(0.05, 0.2, 
                   "Starting from 100 points, deductions are made based on time spent in risky postures.", 
                   fontsize=10)
        ax_rec.text(0.05, 0.1, 
                   "Overhead work (Handle up) and low-level work (Handle down) incur the largest deductions.", 
                   fontsize=10)
        
        # Add title and save
        plt.suptitle(f"Ergonomic Assessment Report - Employee {emp_id}", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Use the save_figure utility from base
        save_figure(fig, output_path / f"{emp_id}_ergonomic_report.png")