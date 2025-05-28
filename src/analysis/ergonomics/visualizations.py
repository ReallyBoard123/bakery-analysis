"""
Ergonomics Visualizations Module

Contains functions for creating visualizations of ergonomic analysis results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

from src.utils.time_utils import format_seconds_to_hms
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import (
    save_figure, set_visualization_style, 
    get_activity_colors, get_activity_order, 
    get_employee_colors, get_text
)

def create_handling_position_visualizations(data, region_categories, output_dir, language='en'):
    """
    Create visualizations for handling position analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    region_categories : dict
        Dictionary mapping categories to regions
    output_dir : Path
        Directory to save visualizations
    language : str, optional
        Language code ('en' or 'de')
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print(f"{get_text('Matplotlib or Seaborn not available, skipping visualizations', language)}")
        return
    
    # Ensure visualization directory exists
    vis_dir = ensure_dir_exists(output_dir / 'visualizations' / 'handling_analysis')
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # Get bread and cake workstation data
    bread_regions = region_categories['Bread Workstation']
    cake_regions = region_categories['Cake Workstation']
    
    # Calculate workstation data
    workstation_summary = []
    
    # For each workstation category
    for category_name, regions in region_categories.items():
        category_data = data[data['region'].isin(regions) & data['activity'].isin(handling_categories)]
        
        if category_data.empty:
            continue
            
        total_duration = category_data['duration'].sum()
        
        # Calculate percentages by activity
        for activity in handling_categories:
            act_data = category_data[category_data['activity'] == activity]
            act_duration = act_data['duration'].sum()
            percentage = (act_duration / total_duration * 100) if total_duration > 0 else 0
            
            workstation_summary.append({
                'Category': category_name,
                'Activity': activity,
                'Duration': act_duration,
                'Percentage': percentage
            })
    
    # Create dataframe
    workstation_df = pd.DataFrame(workstation_summary)
    
    # Only proceed if we have enough data
    if len(workstation_df) > 0:
        # 1. Workstation Category Comparison
        try:
            plt.figure(figsize=(12, 6))
            
            # Create stacked bar chart data
            plot_df = workstation_df.pivot(index='Category', columns='Activity', values='Percentage')
            
            # Fill NaN with 0
            plot_df = plot_df.fillna(0)
            
            # Ensure all activities are present
            for activity in handling_categories:
                if activity not in plot_df.columns:
                    plot_df[activity] = 0
            
            # Custom sort order
            category_order = ['Bread Workstation', 'Cake Workstation', 'Bread Baking', 'Cake Baking', 'Common Regions']
            plot_df = plot_df.reindex(category_order)
            
            # Get activity colors
            activity_colors = get_activity_colors()
            colors = [activity_colors.get(act, '#CCCCCC') for act in handling_categories]
            
            # Create bar chart
            ax = plot_df[handling_categories].plot(kind='bar', stacked=True, color=colors)
            
            plt.title(get_text('Handling Position Distribution by Workstation Type', language), fontsize=14)
            plt.xlabel(get_text('Workstation Type', language))
            plt.ylabel(get_text('Percentage of Time (%)', language))
            plt.xticks(rotation=45, ha='right')
            plt.legend(title=get_text('Position', language))
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add percentage labels
            for container in ax.containers:
                ax.bar_label(container, label_type='center', fmt='%.1f%%')
            
            plt.tight_layout()
            plt.savefig(vis_dir / 'workstation_comparison.png', dpi=300)
            plt.close()
            
            print(f"  {get_text('Saved workstation comparison visualization to {0}/workstation_comparison.png', language).format(vis_dir)}")
        except Exception as e:
            print(f"{get_text('Error creating workstation comparison visualization:', language)} {str(e)}")
        
        # 2. Bread vs Cake Comparison (specifically for these two categories)
        try:
            bread_data = workstation_df[workstation_df['Category'] == 'Bread Workstation']
            cake_data = workstation_df[workstation_df['Category'] == 'Cake Workstation']
            
            if not bread_data.empty and not cake_data.empty:
                plt.figure(figsize=(10, 6))
                
                # Prepare data for grouped bar chart
                bread_values = []
                cake_values = []
                
                for activity in handling_categories:
                    bread_act = bread_data[bread_data['Activity'] == activity]
                    bread_val = bread_act['Percentage'].values[0] if not bread_act.empty else 0
                    bread_values.append(bread_val)
                    
                    cake_act = cake_data[cake_data['Activity'] == activity]
                    cake_val = cake_act['Percentage'].values[0] if not cake_act.empty else 0
                    cake_values.append(cake_val)
                
                # Create the grouped bar chart
                x = np.arange(len(handling_categories))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(10, 6))
                rects1 = ax.bar(x - width/2, bread_values, width, label='Bread Workstation', color='#8c6464')
                rects2 = ax.bar(x + width/2, cake_values, width, label='Cake Workstation', color='#ffd700')
                
                # Add labels and ticks
                ax.set_ylabel(get_text('Percentage of Time (%)', language))
                ax.set_title(get_text('Bread vs. Cake Workstation Handling Patterns', language))
                ax.set_xticks(x)
                ax.set_xticklabels([get_text(activity, language) for activity in handling_categories])
                ax.legend()
                
                # Add value labels
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.1f}%',
                                  xy=(rect.get_x() + rect.get_width()/2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom')
                
                autolabel(rects1)
                autolabel(rects2)
                
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(vis_dir / 'bread_vs_cake_comparison.png', dpi=300)
                plt.close()
                
                print(f"  {get_text('Saved bread vs cake comparison visualization to {0}/bread_vs_cake_comparison.png', language).format(vis_dir)}")
        except Exception as e:
            print(f"{get_text('Error creating bread vs cake comparison visualization:', language)} {str(e)}")