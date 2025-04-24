"""
Ergonomics Analysis Module

Analyzes ergonomic patterns and generates employee and region ergonomic scores with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.colors as mcolors

# Import visualization utilities from base
from src.visualization.base import (
    save_figure, set_visualization_style, 
    get_activity_colors, get_employee_colors,
    get_activity_order, add_duration_percentage_label,
    get_text
)
from src.utils.time_utils import format_seconds_to_hms
from utils.file_utils import ensure_dir_exists

def create_region_categories():
    """Group similar regions into functional clusters for handling analysis"""
    return {
        'Bread Workstation': ['3_Brotstation', '4_Brotstation', '1_Brotstation', '2_Brotstation'],
        'Cake Workstation': ['1_Konditorei_station', '2_Konditorei_station', '3_Konditorei_station', 'konditorei_deco'],
        'Bread Baking': ['bereitstellen_prepared_goods', 'Etagenofen', 'Rollenhandtuchspend', 'Gärraum'],
        'Cake Baking': ['Fettbacken', 'cookies_packing_hand'],
        'Common Regions': ['brotkisten_regal', 'production_plan', 'leere_Kisten', '1_corridor']
    }

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

def analyze_handling_time_in_top_regions(data, output_dir=None, language='en'):
    """
    Analyze handling time in top regions per employee
    
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
    tuple
        (pivot_table, employee_top_regions, results_df)
    """
    # List all employees
    employees = sorted(data['id'].unique())
    print(f"\n{get_text('Analyzing data for {0} employees:', language).format(len(employees))} {', '.join(employees)}")
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # Find top 5 regions for each employee
    employee_top_regions = {}
    for employee_id in employees:
        emp_data = data[data['id'] == employee_id]
        region_times = emp_data.groupby('region')['duration'].sum().reset_index()
        top_regions = region_times.sort_values('duration', ascending=False).head(5)['region'].tolist()
        employee_top_regions[employee_id] = top_regions
        print(f"  {get_text('Employee {0} top 5 regions:', language).format(employee_id)} {', '.join(top_regions)}")
    
    # Get all unique top regions
    all_top_regions = set()
    for regions in employee_top_regions.values():
        all_top_regions.update(regions)
    all_top_regions = sorted(all_top_regions)
    
    print(f"\n{get_text('Identified {0} unique top regions across all employees', language).format(len(all_top_regions))}")
    
    # Calculate handling time for each region and category
    results = []
    for region in all_top_regions:
        region_data = data[(data['region'] == region) & (data['activity'].isin(handling_categories))]
        
        for category in handling_categories:
            category_data = region_data[region_data['activity'] == category]
            duration_seconds = category_data['duration'].sum()
            
            results.append({
                'region': region,
                'category': category,
                'duration_seconds': duration_seconds,
                'employees': [emp for emp, regions in employee_top_regions.items() if region in regions]
            })
    
    # Create pivot table
    results_df = pd.DataFrame(results)
    
    # Create pivot only if there's data
    if not results_df.empty:
        pivot_table = results_df.pivot_table(
            index='region',
            columns='category',
            values='duration_seconds',
            fill_value=0
        )
        
        # Add total column
        for category in handling_categories:
            if category not in pivot_table.columns:
                pivot_table[category] = 0
        pivot_table['Total'] = pivot_table.sum(axis=1)
        
        # Sort by total handling time
        pivot_table = pivot_table.sort_values('Total', ascending=False)
        
        # Format as hours:minutes:seconds
        formatted_pivot = pivot_table.copy()
        for col in formatted_pivot.columns:
            formatted_pivot[col] = formatted_pivot[col].apply(format_seconds_to_hms)
        
        # Add employee column
        region_employees = {}
        for region in pivot_table.index:
            region_employees[region] = ", ".join(
                [emp for emp, regions in employee_top_regions.items() if region in regions]
            )
        formatted_pivot['Employees'] = pd.Series(region_employees)
        
        # Save to CSV if output_dir is provided
        if output_dir:
            output_path = ensure_dir_exists(output_dir / 'handling_analysis')
            formatted_pivot.to_csv(output_path / 'handling_time_by_region.csv')
            print(f"  {get_text('Saved handling time analysis to {0}/handling_time_by_region.csv', language).format(output_path)}")
        
        # Calculate and display totals by category
        print(f"\n{get_text('Total Handling Time by Category (across all top regions):', language)}")
        print("-" * 50)
        
        grand_total = 0
        for category in handling_categories:
            if category in pivot_table.columns:
                duration = pivot_table[category].sum()
                grand_total += duration
                print(f"{get_text(category, language)}: {format_seconds_to_hms(duration)}")
        
        print("-" * 50)
        print(f"{get_text('Grand Total:', language)} {format_seconds_to_hms(grand_total)}")
    else:
        pivot_table = pd.DataFrame()
        print(f"{get_text('No handling data found', language)}")
    
    return pivot_table, employee_top_regions, results_df

def analyze_workstation_categories(data, output_dir=None, language='en'):
    """
    Analyze handling data by workstation category
    
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
    tuple
        (category_df, region_categories)
    """
    # Define region categories
    region_categories = create_region_categories()
    
    # Create dictionary to store aggregated values
    aggregated_results = {
        'Category': [],
        'Handle center': [],
        'Handle down': [],
        'Handle up': [],
        'Total': []
    }
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # Calculate totals for each category
    for category_name, regions in region_categories.items():
        category_data = data[data['region'].isin(regions) & data['activity'].isin(handling_categories)]
        
        handle_center = category_data[category_data['activity'] == 'Handle center']['duration'].sum()
        handle_down = category_data[category_data['activity'] == 'Handle down']['duration'].sum()
        handle_up = category_data[category_data['activity'] == 'Handle up']['duration'].sum()
        total = handle_center + handle_down + handle_up
        
        aggregated_results['Category'].append(category_name)
        aggregated_results['Handle center'].append(handle_center)
        aggregated_results['Handle down'].append(handle_down)
        aggregated_results['Handle up'].append(handle_up)
        aggregated_results['Total'].append(total)
    
    # Convert to DataFrame
    df = pd.DataFrame(aggregated_results)
    
    # Create a formatted version with time strings
    formatted_df = df.copy()
    for col in ['Handle center', 'Handle down', 'Handle up', 'Total']:
        formatted_df[col] = formatted_df[col].apply(format_seconds_to_hms)
    
    # Add percentage columns
    df['Handle center %'] = (df['Handle center'] / df['Total'] * 100).round(1)
    df['Handle down %'] = (df['Handle down'] / df['Total'] * 100).round(1)
    df['Handle up %'] = (df['Handle up'] / df['Total'] * 100).round(1)
    
    # Add percentage columns to formatted DataFrame
    formatted_df['Handle center %'] = df['Handle center %'].apply(lambda x: f"{x}%")
    formatted_df['Handle down %'] = df['Handle down %'].apply(lambda x: f"{x}%")
    formatted_df['Handle up %'] = df['Handle up %'].apply(lambda x: f"{x}%")
    
    # Sort by total time
    formatted_df = formatted_df.sort_values('Total', key=lambda x: df['Total'], ascending=False)
    
    # Save to CSV if output_dir is provided
    if output_dir:
        output_path = ensure_dir_exists(output_dir / 'handling_analysis')
        formatted_df.to_csv(output_path / 'workstation_category_analysis.csv', index=False)
        print(f"  {get_text('Saved workstation category analysis to {0}/workstation_category_analysis.csv', language).format(output_path)}")
    
    print(f"\n{get_text('Handling Time by Workstation Category', language)}")
    print("=" * 80)
    print(formatted_df.to_string(index=False))
    
    return df, region_categories

def analyze_handling_position_patterns(data, region_categories, output_dir=None, language='en'):
    """
    Perform statistical analysis of handling positions by workstation type
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    region_categories : dict
        Dictionary mapping categories to regions
    output_dir : Path, optional
        Directory to save output files
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with comparison results
    """
    print(f"\n{get_text('Statistical Analysis of Handling Positions', language)}")
    print("=" * 80)
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # For comparing bread vs cake workstations
    bread_ws_regions = region_categories['Bread Workstation']
    cake_ws_regions = region_categories['Cake Workstation']
    
    # Extract detailed handling data for statistical tests
    bread_ws_data = data[data['region'].isin(bread_ws_regions) & data['activity'].isin(handling_categories)]
    cake_ws_data = data[data['region'].isin(cake_ws_regions) & data['activity'].isin(handling_categories)]
    
    # Check if both datasets have data
    if bread_ws_data.empty or cake_ws_data.empty:
        print(f"{get_text('Insufficient data for comparative analysis', language)}")
        return pd.DataFrame()
    
    # Calculate percentages for visualization
    bread_total = bread_ws_data['duration'].sum()
    cake_total = cake_ws_data['duration'].sum()
    
    bread_by_position = {}
    cake_by_position = {}
    
    for position in handling_categories:
        bread_position_time = bread_ws_data[bread_ws_data['activity'] == position]['duration'].sum()
        cake_position_time = cake_ws_data[cake_ws_data['activity'] == position]['duration'].sum()
        
        bread_by_position[position] = bread_position_time
        cake_by_position[position] = cake_position_time
    
    bread_percentages = {pos: (time/bread_total*100) if bread_total > 0 else 0 
                       for pos, time in bread_by_position.items()}
    cake_percentages = {pos: (time/cake_total*100) if cake_total > 0 else 0 
                      for pos, time in cake_by_position.items()}
    
    # Perform chi-square test on the distribution of handling positions
    try:
        from scipy import stats
        observed = np.array([
            [bread_by_position.get('Handle up', 0), bread_by_position.get('Handle center', 0), bread_by_position.get('Handle down', 0)],
            [cake_by_position.get('Handle up', 0), cake_by_position.get('Handle center', 0), cake_by_position.get('Handle down', 0)]
        ])
        
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        chi2_result = {'chi2': chi2, 'p_value': p, 'dof': dof}
    except ImportError:
        print(f"{get_text('SciPy not found, skipping statistical tests', language)}")
        chi2_result = {'chi2': None, 'p_value': None, 'dof': None}
    except Exception as e:
        print(f"{get_text('Error in statistical test:', language)} {str(e)}")
        chi2_result = {'chi2': None, 'p_value': None, 'dof': None}
    
    # Create results dataframe
    comparison = pd.DataFrame({
        'Position': handling_categories,
        'Bread Workstation (sec)': [bread_by_position.get(pos, 0) for pos in handling_categories],
        'Bread Workstation (%)': [bread_percentages.get(pos, 0) for pos in handling_categories],
        'Cake Workstation (sec)': [cake_by_position.get(pos, 0) for pos in handling_categories],
        'Cake Workstation (%)': [cake_percentages.get(pos, 0) for pos in handling_categories],
    })
    
    # Add ratio column safely
    comparison['Ratio (Cake/Bread)'] = [
        cake_by_position.get(pos, 0)/bread_by_position.get(pos, 1) 
        if bread_by_position.get(pos, 0) > 0 else float('inf')
        for pos in handling_categories
    ]
    
    # Format time values
    comparison['Bread Workstation (formatted)'] = comparison['Bread Workstation (sec)'].apply(format_seconds_to_hms)
    comparison['Cake Workstation (formatted)'] = comparison['Cake Workstation (sec)'].apply(format_seconds_to_hms)
    
    # Round percentages
    comparison['Bread Workstation (%)'] = comparison['Bread Workstation (%)'].round(1)
    comparison['Cake Workstation (%)'] = comparison['Cake Workstation (%)'].round(1)
    
    # Save to CSV if output_dir is provided
    if output_dir:
        output_path = ensure_dir_exists(output_dir / 'handling_analysis')
        comparison.to_csv(output_path / 'bread_vs_cake_comparison.csv', index=False)
        
        # Save chi-square test results
        if chi2_result['chi2'] is not None:
            with open(output_path / 'statistical_test_results.txt', 'w') as f:
                f.write(f"Chi-Square Test Results:\n")
                f.write(f"Chi² value: {chi2_result['chi2']:.2f}\n")
                f.write(f"p-value: {chi2_result['p_value']:.6f}\n")
                f.write(f"Degrees of freedom: {chi2_result['dof']}\n")
                
                # Interpret results
                alpha = 0.05
                if chi2_result['p_value'] < alpha:
                    f.write("\nSTATISTICAL INFERENCE: There is a statistically significant difference between\n")
                    f.write("handling position distributions at bread and cake workstations (p < 0.05).\n")
                else:
                    f.write("\nSTATISTICAL INFERENCE: There is no statistically significant difference between\n")
                    f.write("handling position distributions at bread and cake workstations (p >= 0.05).\n")
        
        print(f"  {get_text('Saved bread vs cake comparison to {0}/bread_vs_cake_comparison.csv', language).format(output_path)}")
    
    # Print results
    print(f"\n{get_text('Comparison of Handling Positions: Bread vs. Cake Workstations', language)}")
    print(comparison[['Position', 'Bread Workstation (formatted)', 'Bread Workstation (%)', 
                    'Cake Workstation (formatted)', 'Cake Workstation (%)', 'Ratio (Cake/Bread)']].to_string(index=False))
    
    if chi2_result['chi2'] is not None:
        print(f"\n{get_text('Chi-Square Test Results:', language)}")
        print(f"{get_text('Chi² value:', language)} {chi2_result['chi2']:.2f}")
        print(f"{get_text('p-value:', language)} {chi2_result['p_value']:.6f}")
        print(f"{get_text('Degrees of freedom:', language)} {chi2_result['dof']}")
        
        # Interpret results
        alpha = 0.05
        if chi2_result['p_value'] < alpha:
            print(f"\n{get_text('STATISTICAL INFERENCE: There is a statistically significant difference between', language)}")
            print(f"{get_text('handling position distributions at bread and cake workstations (p < 0.05).', language)}")
        else:
            print(f"\n{get_text('STATISTICAL INFERENCE: There is no statistically significant difference between', language)}")
            print(f"{get_text('handling position distributions at bread and cake workstations (p >= 0.05).', language)}")
    
    # Analyze the key differences
    print(f"\n{get_text('Key findings:', language)}")
    
    # Handle up comparison
    up_bread_pct = bread_percentages.get('Handle up', 0)
    up_cake_pct = cake_percentages.get('Handle up', 0)
    
    if up_cake_pct > up_bread_pct:
        diff_pct = up_cake_pct - up_bread_pct
        print(f"- {get_text('Cake workstations involve {0:.1f}% more Handle up positions compared to bread workstations', language).format(diff_pct)}")
        print(f"  ({up_cake_pct:.1f}% vs {up_bread_pct:.1f}%)")
    else:
        diff_pct = up_bread_pct - up_cake_pct
        print(f"- {get_text('Bread workstations involve {0:.1f}% more Handle up positions compared to cake workstations', language).format(diff_pct)}")
        print(f"  ({up_bread_pct:.1f}% vs {up_cake_pct:.1f}%)")
    
    # Handle down comparison
    down_bread_pct = bread_percentages.get('Handle down', 0)
    down_cake_pct = cake_percentages.get('Handle down', 0)
    
    if down_bread_pct > down_cake_pct:
        diff_pct = down_bread_pct - down_cake_pct
        print(f"- {get_text('Bread workstations involve {0:.1f}% more Handle down positions compared to cake workstations', language).format(diff_pct)}")
        print(f"  ({down_bread_pct:.1f}% vs {down_cake_pct:.1f}%)")
    else:
        diff_pct = down_cake_pct - down_bread_pct
        print(f"- {get_text('Cake workstations involve {0:.1f}% more Handle down positions compared to cake workstations', language).format(diff_pct)}")
        print(f"  ({down_cake_pct:.1f}% vs {down_bread_pct:.1f}%)")
    
    # Handle center comparison
    center_bread_pct = bread_percentages.get('Handle center', 0)
    center_cake_pct = cake_percentages.get('Handle center', 0)
    
    if center_bread_pct > center_cake_pct:
        diff_pct = center_bread_pct - center_cake_pct
        print(f"- {get_text('Bread workstations involve {0:.1f}% more Handle center positions compared to cake workstations', language).format(diff_pct)}")
        print(f"  ({center_bread_pct:.1f}% vs {center_cake_pct:.1f}%)")
    else:
        diff_pct = center_cake_pct - center_bread_pct
        print(f"- {get_text('Cake workstations involve {0:.1f}% more Handle center positions compared to cake workstations', language).format(diff_pct)}")
        print(f"  ({center_cake_pct:.1f}% vs {center_bread_pct:.1f}%)")
    
    # Provide ergonomic implications
    print(f"\n{get_text('Ergonomic implications:', language)}")
    if up_cake_pct > up_bread_pct:
        print(f"- {get_text('Cake workstations may require more ergonomic attention for overhead reaching tasks', language)}")
    if down_bread_pct > down_cake_pct:
        print(f"- {get_text('Bread workstations may require more ergonomic attention for low-level handling tasks', language)}")
    
    return comparison

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

def run_handling_time_analysis(data, output_dir, language='en'):
    """
    Run comprehensive handling time analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path
        Directory to save outputs
    language : str, optional
        Language code ('en' or 'de')
    """
    print("\n" + "=" * 80)
    print(f"=== {get_text('Handling Time Analysis', language)} ===")
    print("=" * 80)
    
    # Create output directory
    handling_dir = ensure_dir_exists(output_dir / 'handling_analysis')
    
    # Step 1: Top regions analysis
    print(f"\n=== {get_text('Step 1: Analyzing Top Regions by Employee', language)} ===")
    pivot_table, employee_top_regions, results_df = analyze_handling_time_in_top_regions(data, handling_dir, language)
    
    # Step 2: Workstation categories analysis
    print(f"\n=== {get_text('Step 2: Analyzing Workstation Categories', language)} ===")
    category_df, region_categories = analyze_workstation_categories(data, handling_dir, language)
    
    # Step 3: Statistical analysis
    print(f"\n=== {get_text('Step 3: Statistical Analysis of Handling Positions', language)} ===")
    comparison_df = analyze_handling_position_patterns(data, region_categories, handling_dir, language)
    
    # Step 4: Create visualizations
    print(f"\n=== {get_text('Step 4: Creating Handling Position Visualizations', language)} ===")
    create_handling_position_visualizations(data, region_categories, output_dir, language)
    
    print(f"\n=== {get_text('Handling Time Analysis Complete', language)} ===")

def calculate_ergonomic_score(data, min_duration=5):
    """
    Calculate ergonomic scores out of 100 for each employee with revised methodology
    that more heavily penalizes problematic postures
    
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
    # Filter out very short activities that might be false readings
    filtered_data = data[data['duration'] >= min_duration]
    
    # Starting score is 100, deductions will be made for ergonomic issues
    base_score = 100
    
    # Define direct percentage-based penalties by posture
    # These represent deduction points per percentage point of time spent in the posture
    posture_penalties = {
        'Handle up': 0.4,     # 40% penalty per % of time (high risk)
        'Handle down': 0.5,   # 50% penalty per % of time (highest risk)
        'Handle center': 0.05, # 5% penalty per % of time (low risk)
        'Stand': 0.15,        # 15% penalty per % of time (medium risk)
        'Walk': 0.05          # 5% penalty per % of time (low risk)
    }
    
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
        }
        
        # Calculate direct percentage-based deductions by activity
        total_deduction = 0
        
        for activity in posture_penalties.keys():
            act_data = emp_data[emp_data['activity'] == activity]
            if act_data.empty:
                continue
                
            activity_duration = act_data['duration'].sum()
            activity_percentage = (activity_duration / total_duration) * 100
            
            # Calculate direct percentage-based deduction
            direct_deduction = activity_percentage * posture_penalties[activity]
            total_deduction += direct_deduction
            score_components['deductions'][activity] = direct_deduction
            
            # Store activity breakdown
            score_components['activity_breakdown'][activity] = {
                'percentage': activity_percentage,
                'duration': activity_duration,
                'deduction': direct_deduction
            }
        
        # Calculate region-specific deductions
        region_deductions = {}
        for region, region_data in emp_data.groupby('region'):
            region_duration = region_data['duration'].sum()
            region_percentage = (region_duration / total_duration) * 100
            
            # Calculate weighted deduction for this region
            region_deduction = 0
            for activity, act_data in region_data.groupby('activity'):
                if activity not in posture_penalties:
                    continue
                    
                activity_duration = act_data['duration'].sum()
                activity_percentage = (activity_duration / region_duration) * 100
                region_deduction += activity_percentage * posture_penalties[activity] / 100
            
            # Store region breakdown
            region_deductions[region] = region_deduction * region_percentage / 100
            score_components['region_breakdown'][region] = {
                'percentage': region_percentage,
                'duration': region_duration,
                'deduction': region_deductions[region]
            }
        
        # Calculate final score (capped at minimum of 0)
        final_score = max(base_score - total_deduction, 0)
        score_components['final_score'] = final_score
        
        employee_scores.append({
            'employee_id': emp_id,
            'final_score': final_score,
            'total_duration': total_duration,
            'components': score_components
        })
    
    return employee_scores

def export_duration_factor_thresholds(data, output_dir, language='en'):
    """
    Export the duration factor thresholds used in ergonomic scoring
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path
        Directory to save the threshold information
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    activity_stats = analyze_activity_durations(data)
    output_path = Path(output_dir)
    
    with open(output_path / 'duration_factor_thresholds.txt', 'w') as f:
        f.write(get_text("Ergonomic Scoring System", language) + "\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(get_text("Deduction points per percentage of time spent in each posture:", language) + "\n")
        f.write(get_text("  Handle up: 0.4 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Handle down: 0.5 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Handle center: 0.05 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Stand: 0.15 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Walk: 0.05 points deducted per percentage point of time spent", language) + "\n\n")
        
        f.write(get_text("Example: An employee spending 50% of time in Handle down position", language) + "\n")
        f.write(get_text("  would receive a deduction of 50 × 0.5 = 25 points", language) + "\n\n")
        
        f.write(get_text("Score categories:", language) + "\n")
        f.write(get_text("  Good: 75-100", language) + "\n")
        f.write(get_text("  Fair: 60-75", language) + "\n")
        f.write(get_text("  Poor: 0-60", language) + "\n\n")
        
        f.write(get_text("Duration statistics by activity:", language) + "\n")
        for activity, stats in activity_stats.items():
            f.write(f"{get_text(activity, language)}:\n")
            f.write(f"  {get_text('Count', language)}: {stats['count']}\n")
            f.write(f"  {get_text('Mean', language)}: {stats['mean']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('Median', language)}: {stats['median']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('25th percentile', language)}: {stats['p25']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('75th percentile', language)}: {stats['p75']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('90th percentile', language)}: {stats['p90']:.1f} {get_text('seconds', language)}\n\n")

def generate_ergonomic_report(employee_scores, output_dir, language='en'):
    """
    Generate detailed ergonomic report visualizations for each employee
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries from calculate_ergonomic_score
    output_dir : Path
        Directory to save the reports
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard activity colors and order
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
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
        
        # Create figure with 2 panels
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 2])
        
        # 1. Main score display
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.axis('off')
        
        # Get employee-specific color
        emp_color = employee_colors.get(emp_id, '#333333')
        
        # Display the score prominently
        score_color = 'green' if score >= 75 else ('orange' if score >= 60 else 'red')
        ax_main.text(0.5, 0.7, get_text("Ergonomic Score: {0}/100", language).format(f"{score:.1f}"), 
                    fontsize=28, ha='center', color=score_color, fontweight='bold')
        
        # Add comparison to other employees
        comparison_text = get_text("Average: {0:.1f} | Min: {1:.1f} | Max: {2:.1f}", language).format(avg_score, min_score, max_score)
        ax_main.text(0.5, 0.5, comparison_text, fontsize=14, ha='center')
        
        # Add employee ID and total duration
        hours = emp_data['total_duration'] / 3600
        ax_main.text(0.5, 0.3, get_text("Employee: {0} | Hours Analyzed: {1:.1f}", language).format(emp_id, hours), 
                    fontsize=16, ha='center', fontweight='bold', color=emp_color)
        
        # 2. Deductions breakdown
        ax_deduct = fig.add_subplot(gs[1, 0])
        
        # Extract deductions
        deductions = components['deductions']
        activities = list(deductions.keys())
        values = list(deductions.values())
        
        # Sort by deduction value
        sorted_idx = np.argsort(values)
        activities = [activities[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Use activity-specific colors from base visualization
        colors = [activity_colors.get(act, '#CCCCCC') for act in activities]
        
        # Create horizontal bar chart
        bars = ax_deduct.barh([get_text(act, language) for act in activities], values, color=colors)
        ax_deduct.set_xlabel(get_text('Deduction Points', language))
        ax_deduct.set_title(get_text('Ergonomic Deductions by Activity', language), fontsize=14)
        
        # Add labels
        for bar in bars:
            width = bar.get_width()
            ax_deduct.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                          f"{width:.1f}", va='center')
        
        # 3. Activity time distribution (bar chart)
        ax_activity = fig.add_subplot(gs[1, 1])
        
        # Extract activity durations
        act_breakdown = components['activity_breakdown']
        
        # Order activities according to standard order
        ordered_activities = []
        ordered_durations = []
        ordered_colors = []
        
        # First add activities in standard order if they exist
        for activity in activity_order:
            if activity in act_breakdown:
                ordered_activities.append(activity)
                ordered_durations.append(act_breakdown[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Then add any remaining activities
        for activity in act_breakdown.keys():
            if activity not in ordered_activities and activity in activity_colors:
                ordered_activities.append(activity)
                ordered_durations.append(act_breakdown[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Create bar chart
        bars = ax_activity.bar([get_text(act, language) for act in ordered_activities], ordered_durations, color=ordered_colors)
        
        # Add percentage and duration labels
        total_seconds = sum(act_breakdown[act]['duration'] for act in act_breakdown)
        for i, activity in enumerate(ordered_activities):
            act_duration = act_breakdown[activity]['duration']
            act_percentage = act_breakdown[activity]['percentage']
            
            # Format duration
            formatted_duration = format_seconds_to_hms(act_duration)
            
            # Position label at top of bar
            bar = bars[i]
            height = bar.get_height()
            ax_activity.text(bar.get_x() + bar.get_width()/2., height,
                           f"{act_percentage:.1f}%\n{formatted_duration}", 
                           ha='center', va='bottom', fontsize=10)
        
        ax_activity.set_ylabel(get_text('Duration (hours)', language))
        ax_activity.set_title(get_text('Activity Time Distribution', language), fontsize=14)
        plt.setp(ax_activity.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_activity.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Add title and save
        plt.suptitle(get_text("Ergonomic Assessment Report - Employee {0}", language).format(emp_id), fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Use the save_figure utility from base
        save_figure(fig, output_path / f"{emp_id}_ergonomic_report.png")
        
def analyze_region_ergonomics(data, min_percentage=5, min_duration=0):
    """
    Analyze ergonomic patterns for each region - modified to only use handling activities
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    min_percentage : float, optional
        Minimum percentage of time an employee must spend at a region to be included
    min_duration : int, optional
        Minimum duration in seconds to filter out potential false readings
    
    Returns:
    --------
    dict
        Dictionary containing region-level ergonomic analyses
    """
    # Filter activities to include only handling activities and with min duration
    filtered_data = data[(data['activity'].str.contains('Handle')) & (data['duration'] >= min_duration)]
    
    results = {}
    
    # Get all unique regions
    all_regions = filtered_data['region'].unique()
    
    # Process each region
    for region in all_regions:
        region_data = filtered_data[filtered_data['region'] == region]
        
        if region_data.empty:
            continue
            
        # Total duration in this region across all employees
        total_region_duration = region_data['duration'].sum()
        
        # Identify employees who use this region significantly
        qualified_employees = []
        
        for emp_id, emp_data in filtered_data.groupby('id'):
            # Total time for this employee
            emp_total_time = emp_data['duration'].sum()
            
            # Time in this region
            region_time = emp_data[emp_data['region'] == region]['duration'].sum()
            region_percentage = (region_time / emp_total_time * 100) if emp_total_time > 0 else 0
            
            # Check if this region is in the employee's top 5
            emp_top_regions = emp_data.groupby('region')['duration'].sum().sort_values(ascending=False).head(5).index.tolist()
            
            if region_percentage >= min_percentage and region in emp_top_regions:
                qualified_employees.append({
                    'id': emp_id,
                    'time': region_time,
                    'percentage': region_percentage,
                    'in_top5': region in emp_top_regions
                })
        
        # If no qualified employees, skip this region
        if not qualified_employees:
            continue
        
        # Calculate activity distribution in this region (considering only qualified employees)
        qualified_emp_ids = [emp['id'] for emp in qualified_employees]
        qualified_region_data = region_data[region_data['id'].isin(qualified_emp_ids)]
        
        activity_distribution = {}
        for activity in ['Handle up', 'Handle down', 'Handle center']:
            act_data = qualified_region_data[qualified_region_data['activity'] == activity]
            act_duration = act_data['duration'].sum()
            act_percentage = (act_duration / qualified_region_data['duration'].sum() * 100) if not qualified_region_data.empty else 0
            
            activity_distribution[activity] = {
                'duration': act_duration,
                'percentage': act_percentage
            }
        
        # Calculate ergonomic score for this region
        posture_penalties = {
            'Handle up': 0.4,
            'Handle down': 0.5, 
            'Handle center': 0.05
        }
        
        deduction = sum(activity_distribution[act]['percentage'] * penalty 
                        for act, penalty in posture_penalties.items() if act in activity_distribution)
        ergonomic_score = max(100 - deduction, 0)
        
        # Calculate employee contribution to this region
        employee_contributions = []
        for emp in qualified_employees:
            contribution_percentage = (emp['time'] / qualified_region_data['duration'].sum() * 100) if not qualified_region_data.empty else 0
            
            # Get activity breakdown for this employee in this region
            emp_region_data = qualified_region_data[qualified_region_data['id'] == emp['id']]
            emp_activity_breakdown = {}
            
            for activity in ['Handle up', 'Handle down', 'Handle center']:
                act_data = emp_region_data[emp_region_data['activity'] == activity]
                act_duration = act_data['duration'].sum()
                act_percentage = (act_duration / emp_region_data['duration'].sum() * 100) if not emp_region_data.empty else 0
                
                emp_activity_breakdown[activity] = {
                    'duration': act_duration,
                    'percentage': act_percentage
                }
            
            employee_contributions.append({
                'id': emp['id'],
                'time': emp['time'],
                'contribution_percentage': contribution_percentage,
                'region_percentage': emp['percentage'],
                'activity_breakdown': emp_activity_breakdown
            })
        
        # Store results
        results[region] = {
            'qualified_employees': len(qualified_employees),
            'total_duration': qualified_region_data['duration'].sum(),
            'activity_distribution': activity_distribution,
            'ergonomic_score': ergonomic_score,
            'employee_contributions': employee_contributions
        }
    
    return results

def generate_region_ergonomic_report(region_analyses, output_dir, language='en'):
    """
    Generate ergonomic reports for each analyzed region
    
    Parameters:
    -----------
    region_analyses : dict
        Dictionary of region analyses from analyze_region_ergonomics
    output_dir : Path
        Directory to save the reports
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard activity colors and order
    activity_colors = get_activity_colors()
    activity_order = get_activity_order()
    
    # Generate report for each region
    for region, analysis in region_analyses.items():
        # Skip regions with no ergonomic score (should not happen but just in case)
        if 'ergonomic_score' not in analysis:
            continue
            
        score = analysis['ergonomic_score']
        total_duration = analysis['total_duration']
        activity_distribution = analysis['activity_distribution']
        employee_contributions = analysis['employee_contributions']
        
        # Create figure with 3 panels
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1.5])
        
        # 1. Main score display
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.axis('off')
        
        # Display the score prominently
        score_color = 'green' if score >= 75 else ('orange' if score >= 60 else 'red')
        ax_main.text(0.5, 0.7, get_text("Region Ergonomic Score: {0}/100", language).format(f"{score:.1f}"), 
                    fontsize=28, ha='center', color=score_color, fontweight='bold')
        
        # Add region info
        hours = total_duration / 3600
        qualified_employees = analysis['qualified_employees']
        ax_main.text(0.5, 0.4, get_text("Region: {0} | Hours Analyzed: {1:.1f} | Employees: {2}", language).format(
            region, hours, qualified_employees), 
            fontsize=16, ha='center', fontweight='bold')
        
        # 2. Activity distribution in this region (keep the original bar chart)
        ax_activity = fig.add_subplot(gs[1, 0])
        
        # Order activities according to standard order
        ordered_activities = []
        ordered_durations = []
        ordered_colors = []
        
        for activity in activity_order:
            if activity in activity_distribution:
                ordered_activities.append(activity)
                ordered_durations.append(activity_distribution[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Add any remaining activities
        for activity in activity_distribution.keys():
            if activity not in ordered_activities and activity in activity_colors:
                ordered_activities.append(activity)
                ordered_durations.append(activity_distribution[activity]['duration'] / 3600)  # Convert to hours
                ordered_colors.append(activity_colors.get(activity, '#CCCCCC'))
        
        # Create bar chart
        bars = ax_activity.bar([get_text(act, language) for act in ordered_activities], ordered_durations, color=ordered_colors)
        
        # Add percentage and duration labels
        for i, activity in enumerate(ordered_activities):
            act_duration = activity_distribution[activity]['duration']
            act_percentage = activity_distribution[activity]['percentage']
            
            # Format duration
            formatted_duration = format_seconds_to_hms(act_duration)
            
            # Position label at top of bar
            bar = bars[i]
            height = bar.get_height()
            ax_activity.text(bar.get_x() + bar.get_width()/2., height,
                           f"{act_percentage:.1f}%\n{formatted_duration}", 
                           ha='center', va='bottom', fontsize=10)
        
        ax_activity.set_ylabel(get_text('Duration (hours)', language))
        ax_activity.set_title(get_text('Activity Distribution in Region', language), fontsize=14, pad=15)
        plt.setp(ax_activity.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid for readability
        ax_activity.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Employee contributions as a table
        ax_contrib_table = fig.add_subplot(gs[1, 1])
        ax_contrib_table.axis('off')
        
        # Sort employees by contribution percentage
        sorted_contributions = sorted(employee_contributions, key=lambda x: x['contribution_percentage'], reverse=True)
        
        # Prepare table data
        contrib_table_data = [[
            get_text("Employee", language),
            get_text("Duration", language),
            get_text("Contribution", language),
            get_text("Region %", language)
        ]]
        
        for contrib in sorted_contributions:
            contrib_table_data.append([
                contrib['id'],
                format_seconds_to_hms(contrib['time']),
                f"{contrib['contribution_percentage']:.1f}%",
                f"{contrib['region_percentage']:.1f}%"
            ])
        
        # Create table
        contrib_table = ax_contrib_table.table(
            cellText=contrib_table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        
        # Style table
        contrib_table.auto_set_font_size(False)
        contrib_table.set_fontsize(10)
        contrib_table.scale(1, 1.5)
        
        # Header formatting
        for j in range(4):
            contrib_table[(0, j)].set_facecolor('#4472C4')
            contrib_table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        ax_contrib_table.set_title(get_text('Employee Contributions to Region', language), fontsize=14, pad=15)
        
        # 4. Employee activity breakdown as a table
        ax_breakdown_table = fig.add_subplot(gs[2, :])
        ax_breakdown_table.axis('off')
        
        # Only include the top 5 contributing employees
        top_contributors = sorted_contributions[:min(5, len(sorted_contributions))]
        
        # Prepare table data
        breakdown_table_data = [[get_text("Employee", language)]]
        
        # Add activity columns
        for activity in activity_order:
            if activity in activity_distribution:
                breakdown_table_data[0].append(get_text(activity, language))
        
        # Add data rows for each employee
        for contrib in top_contributors:
            row = [contrib['id']]
            for activity in activity_order:
                if activity in activity_distribution:
                    if activity in contrib['activity_breakdown']:
                        act_data = contrib['activity_breakdown'][activity]
                        row.append(f"{act_data['percentage']:.1f}%\n({format_seconds_to_hms(act_data['duration'])})")
                    else:
                        row.append("0%")
            breakdown_table_data.append(row)
        
        # Create table
        breakdown_table = ax_breakdown_table.table(
            cellText=breakdown_table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.2] + [0.8/(len(breakdown_table_data[0])-1)] * (len(breakdown_table_data[0])-1)
        )
        
        # Style table
        breakdown_table.auto_set_font_size(False)
        breakdown_table.set_fontsize(10)
        breakdown_table.scale(1, 1.5)
        
        # Header formatting
        for j in range(len(breakdown_table_data[0])):
            breakdown_table[(0, j)].set_facecolor('#4472C4')
            breakdown_table[(0, j)].set_text_props(color='white', fontweight='bold')
            
        # Color activity columns based on activity colors
        for j, header in enumerate(breakdown_table_data[0][1:], 1):
            # Get the activity name from the header
            for act in activity_order:
                if get_text(act, language) == header:
                    color = activity_colors.get(act, '#CCCCCC')
                    for i in range(1, len(breakdown_table_data)):
                        breakdown_table[(i, j)].set_facecolor(color)
                        breakdown_table[(i, j)].set_alpha(0.3)
        
        ax_breakdown_table.set_title(get_text('Activity Breakdown by Employee in this Region', language), fontsize=14, pad=15)
        
        # Add title and save
        plt.suptitle(get_text("Region Ergonomic Assessment - {0}", language).format(region), fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Use the save_figure utility from base
        save_figure(fig, output_path / f"{region}_region_report.png")

def generate_all_employees_comparison(employee_scores, output_dir, language='en'):
    """
    Generate a combined visualization comparing all employees' ergonomic scores
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries
    output_dir : Path
        Directory to save the visualization
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard colors
    employee_colors = get_employee_colors()
    activity_colors = get_activity_colors()
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Sort employees by score
    sorted_employees = sorted(employee_scores, key=lambda x: x['final_score'], reverse=True)
    
    # 1. Bar chart of ergonomic scores
    ax_scores = fig.add_subplot(gs[0, 0])
    
    emp_ids = [emp['employee_id'] for emp in sorted_employees]
    scores = [emp['final_score'] for emp in sorted_employees]
    colors = [employee_colors.get(emp_id, '#333333') for emp_id in emp_ids]
    
    # Create horizontal bar chart
    bars = ax_scores.barh(emp_ids, scores, color=colors)
    
    # Add score values
    for bar in bars:
        width = bar.get_width()
        ax_scores.text(width + 1, bar.get_y() + bar.get_height()/2, 
                      f"{width:.1f}", va='center', fontweight='bold')
    
    # Add red-yellow-green background zones
    ax_scores.axvspan(0, 60, alpha=0.1, color='red')
    ax_scores.axvspan(60, 75, alpha=0.1, color='yellow')
    ax_scores.axvspan(75, 100, alpha=0.1, color='green')
    
    # Add text labels for zones
    ax_scores.text(30, -0.5, get_text("Poor", language), ha='center', va='top', color='red', fontsize=10)
    ax_scores.text(67.5, -0.5, get_text("Fair", language), ha='center', va='top', color='orange', fontsize=10)
    ax_scores.text(87.5, -0.5, get_text("Good", language), ha='center', va='top', color='green', fontsize=10)
    
    ax_scores.set_xlabel(get_text('Ergonomic Score', language), fontsize=12)
    ax_scores.set_title(get_text('Employee Ergonomic Scores', language), fontsize=14)
    ax_scores.set_xlim(0, 102)  # Leave room for labels
    
    # 2. Activity distribution heatmap
    ax_heatmap = fig.add_subplot(gs[0, 1])
    
    # Prepare data for heatmap
    heatmap_data = []
    for emp in sorted_employees:
        components = emp['components']
        act_breakdown = components['activity_breakdown']
        row = {'employee': emp['employee_id']}
        
        for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
            if activity in act_breakdown:
                row[activity] = act_breakdown[activity]['percentage']
            else:
                row[activity] = 0
        
        heatmap_data.append(row)
    
    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index('employee', inplace=True)
    
    # Create heatmap
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".1f", 
        cmap="YlOrRd", 
        cbar_kws={'label': get_text('Percentage of Time (%)', language)},
        ax=ax_heatmap
    )
    
    ax_heatmap.set_title(get_text('Activity Distribution by Employee (%)', language), fontsize=14)
    ax_heatmap.set_ylabel('')  # Remove y-label as it's redundant with the bar chart
    
    # 3. Deduction factors stacked bar chart
    ax_deductions = fig.add_subplot(gs[1, 0])
    
    # Prepare deduction data
    deduction_data = []
    for emp in sorted_employees:
        components = emp['components']
        deductions = components['deductions']
        row = {'employee': emp['employee_id']}
        
        for activity in ['Handle up', 'Handle down', 'Handle center', 'Stand', 'Walk']:
            row[activity] = deductions.get(activity, 0)
        
        deduction_data.append(row)
    
    # Convert to DataFrame
    deduction_df = pd.DataFrame(deduction_data)
    deduction_df.set_index('employee', inplace=True)
    
    # Create stacked bar chart
    deduction_df.plot(
        kind='barh', 
        stacked=True, 
        color=[activity_colors.get(col, '#CCCCCC') for col in deduction_df.columns],
        ax=ax_deductions
    )
    
    ax_deductions.set_title(get_text('Ergonomic Deduction Factors by Employee', language), fontsize=14)
    ax_deductions.set_xlabel(get_text('Deduction Points', language), fontsize=12)
    ax_deductions.legend(title=get_text('Activity', language))
    
    # 4. Key metrics table
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_metrics.axis('off')
    
    # Prepare metrics data
    metrics_data = [[
        get_text("Employee", language),
        get_text("Score", language),
        get_text("Hours", language),
        get_text("Handle up %", language),
        get_text("Handle down %", language),
        get_text("Stand %", language)
    ]]
    
    for emp in sorted_employees:
        components = emp['components']
        act_breakdown = components['activity_breakdown']
        
        handle_up_pct = act_breakdown.get('Handle up', {}).get('percentage', 0)
        handle_down_pct = act_breakdown.get('Handle down', {}).get('percentage', 0)
        stand_pct = act_breakdown.get('Stand', {}).get('percentage', 0)
        
        metrics_data.append([
            emp['employee_id'],
            f"{emp['final_score']:.1f}",
            f"{emp['total_duration']/3600:.1f}",
            f"{handle_up_pct:.1f}%",
            f"{handle_down_pct:.1f}%",
            f"{stand_pct:.1f}%"
        ])
    
    # Create table
    metrics_table = ax_metrics.table(
        cellText=metrics_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Style table
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(10)
    metrics_table.scale(1, 1.5)
    
    # Header formatting
    for j in range(len(metrics_data[0])):
        metrics_table[(0, j)].set_facecolor('#4472C4')
        metrics_table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color code scores
    for i in range(1, len(metrics_data)):
        score = float(metrics_data[i][1])
        if score >= 75:
            metrics_table[(i, 1)].set_facecolor('lightgreen')
        elif score >= 60:
            metrics_table[(i, 1)].set_facecolor('lightyellow')
        else:
            metrics_table[(i, 1)].set_facecolor('lightcoral')
    
    ax_metrics.set_title(get_text('Employee Ergonomic Metrics Summary', language), fontsize=14)
    
    # Add main title
    plt.suptitle(get_text("Bakery Employee Ergonomic Comparison", language), fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_figure(fig, output_path / "employee_ergonomic_comparison.png")

def generate_all_regions_comparison(region_analyses, output_dir, language='en'):
    """
    Generate a combined visualization comparing all regions' ergonomic metrics
    
    Parameters:
    -----------
    region_analyses : dict
        Dictionary of region analyses
    output_dir : Path
        Directory to save the visualization
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard colors
    activity_colors = get_activity_colors()
    
    # Filter out regions with no data
    valid_regions = {r: data for r, data in region_analyses.items() if 'ergonomic_score' in data}
    
    # If no valid regions, return
    if not valid_regions:
        print(f"No valid regions found for comparison")
        return
    
    # Sort regions by ergonomic score
    sorted_regions = sorted(valid_regions.items(), key=lambda x: x[1]['ergonomic_score'], reverse=True)
    
    # Get list of regions to display (top 15 to avoid overcrowding)
    display_regions = sorted_regions[:15]
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # 1. Bar chart of ergonomic scores
    ax_scores = fig.add_subplot(gs[0, 0])
    
    region_names = [r[0] for r in display_regions]
    scores = [r[1]['ergonomic_score'] for r in display_regions]
    
    # Truncate long region names
    display_names = [name[:15] + '...' if len(name) > 15 else name for name in region_names]
    
    # Create horizontal bar chart with color gradient based on score
    cmap = plt.cm.get_cmap('RdYlGn')
    score_colors = [cmap(score/100) for score in scores]
    
    bars = ax_scores.barh(display_names, scores, color=score_colors)
    
    # Add score values
    for bar in bars:
        width = bar.get_width()
        ax_scores.text(width + 1, bar.get_y() + bar.get_height()/2, 
                      f"{width:.1f}", va='center', fontweight='bold')
    
    # Add red-yellow-green background zones
    ax_scores.axvspan(0, 60, alpha=0.1, color='red')
    ax_scores.axvspan(60, 75, alpha=0.1, color='yellow')
    ax_scores.axvspan(75, 100, alpha=0.1, color='green')
    
    # Add text labels for zones
    ax_scores.text(30, -0.5, get_text("Poor", language), ha='center', va='top', color='red', fontsize=10)
    ax_scores.text(67.5, -0.5, get_text("Fair", language), ha='center', va='top', color='orange', fontsize=10)
    ax_scores.text(87.5, -0.5, get_text("Good", language), ha='center', va='top', color='green', fontsize=10)
    
    ax_scores.set_xlabel(get_text('Ergonomic Score', language), fontsize=12)
    ax_scores.set_title(get_text('Top Regions by Ergonomic Score', language), fontsize=14)
    ax_scores.set_xlim(0, 102)  # Leave room for labels
    
    # 2. Activity distribution heatmap
    ax_heatmap = fig.add_subplot(gs[0, 1])
    
    # Prepare data for heatmap
    heatmap_data = []
    for region, data in display_regions:
        activity_distribution = data['activity_distribution']
        row = {'region': region}
        
        for activity in ['Handle up', 'Handle down', 'Handle center']:
            if activity in activity_distribution:
                row[activity] = activity_distribution[activity]['percentage']
            else:
                row[activity] = 0
        
        heatmap_data.append(row)
    
    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index('region', inplace=True)
    
    # Truncate index names
    heatmap_df.index = [idx[:15] + '...' if len(idx) > 15 else idx for idx in heatmap_df.index]
    
    # Create heatmap
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".1f", 
        cmap="YlOrRd", 
        cbar_kws={'label': get_text('Percentage of Time (%)', language)},
        ax=ax_heatmap
    )
    
    ax_heatmap.set_title(get_text('Activity Distribution by Region (%)', language), fontsize=14)
    ax_heatmap.set_ylabel('')  # Remove y-label as it's redundant with the bar chart
    
    # 3. Region usage bubble chart
    ax_bubble = fig.add_subplot(gs[1, 0])
    
    # Prepare bubble chart data
    bubble_data = []
    for region, data in display_regions:
        bubble_data.append({
            'region': region[:10] + '...' if len(region) > 10 else region,
            'score': data['ergonomic_score'],
            'hours': data['total_duration'] / 3600,
            'employees': data['qualified_employees']
        })
    
    # Convert to DataFrame
    bubble_df = pd.DataFrame(bubble_data)
    
    # Create bubble chart - score vs hours with bubble size as employee count
    scatter = ax_bubble.scatter(
        bubble_df['hours'], 
        bubble_df['score'], 
        s=bubble_df['employees'] * 100,  # Scale bubble size
        c=bubble_df['score'],  # Color by score
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )
    
    # Add region labels
    for i, row in bubble_df.iterrows():
        ax_bubble.annotate(
            row['region'], 
            (row['hours'], row['score']),
            xytext=(7, 0),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax_bubble)
    cbar.set_label(get_text('Ergonomic Score', language))
    
    # Add a legend for bubble size
    sizes = [1, 3, 5, 7]
    labels = [f"{s} {get_text('employees', language)}" for s in sizes]
    legend_bubbles = []
    for size in sizes:
        legend_bubbles.append(ax_bubble.scatter([], [], s=size*100, c='gray', alpha=0.7, edgecolors='black'))
    
    ax_bubble.legend(legend_bubbles, labels, loc='lower right', title=get_text('Employees', language))
    
    ax_bubble.set_xlabel(get_text('Usage (hours)', language), fontsize=12)
    ax_bubble.set_ylabel(get_text('Ergonomic Score', language), fontsize=12)
    ax_bubble.set_title(get_text('Region Usage vs. Ergonomic Score', language), fontsize=14)
    ax_bubble.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Key metrics table
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_metrics.axis('off')
    
    # Prepare metrics data
    metrics_data = [[
        get_text("Region", language),
        get_text("Score", language),
        get_text("Hours", language),
        get_text("Employees", language),
        get_text("Handle up %", language),
        get_text("Handle down %", language)
    ]]
    
    for region, data in display_regions:
        activity_distribution = data['activity_distribution']
        handle_up_pct = activity_distribution.get('Handle up', {}).get('percentage', 0)
        handle_down_pct = activity_distribution.get('Handle down', {}).get('percentage', 0)
        
        metrics_data.append([
            region[:15] + '...' if len(region) > 15 else region,
            f"{data['ergonomic_score']:.1f}",
            f"{data['total_duration']/3600:.1f}",
            str(data['qualified_employees']),
            f"{handle_up_pct:.1f}%",
            f"{handle_down_pct:.1f}%"
        ])
    
    # Create table
    metrics_table = ax_metrics.table(
        cellText=metrics_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Style table
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(10)
    metrics_table.scale(1, 1.5)
    
    # Header formatting
    for j in range(len(metrics_data[0])):
        metrics_table[(0, j)].set_facecolor('#4472C4')
        metrics_table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color code scores
    for i in range(1, len(metrics_data)):
        score = float(metrics_data[i][1])
        if score >= 75:
            metrics_table[(i, 1)].set_facecolor('lightgreen')
        elif score >= 60:
            metrics_table[(i, 1)].set_facecolor('lightyellow')
        else:
            metrics_table[(i, 1)].set_facecolor('lightcoral')
    
    ax_metrics.set_title(get_text('Region Ergonomic Metrics Summary', language), fontsize=14)
    
    # Add main title
    plt.suptitle(get_text("Bakery Region Ergonomic Comparison", language), fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_figure(fig, output_path / "region_ergonomic_comparison.png")

def generate_circular_packing_visualization(employee_scores, region_analyses, output_dir, language='en'):
    """
    Generate a treemap visualization showing employees and regions 
    grouped by ergonomic score
    
    Parameters:
    -----------
    employee_scores : list
        List of employee ergonomic score dictionaries
    region_analyses : dict
        Dictionary of region analyses
    output_dir : Path
        Directory to save the visualization
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    try:
        import squarify
    except ImportError:
        print("Warning: squarify module not found. Installing required packages for treemap visualization.")
        import subprocess
        subprocess.check_call(["pip", "install", "squarify"])
        import squarify
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Employee treemap
    ax_emp = fig.add_subplot(1, 2, 1)
    
    # Group employees by score range
    score_ranges = {
        "Good (75-100)": [],
        "Fair (60-75)": [],
        "Poor (0-60)": []
    }
    
    for emp in employee_scores:
        score = emp['final_score']
        emp_id = emp['employee_id']
        hours = emp['total_duration'] / 3600
        
        if score >= 75:
            score_ranges["Good (75-100)"].append((emp_id, score, hours))
        elif score >= 60:
            score_ranges["Fair (60-75)"].append((emp_id, score, hours))
        else:
            score_ranges["Poor (0-60)"].append((emp_id, score, hours))
    
    # Prepare data for treemap
    treemap_data = []
    labels = []
    colors = []
    
    # Create nested data for treemap
    for category, employees in score_ranges.items():
        if not employees:
            continue
            
        # Sort employees by score
        sorted_emps = sorted(employees, key=lambda x: x[1], reverse=True)
        
        # Add entries for each employee
        for emp_id, score, hours in sorted_emps:
            # Value for treemap is hours worked
            treemap_data.append(hours)
            
            # Label is employee ID and score
            labels.append(f"{emp_id}\n{score:.1f}")
            
            # Color based on score
            if score >= 75:
                colors.append('#78C47B')  # Good - Green
            elif score >= 60:
                colors.append('#F0CF57')  # Fair - Yellow
            else:
                colors.append('#E07F70')  # Poor - Red
    
    # Create treemap
    if treemap_data:
        squarify.plot(
            sizes=treemap_data,
            label=labels,
            color=colors,
            alpha=0.8,
            pad=True,
            ax=ax_emp
        )
    
    ax_emp.set_xticks([])
    ax_emp.set_yticks([])
    ax_emp.set_title(get_text("Employee Ergonomic Scores", language), fontsize=16)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#78C47B', edgecolor='k', alpha=0.8, label=get_text('Good (75-100)', language)),
        patches.Patch(facecolor='#F0CF57', edgecolor='k', alpha=0.8, label=get_text('Fair (60-75)', language)),
        patches.Patch(facecolor='#E07F70', edgecolor='k', alpha=0.8, label=get_text('Poor (0-60)', language))
    ]
    
    ax_emp.legend(handles=legend_elements, loc='upper right')
    
    # 2. Region treemap
    ax_region = fig.add_subplot(1, 2, 2)
    
    # Filter valid regions
    valid_regions = {r: data for r, data in region_analyses.items() if 'ergonomic_score' in data}
    
    # Group regions by score range
    region_score_ranges = {
        "Good (75-100)": [],
        "Fair (60-75)": [],
        "Poor (0-60)": []
    }
    
    for region, data in valid_regions.items():
        score = data['ergonomic_score']
        hours = data['total_duration'] / 3600
        employees = data['qualified_employees']
        
        if score >= 75:
            region_score_ranges["Good (75-100)"].append((region, score, hours, employees))
        elif score >= 60:
            region_score_ranges["Fair (60-75)"].append((region, score, hours, employees))
        else:
            region_score_ranges["Poor (0-60)"].append((region, score, hours, employees))
    
    # Prepare data for treemap
    region_treemap_data = []
    region_labels = []
    region_colors = []
    
    # Process each category
    for category, regions in region_score_ranges.items():
        if not regions:
            continue
            
        # Sort regions by score
        sorted_regions = sorted(regions, key=lambda x: x[1], reverse=True)
        
        # Take top 10 regions in each category to avoid overcrowding
        display_regions = sorted_regions[:10]
        
        for region, score, hours, employees in display_regions:
            # Value is hours used
            region_treemap_data.append(hours)
            
            # Truncate long region names
            display_name = region[:12] + '...' if len(region) > 12 else region
            
            # Label includes region name, score and employee count
            region_labels.append(f"{display_name}\n{score:.1f} ({employees})")
            
            # Color based on score
            if score >= 75:
                region_colors.append('#78C47B')  # Good - Green
            elif score >= 60:
                region_colors.append('#F0CF57')  # Fair - Yellow
            else:
                region_colors.append('#E07F70')  # Poor - Red
    
    # Create treemap
    if region_treemap_data:
        squarify.plot(
            sizes=region_treemap_data,
            label=region_labels,
            color=region_colors,
            alpha=0.8,
            pad=True,
            ax=ax_region
        )
    
    ax_region.set_xticks([])
    ax_region.set_yticks([])
    ax_region.set_title(get_text("Region Ergonomic Scores", language), fontsize=16)
    
    # Add legend
    region_legend_elements = [
        patches.Patch(facecolor='#78C47B', edgecolor='k', alpha=0.8, label=get_text('Good (75-100)', language)),
        patches.Patch(facecolor='#F0CF57', edgecolor='k', alpha=0.8, label=get_text('Fair (60-75)', language)),
        patches.Patch(facecolor='#E07F70', edgecolor='k', alpha=0.8, label=get_text('Poor (0-60)', language))
    ]
    
    ax_region.legend(handles=region_legend_elements, loc='upper right')
    
    # Add main title
    plt.suptitle(get_text("Bakery Ergonomic Overview", language), fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    save_figure(fig, output_path / "ergonomic_overview_treemap.png")

def generate_employee_handling_comparison_by_region(data, region_analyses, output_dir, language='en'):
    """
    Generate bar charts comparing employee handling activities across regions
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    region_analyses : dict
        Dictionary of region analyses from analyze_region_ergonomics
    output_dir : Path
        Directory to save the visualizations
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    output_path = Path(output_dir) / 'region_employee_comparisons'
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Apply consistent visualization styling
    set_visualization_style()
    
    # Get standard colors
    activity_colors = get_activity_colors()
    employee_colors = get_employee_colors()
    
    # Filter data to only include handling activities
    handling_activities = ['Handle up', 'Handle center', 'Handle down']
    handling_data = data[data['activity'].isin(handling_activities)]
    
    # Sort regions by ergonomic score (if available)
    if region_analyses:
        sorted_regions = sorted(
            [(r, data['ergonomic_score']) for r, data in region_analyses.items() if 'ergonomic_score' in data], 
            key=lambda x: x[1], 
            reverse=True
        )
        regions_to_process = [r[0] for r in sorted_regions]  # Process all regions in order
    else:
        # If no region analyses provided, process regions with most handling activity
        region_handling_time = handling_data.groupby('region')['duration'].sum().sort_values(ascending=False)
        regions_to_process = region_handling_time.index.tolist()
    
    # Minimum percentage threshold for including employees
    min_percentage = 5.0
    
    # List to keep track of regions with qualified employees
    valid_regions = []
    
    # Process each region
    for region in regions_to_process:
        # Filter data for this region
        region_data = handling_data[handling_data['region'] == region]
        
        if region_data.empty:
            continue
        
        # Calculate total duration for this region
        total_region_duration = region_data['duration'].sum()
        
        # Get employees who work significantly in this region (at least 5%)
        qualified_employees = []
        
        for emp_id in region_data['id'].unique():
            # Get all data for this employee
            all_emp_data = handling_data[handling_data['id'] == emp_id]
            emp_total_time = all_emp_data['duration'].sum()
            
            # Get time in this region for this employee
            emp_region_data = region_data[region_data['id'] == emp_id]
            emp_region_time = emp_region_data['duration'].sum()
            
            # Calculate percentage
            percentage = (emp_region_time / emp_total_time * 100) if emp_total_time > 0 else 0
            
            # Include only if employee spends at least 5% of their time here
            if percentage >= min_percentage:
                qualified_employees.append(emp_id)
        
        if not qualified_employees:
            continue
            
        # Add to valid regions list
        valid_regions.append(region)
        
        # Calculate total region hours
        total_region_hours = total_region_duration / 3600
        
        # Create figure - size depends on number of employees
        fig_width = max(10, min(20, 2 + len(qualified_employees) * 3))
        fig_height = 8
        fig, axes = plt.subplots(1, len(qualified_employees), figsize=(fig_width, fig_height), sharey=False)
        
        # Ensure axes is a list even with one employee
        if len(qualified_employees) == 1:
            axes = [axes]
        
        # Process each qualified employee
        for i, emp_id in enumerate(sorted(qualified_employees)):
            # Filter data for this employee in this region
            emp_data = region_data[region_data['id'] == emp_id]
            
            # Calculate activity durations
            activity_durations = {}
            for activity in handling_activities:
                act_data = emp_data[emp_data['activity'] == activity]
                activity_durations[activity] = act_data['duration'].sum() / 3600  # Convert to hours
            
            # Calculate employee's total time in this region
            emp_total_duration = emp_data['duration'].sum()
            emp_total_hours = emp_total_duration / 3600
            emp_percentage = (emp_total_duration / total_region_duration * 100) if total_region_duration > 0 else 0
            
            # Create bar chart
            ax = axes[i]
            activities = []
            hours = []
            colors = []
            
            for activity in handling_activities:
                activities.append(get_text(activity, language))
                hours.append(activity_durations.get(activity, 0))
                colors.append(activity_colors.get(activity, '#CCCCCC'))
            
            # Create bars without labels
            ax.bar(activities, hours, color=colors)
            
            # Set employee-specific title and color
            emp_color = employee_colors.get(emp_id, '#333333')
            ax.set_title(f"{emp_id}\n{emp_total_hours:.1f}h ({emp_percentage:.1f}%)", 
                       color=emp_color, fontweight='bold')
            
            # Customize appearance
            if i == 0:  # Only add y-label to first plot
                ax.set_ylabel(get_text('Hours', language))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add overall region title
        ergonomic_score = region_analyses.get(region, {}).get('ergonomic_score', None)
        if ergonomic_score is not None:
            score_color = 'green' if ergonomic_score >= 75 else ('orange' if ergonomic_score >= 60 else 'red')
            title = f"{get_text('Region', language)}: {region} - {get_text('Score', language)}: {ergonomic_score:.1f}/100"
            plt.suptitle(title, fontsize=16, color=score_color, y=0.98)
        else:
            plt.suptitle(f"{get_text('Region', language)}: {region}", fontsize=16, y=0.98)
        
        # Add subtitle with region total
        subtitle = f"{get_text('Total Hours', language)}: {total_region_hours:.1f} | "
        subtitle += f"{get_text('Employees', language)}: {len(qualified_employees)}"
        fig.text(0.5, 0.91, subtitle, ha='center', fontsize=12)
        
        # Add common legend at the bottom
        handles = [patches.Patch(color=activity_colors.get(act, '#CCCCCC'), label=get_text(act, language)) 
                for act in handling_activities]
        fig.legend(handles=handles, loc='lower center', ncol=len(handling_activities), 
                title=get_text('Activity', language), bbox_to_anchor=(0.5, 0.01))
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.05, 1, 0.9])
        save_figure(fig, output_path / f"{region}_employee_handling_comparison.png")
        
        # Report progress
        print(f"  {get_text('Generated employee handling comparison for region', language)}: {region}")
    
    # Create combined visualization for all valid regions
    if valid_regions:
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        
        # Calculate grid dimensions based on number of regions
        num_regions = len(valid_regions)
        num_cols = 2  # Arrange in 2 columns
        num_rows = (num_regions + num_cols - 1) // num_cols  # Ceiling division
        
        # Create subplot grid
        gs = gridspec.GridSpec(num_rows, num_cols, figure=fig)
        
        # Process each region
        for idx, region in enumerate(valid_regions):
            # Calculate grid position
            row = idx // num_cols
            col = idx % num_cols
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Filter data for this region
            region_data = handling_data[handling_data['region'] == region]
            
            if region_data.empty:
                continue
                
            # Calculate region total
            region_total = region_data['duration'].sum() / 3600  # hours
            
            # Get qualified employees for this region
            qualified_employees = []
            
            for emp_id in region_data['id'].unique():
                # Get all data for this employee
                all_emp_data = handling_data[handling_data['id'] == emp_id]
                emp_total_time = all_emp_data['duration'].sum()
                
                # Get time in this region for this employee
                emp_region_data = region_data[region_data['id'] == emp_id]
                emp_region_time = emp_region_data['duration'].sum()
                
                # Calculate percentage
                percentage = (emp_region_time / emp_total_time * 100) if emp_total_time > 0 else 0
                
                # Include only if employee spends at least 5% of their time here
                if percentage >= min_percentage:
                    qualified_employees.append(emp_id)
            
            if not qualified_employees:
                continue
            
            # Calculate activity breakdown by employee
            data_to_plot = []
            
            for emp_id in sorted(qualified_employees):
                emp_data = region_data[region_data['id'] == emp_id]
                
                row_data = {'Employee': emp_id}
                
                for activity in handling_activities:
                    act_data = emp_data[emp_data['activity'] == activity]
                    act_hours = act_data['duration'].sum() / 3600
                    row_data[activity] = act_hours
                
                data_to_plot.append(row_data)
            
            # Convert to DataFrame
            plot_df = pd.DataFrame(data_to_plot)
            if not plot_df.empty and 'Employee' in plot_df.columns:
                plot_df.set_index('Employee', inplace=True)
                
                # Create stacked bar chart
                plot_df.plot(
                    kind='bar', 
                    stacked=True, 
                    ax=ax, 
                    color=[activity_colors.get(col, '#CCCCCC') for col in plot_df.columns],
                    legend=False  # Don't show legend on individual plots
                )
                
                # Get ergonomic score if available
                ergonomic_score = region_analyses.get(region, {}).get('ergonomic_score', None)
                if ergonomic_score is not None:
                    score_color = 'green' if ergonomic_score >= 75 else ('orange' if ergonomic_score >= 60 else 'red')
                    title = f"{region} ({ergonomic_score:.1f}/100)"
                else:
                    score_color = 'black'
                    title = f"{region}"
                
                # Add title with region info
                ax.set_title(f"{title}\n{get_text('Total', language)}: {region_total:.1f}h", 
                           fontsize=12, color=score_color)
                
                # Customize appearance
                ax.set_ylabel(get_text('Hours', language))
                ax.set_xlabel('')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add common legend at the bottom
        handles = [patches.Patch(color=activity_colors.get(act, '#CCCCCC'), label=get_text(act, language)) 
                for act in handling_activities]
        fig.legend(handles=handles, loc='lower center', ncol=len(handling_activities), 
                title=get_text('Activity', language), bbox_to_anchor=(0.5, 0.01))
        
        # Add overall title
        plt.suptitle(get_text('Employee Handling Activities in Top Regions', language), fontsize=18, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure
        save_figure(fig, output_path / "all_regions_employee_handling_comparison.png")
        print(f"  {get_text('Generated combined handling comparison for all regions', language)}")