"""
Ergonomics Data Analysis Module

Contains functions for analyzing activity durations, handling patterns, and calculating ergonomic scores.
"""

import pandas as pd
import numpy as np
from src.utils.time_utils import format_seconds_to_hms
from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import get_text

from .utilities import create_region_categories, create_region_clusters

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