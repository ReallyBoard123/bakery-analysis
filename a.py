#!/usr/bin/env python3
"""
Comprehensive script to analyze handling time patterns in bakery employee data,
including regression analysis and visualizations.
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# Add project root to path
sys.path.append(os.getcwd())

try:
    from src.utils.time_utils import format_seconds_to_hms
    from src.utils.data_utils import load_sensor_data, classify_departments
except ImportError:
    # Fallback implementations if imports fail
    def format_seconds_to_hms(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def load_sensor_data(file_path):
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} records from {file_path}")
        return data
    
    def classify_departments(data):
        data_with_dept = data.copy()
        bread_dept = ['32-A', '32-C', '32-D', '32-E']
        cake_dept = ['32-F', '32-G', '32-H']
        data_with_dept['department'] = data_with_dept['id'].apply(
            lambda x: 'Bread' if x in bread_dept else ('Cake' if x in cake_dept else 'Other')
        )
        return data_with_dept

def analyze_handling_time_in_top_regions(data):
    """Step 1: Analyze handling time in top regions per employee"""
    
    # List all employees
    employees = sorted(data['id'].unique())
    print(f"Analyzing data for {len(employees)} employees: {', '.join(employees)}")
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # Find top 5 regions for each employee
    employee_top_regions = {}
    for employee_id in employees:
        emp_data = data[data['id'] == employee_id]
        region_times = emp_data.groupby('region')['duration'].sum().reset_index()
        top_regions = region_times.sort_values('duration', ascending=False).head(5)['region'].tolist()
        employee_top_regions[employee_id] = top_regions
        print(f"  Employee {employee_id} top 5 regions: {', '.join(top_regions)}")
    
    # Get all unique top regions
    all_top_regions = set()
    for regions in employee_top_regions.values():
        all_top_regions.update(regions)
    all_top_regions = sorted(all_top_regions)
    
    print(f"\nIdentified {len(all_top_regions)} unique top regions across all employees")
    
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
    
    # Print results
    print("\nSTEP 1: Total Handling Time by Category in Top Regions (All Employees Combined)")
    print("=" * 100)
    print(formatted_pivot)
    
    # Calculate and display totals by category
    print("\nTotal Handling Time by Category (across all top regions):")
    print("-" * 50)
    
    grand_total = 0
    for category in handling_categories:
        if category in pivot_table.columns:
            duration = pivot_table[category].sum()
            grand_total += duration
            print(f"{category}: {format_seconds_to_hms(duration)}")
    
    print("-" * 50)
    print(f"Grand Total: {format_seconds_to_hms(grand_total)}")
    
    return pivot_table, employee_top_regions, results_df

def analyze_workstation_categories(data):
    """Step 2: Analyze data by workstation category"""
    
    # Define region categories
    region_categories = {
        'Bread Workstation': ['3_Brotstation', '4_Brotstation', '1_Brotstation', '2_Brotstation'],
        'Cake Workstation': ['1_Konditorei_station', '2_Konditorei_station', '3_Konditorei_station', 'konditorei_deco'],
        'Bread Baking': ['bereitstellen_prepared_goods', 'Etagenofen', 'Rollenhandtuchspend', 'Gärraum'],
        'Cake Baking': ['Fettbacken', 'cookies_packing_hand'],
        'Common Regions': ['brotkisten_regal', 'production_plan', 'leere_Kisten', '1_corridor']
    }
    
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
    
    print("\nSTEP 2: Handling Time by Workstation Category")
    print("=" * 100)
    print(formatted_df)
    
    return df, region_categories

def analyze_handling_position_patterns(data, region_categories):
    """Step 3: Statistical analysis of handling positions by workstation type"""
    
    print("\nSTEP 3: Statistical Analysis of Handling Positions")
    print("=" * 100)
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # For comparing bread vs cake workstations
    bread_ws_regions = region_categories['Bread Workstation']
    cake_ws_regions = region_categories['Cake Workstation']
    
    # Extract detailed handling data for statistical tests
    bread_ws_data = data[data['region'].isin(bread_ws_regions) & data['activity'].isin(handling_categories)]
    cake_ws_data = data[data['region'].isin(cake_ws_regions) & data['activity'].isin(handling_categories)]
    
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
    
    bread_percentages = {pos: (time/bread_total*100) for pos, time in bread_by_position.items()}
    cake_percentages = {pos: (time/cake_total*100) for pos, time in cake_by_position.items()}
    
    # Perform chi-square test on the distribution of handling positions
    observed = np.array([
        [bread_by_position['Handle up'], bread_by_position['Handle center'], bread_by_position['Handle down']],
        [cake_by_position['Handle up'], cake_by_position['Handle center'], cake_by_position['Handle down']]
    ])
    
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    
    # Create results dataframe
    comparison = pd.DataFrame({
        'Position': handling_categories,
        'Bread Workstation (sec)': [bread_by_position[pos] for pos in handling_categories],
        'Bread Workstation (%)': [bread_percentages[pos] for pos in handling_categories],
        'Cake Workstation (sec)': [cake_by_position[pos] for pos in handling_categories],
        'Cake Workstation (%)': [cake_percentages[pos] for pos in handling_categories],
        'Ratio (Cake/Bread)': [cake_by_position[pos]/bread_by_position[pos] if bread_by_position[pos] > 0 else float('inf') 
                              for pos in handling_categories]
    })
    
    # Format time values
    comparison['Bread Workstation (formatted)'] = comparison['Bread Workstation (sec)'].apply(format_seconds_to_hms)
    comparison['Cake Workstation (formatted)'] = comparison['Cake Workstation (sec)'].apply(format_seconds_to_hms)
    
    # Round percentages
    comparison['Bread Workstation (%)'] = comparison['Bread Workstation (%)'].round(1)
    comparison['Cake Workstation (%)'] = comparison['Cake Workstation (%)'].round(1)
    
    # Print results
    print("\nComparison of Handling Positions: Bread vs. Cake Workstations")
    print(comparison[['Position', 'Bread Workstation (formatted)', 'Bread Workstation (%)', 
                    'Cake Workstation (formatted)', 'Cake Workstation (%)', 'Ratio (Cake/Bread)']])
    
    print(f"\nChi-Square Test Results:")
    print(f"Chi² value: {chi2:.2f}")
    print(f"p-value: {p:.6f}")
    print(f"Degrees of freedom: {dof}")
    
    # Interpret results
    alpha = 0.05
    if p < alpha:
        print("\nSTATISTICAL INFERENCE: There is a statistically significant difference between")
        print("handling position distributions at bread and cake workstations (p < 0.05).")
    else:
        print("\nSTATISTICAL INFERENCE: There is no statistically significant difference between")
        print("handling position distributions at bread and cake workstations (p >= 0.05).")
    
    # Analyze the key differences
    print("\nKey findings:")
    
    # Handle up comparison
    up_bread_pct = bread_percentages['Handle up']
    up_cake_pct = cake_percentages['Handle up']
    
    if up_cake_pct > up_bread_pct:
        diff_pct = up_cake_pct - up_bread_pct
        print(f"- Cake workstations involve {diff_pct:.1f}% more 'Handle up' positions compared to bread workstations")
        print(f"  ({up_cake_pct:.1f}% vs {up_bread_pct:.1f}%)")
    else:
        diff_pct = up_bread_pct - up_cake_pct
        print(f"- Bread workstations involve {diff_pct:.1f}% more 'Handle up' positions compared to cake workstations")
        print(f"  ({up_bread_pct:.1f}% vs {up_cake_pct:.1f}%)")
    
    # Handle down comparison
    down_bread_pct = bread_percentages['Handle down']
    down_cake_pct = cake_percentages['Handle down']
    
    if down_bread_pct > down_cake_pct:
        diff_pct = down_bread_pct - down_cake_pct
        print(f"- Bread workstations involve {diff_pct:.1f}% more 'Handle down' positions compared to cake workstations")
        print(f"  ({down_bread_pct:.1f}% vs {down_cake_pct:.1f}%)")
    else:
        diff_pct = down_cake_pct - down_bread_pct
        print(f"- Cake workstations involve {diff_pct:.1f}% more 'Handle down' positions compared to cake workstations")
        print(f"  ({down_cake_pct:.1f}% vs {down_bread_pct:.1f}%)")
    
    # Handle center comparison
    center_bread_pct = bread_percentages['Handle center']
    center_cake_pct = cake_percentages['Handle center']
    
    if center_bread_pct > center_cake_pct:
        diff_pct = center_bread_pct - center_cake_pct
        print(f"- Bread workstations involve {diff_pct:.1f}% more 'Handle center' positions compared to cake workstations")
        print(f"  ({center_bread_pct:.1f}% vs {center_cake_pct:.1f}%)")
    else:
        diff_pct = center_cake_pct - center_bread_pct
        print(f"- Cake workstations involve {diff_pct:.1f}% more 'Handle center' positions compared to cake workstations")
        print(f"  ({center_cake_pct:.1f}% vs {center_bread_pct:.1f}%)")
    
    # Provide ergonomic implications
    print("\nErgonomic implications:")
    if up_cake_pct > up_bread_pct:
        print("- Cake workstations may require more ergonomic attention for overhead reaching tasks")
    if down_bread_pct > down_cake_pct:
        print("- Bread workstations may require more ergonomic attention for low-level handling tasks")
    
    return comparison

def run_regression_analysis(data, region_categories):
    """Step 4: Analysis of employees within their specialized departments"""
    
    print("\nSTEP 4: Department-Based Analysis - Employee Specialization")
    print("=" * 100)
    
    # Handling activity categories
    handling_categories = ['Handle up', 'Handle center', 'Handle down']
    
    # Get bread and cake regions
    bread_regions = region_categories['Bread Workstation']
    cake_regions = region_categories['Cake Workstation']
    
    # Get employees by department
    bread_employees = ['32-A', '32-C', '32-D', '32-E']  # Based on department classification
    cake_employees = ['32-F', '32-G', '32-H']
    
    # Analysis 1: "How do individuals differ from their peers in their specialized area?"
    print("\nAnalysis 1: Individual differences within specialized departments")
    print("-" * 80)
    
    # Analyze bread department employees
    bread_emp_data = []
    for emp_id in bread_employees:
        emp_data = data[data['id'] == emp_id]
        bread_workstation_data = emp_data[emp_data['region'].isin(bread_regions) & 
                                        emp_data['activity'].isin(handling_categories)]
        
        bread_total = bread_workstation_data['duration'].sum()
        
        # Skip if insufficient data
        if bread_total < 900:  # Reduced threshold to 15 minutes
            print(f"Employee {emp_id} has insufficient data in bread workstations (<15 min)")
            continue
            
        # Calculate position percentages
        position_data = {}
        for position in handling_categories:
            position_time = bread_workstation_data[bread_workstation_data['activity'] == position]['duration'].sum()
            position_data[position] = (position_time/bread_total*100) if bread_total > 0 else 0
        
        bread_emp_data.append({
            'employee': emp_id,
            'total_hours': bread_total/3600,
            'up_pct': position_data['Handle up'],
            'center_pct': position_data['Handle center'],
            'down_pct': position_data['Handle down']
        })
    
    # Create bread employee summary
    if bread_emp_data:
        bread_summary = pd.DataFrame(bread_emp_data)
        
        # Calculate department averages
        bread_avg_up = bread_summary['up_pct'].mean()
        bread_avg_center = bread_summary['center_pct'].mean()
        bread_avg_down = bread_summary['down_pct'].mean()
        
        # Calculate deviations from department average
        bread_summary['up_dev'] = bread_summary['up_pct'] - bread_avg_up
        bread_summary['center_dev'] = bread_summary['center_pct'] - bread_avg_center
        bread_summary['down_dev'] = bread_summary['down_pct'] - bread_avg_down
        
        print("\nBread Department Employees (Working in Bread Workstations):")
        print(f"Department Averages: Up {bread_avg_up:.1f}%, Center {bread_avg_center:.1f}%, Down {bread_avg_down:.1f}%")
        print(bread_summary[['employee', 'total_hours', 'up_pct', 'center_pct', 'down_pct']]
              .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        
        # Identify outliers
        significant_deviation = 10.0  # Lower threshold for within-department comparison
        outliers = bread_summary[(abs(bread_summary['up_dev']) > significant_deviation) | 
                              (abs(bread_summary['center_dev']) > significant_deviation) |
                              (abs(bread_summary['down_dev']) > significant_deviation)]
        
        if not outliers.empty:
            print("\nBread employees with notable deviations from department average:")
            for _, row in outliers.iterrows():
                print(f"  Employee {row['employee']}:")
                for pos, avg in [('up', bread_avg_up), ('center', bread_avg_center), ('down', bread_avg_down)]:
                    deviation = row[f'{pos}_dev']
                    if abs(deviation) > significant_deviation:
                        print(f"    Handle {pos}: {deviation:+.1f}% from department average")
        else:
            print("\nAll bread employees follow similar handling patterns within their department")
    else:
        print("Insufficient data for bread department employees analysis")
    
    # Analyze cake department employees
    cake_emp_data = []
    for emp_id in cake_employees:
        emp_data = data[data['id'] == emp_id]
        cake_workstation_data = emp_data[emp_data['region'].isin(cake_regions) & 
                                       emp_data['activity'].isin(handling_categories)]
        
        cake_total = cake_workstation_data['duration'].sum()
        
        # Skip if insufficient data
        if cake_total < 900:  # Reduced threshold to 15 minutes
            print(f"Employee {emp_id} has insufficient data in cake workstations (<15 min)")
            continue
            
        # Calculate position percentages
        position_data = {}
        for position in handling_categories:
            position_time = cake_workstation_data[cake_workstation_data['activity'] == position]['duration'].sum()
            position_data[position] = (position_time/cake_total*100) if cake_total > 0 else 0
        
        cake_emp_data.append({
            'employee': emp_id,
            'total_hours': cake_total/3600,
            'up_pct': position_data['Handle up'],
            'center_pct': position_data['Handle center'],
            'down_pct': position_data['Handle down']
        })
    
    # Create cake employee summary
    if cake_emp_data:
        cake_summary = pd.DataFrame(cake_emp_data)
        
        # Calculate department averages
        cake_avg_up = cake_summary['up_pct'].mean()
        cake_avg_center = cake_summary['center_pct'].mean()
        cake_avg_down = cake_summary['down_pct'].mean()
        
        # Calculate deviations from department average
        cake_summary['up_dev'] = cake_summary['up_pct'] - cake_avg_up
        cake_summary['center_dev'] = cake_summary['center_pct'] - cake_avg_center
        cake_summary['down_dev'] = cake_summary['down_pct'] - cake_avg_down
        
        print("\nCake Department Employees (Working in Cake Workstations):")
        print(f"Department Averages: Up {cake_avg_up:.1f}%, Center {cake_avg_center:.1f}%, Down {cake_avg_down:.1f}%")
        print(cake_summary[['employee', 'total_hours', 'up_pct', 'center_pct', 'down_pct']]
              .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        
        # Identify outliers
        significant_deviation = 10.0  # Lower threshold for within-department comparison
        outliers = cake_summary[(abs(cake_summary['up_dev']) > significant_deviation) | 
                             (abs(cake_summary['center_dev']) > significant_deviation) |
                             (abs(cake_summary['down_dev']) > significant_deviation)]
        
        if not outliers.empty:
            print("\nCake employees with notable deviations from department average:")
            for _, row in outliers.iterrows():
                print(f"  Employee {row['employee']}:")
                for pos, avg in [('up', cake_avg_up), ('center', cake_avg_center), ('down', cake_avg_down)]:
                    deviation = row[f'{pos}_dev']
                    if abs(deviation) > significant_deviation:
                        print(f"    Handle {pos}: {deviation:+.1f}% from department average")
        else:
            print("\nAll cake employees follow similar handling patterns within their department")
    else:
        print("Insufficient data for cake department employees analysis")
    
    # Analysis 2: "How do bread workers' patterns differ from cake workers' patterns in their specialized environments?"
    print("\nAnalysis 2: Department differences in specialized environments")
    print("-" * 80)
    
    # Combine summaries if both have data
    if bread_emp_data and cake_emp_data:
        # Prepare for department comparison
        if 'bread_summary' in locals() and 'cake_summary' in locals():
            print("\nDepartment Comparison (employees in their specialized workstations):")
            print(f"Bread Department (n={len(bread_summary)}): "
                  f"Up {bread_avg_up:.1f}%, Center {bread_avg_center:.1f}%, Down {bread_avg_down:.1f}%")
            print(f"Cake Department (n={len(cake_summary)}): "
                  f"Up {cake_avg_up:.1f}%, Center {cake_avg_center:.1f}%, Down {cake_avg_down:.1f}%")
            
            # Calculate differences
            up_diff = cake_avg_up - bread_avg_up
            center_diff = cake_avg_center - bread_avg_center
            down_diff = cake_avg_down - bread_avg_down
            
            # Print differences with direction
            print("\nKey differences (Cake vs. Bread):")
            print(f"  Handle up: {up_diff:+.1f}% {'higher' if up_diff > 0 else 'lower'} in cake department")
            print(f"  Handle center: {center_diff:+.1f}% {'higher' if center_diff > 0 else 'lower'} in cake department")
            print(f"  Handle down: {down_diff:+.1f}% {'higher' if down_diff > 0 else 'lower'} in cake department")
            
            # Run statistical test if sufficient data
            if len(bread_summary) >= 2 and len(cake_summary) >= 2:
                try:
                    from scipy import stats
                    # T-test for Handle up percentage
                    t_stat, p_value = stats.ttest_ind(
                        bread_summary['up_pct'], 
                        cake_summary['up_pct'], 
                        equal_var=False
                    )
                    print(f"\nStatistical Analysis:")
                    print(f"  Handle up: t={t_stat:.2f}, p={p_value:.4f} "
                          f"({'significant' if p_value < 0.05 else 'not significant'})")
                    
                    # T-test for Handle center percentage
                    t_stat, p_value = stats.ttest_ind(
                        bread_summary['center_pct'], 
                        cake_summary['center_pct'], 
                        equal_var=False
                    )
                    print(f"  Handle center: t={t_stat:.2f}, p={p_value:.4f} "
                          f"({'significant' if p_value < 0.05 else 'not significant'})")
                    
                    # T-test for Handle down percentage
                    t_stat, p_value = stats.ttest_ind(
                        bread_summary['down_pct'], 
                        cake_summary['down_pct'], 
                        equal_var=False
                    )
                    print(f"  Handle down: t={t_stat:.2f}, p={p_value:.4f} "
                          f"({'significant' if p_value < 0.05 else 'not significant'})")
                except Exception as e:
                    print(f"Could not run statistical tests: {e}")
            else:
                print("\nInsufficient data for statistical tests between departments")
    else:
        print("Insufficient data to compare departments")
    
    # Create a model with region as a nested factor if data permits
    model_results = {}
    try:
        # Get aggregate data for all employees in their specialized regions
        specialized_data = data[
            ((data['id'].isin(bread_employees)) & (data['region'].isin(bread_regions)) |
             (data['id'].isin(cake_employees)) & (data['region'].isin(cake_regions))) &
            (data['activity'].isin(handling_categories))
        ]
        
        # Add department column
        specialized_data['department'] = specialized_data['id'].apply(
            lambda x: 'Bread' if x in bread_employees else 'Cake'
        )
        
        # Create position-specific datasets
        for position in handling_categories:
            position_data = specialized_data[specialized_data['activity'] == position].copy()
            
            # Calculate total duration per employee-region
            total_by_emp_region = position_data.groupby(['id', 'region'])['duration'].sum().reset_index()
            total_by_emp_region.rename(columns={'duration': 'total_duration'}, inplace=True)
            
            # Merge back to get proportions
            position_data = pd.merge(position_data, total_by_emp_region, on=['id', 'region'])
            position_data['proportion'] = position_data['duration'] / position_data['total_duration']
            
            # Add department indicator
            position_data['is_cake'] = (position_data['department'] == 'Cake').astype(int)
            
            # Run mixed model with position-specific results
            if len(position_data) > 10:  # Ensure minimum sample size
                print(f"\nMixed model analysis for '{position}' position:")
                
                try:
                    # Mixed model with employee as random effect and department as fixed effect
                    formula = 'proportion ~ is_cake'
                    md = smf.mixedlm(formula, position_data, groups=position_data["id"])
                    mdf = md.fit(reml=True)
                    
                    # Save model results
                    model_results[position] = {
                        'model': 'mixedlm',
                        'result': mdf,
                        'description': f"Mixed-effects model for {position}",
                        'cake_effect': mdf.params['is_cake'],
                        'p_value': mdf.pvalues['is_cake']
                    }
                    
                    # Extract key results
                    cake_effect = mdf.params['is_cake']
                    p_value = mdf.pvalues['is_cake']
                    significance = "significant" if p_value < 0.05 else "not significant"
                    
                    print(f"  Department effect (cake vs bread): {cake_effect:+.3f} ({significance}, p={p_value:.4f})")
                    
                    # Variance components
                    group_var = mdf.cov_re.iloc[0, 0]
                    resid_var = mdf.scale
                    total_var = group_var + resid_var
                    icc = group_var / total_var
                    
                    print(f"  Employee effect: {group_var/total_var*100:.1f}% of variance")
                    print(f"  Department/Region effect: {resid_var/total_var*100:.1f}% of variance")
                    
                    if icc > 0.5:
                        print("  Employee differences explain most of the variation in handling patterns")
                    elif icc > 0.25:
                        print("  Both employee and department/region factors contribute to handling patterns")
                    else:
                        print("  Department and region differences explain most of the variation")
                    
                except Exception as e:
                    print(f"  Error fitting model: {e}")
    except Exception as e:
        print(f"Error in specialized analysis: {e}")
    
    # Collect summary dataframes for visualization
    dept_summaries = {}
    if 'bread_summary' in locals():
        dept_summaries['Bread'] = bread_summary
    if 'cake_summary' in locals():
        dept_summaries['Cake'] = cake_summary
    
    return model_results, dept_summaries

def create_visualizations(pivot_table, category_df, model_results, agg_data, region_categories):
    """Step 5: Create visualizations to illustrate the findings"""
    
    print("\nSTEP 5: Creating Visualizations")
    print("=" * 100)
    
    # Ensure output directory exists
    output_dir = Path('output/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Workstation Category Comparison
    plt.figure(figsize=(12, 6))
    
    # Create a modified dataframe for plotting
    plot_df = category_df.copy()
    plot_df = plot_df[['Category', 'Handle up %', 'Handle center %', 'Handle down %']]
    
    # Sort by workstation category type (Bread, Cake, etc.)
    # Define a custom sort order
    category_order = ['Bread Workstation', 'Cake Workstation', 'Bread Baking', 'Cake Baking', 'Common Regions']
    plot_df['sort_order'] = plot_df['Category'].apply(lambda x: category_order.index(x) if x in category_order else 999)
    plot_df = plot_df.sort_values('sort_order')
    plot_df.drop('sort_order', axis=1, inplace=True)
    
    # Create stacked bar chart
    ax = plot_df.set_index('Category').plot(kind='bar', stacked=True, figsize=(12, 6),
                                         color=['#ff9999', '#99cc99', '#9999ff'])
    
    plt.title('Handling Position Distribution by Workstation Category', fontsize=16)
    plt.xlabel('Workstation Category', fontsize=14)
    plt.ylabel('Percentage of Time', fontsize=14)
    plt.legend(title='Position Type')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / 'workstation_category_comparison.png', dpi=300)
    plt.close()
    print(f"Saved visualization to {output_dir}/workstation_category_comparison.png")
    
    # 2. Bread vs Cake Comparison
    plt.figure(figsize=(10, 6))
    
    # Extract bread and cake workstation data
    bread_cake_data = category_df[category_df['Category'].isin(['Bread Workstation', 'Cake Workstation'])]
    
    # Prepare data for grouped bar chart
    bread_row = bread_cake_data[bread_cake_data['Category'] == 'Bread Workstation'].iloc[0]
    cake_row = bread_cake_data[bread_cake_data['Category'] == 'Cake Workstation'].iloc[0]
    
    # Create positions and widths for grouped bars
    positions = np.arange(3)
    width = 0.35
    
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bread_values = [bread_row['Handle up %'], bread_row['Handle center %'], bread_row['Handle down %']]
    cake_values = [cake_row['Handle up %'], cake_row['Handle center %'], cake_row['Handle down %']]
    
    rects1 = ax.bar(positions - width/2, bread_values, width, label='Bread Workstation', color='#8c6464')
    rects2 = ax.bar(positions + width/2, cake_values, width, label='Cake Workstation', color='#ffd700')
    
    # Add labels and formatting
    ax.set_title('Bread vs. Cake Workstation Handling Patterns', fontsize=16)
    ax.set_ylabel('Percentage of Time (%)', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Handle up', 'Handle center', 'Handle down'])
    ax.legend()
    
    # Add value labels on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Add statistical significance indicators
    for i, position in enumerate(['Handle up', 'Handle center', 'Handle down']):
        if position in model_results and model_results[position]['p_value'] < 0.05:
            # Add a star for significance
            ax.text(i, max(bread_values[i], cake_values[i]) + 3, '*', 
                    ha='center', va='bottom', fontsize=24, color='red')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / 'bread_vs_cake_comparison.png', dpi=300)
    plt.close()
    print(f"Saved visualization to {output_dir}/bread_vs_cake_comparison.png")
    
    # 3. Create a heatmap showing handling patterns by employee and region
    # First, reshape the data to have employees as rows, regions as columns, and values as handling up %
    pivot_data = agg_data[agg_data['activity'] == 'Handle up'].pivot_table(
        index='id', 
        columns='region', 
        values='proportion',
        fill_value=0
    )
    
    # Keep only the most important regions (those in bread and cake workstations)
    important_regions = region_categories['Bread Workstation'] + region_categories['Cake Workstation']
    pivot_data = pivot_data[pivot_data.columns.intersection(important_regions)]
    
    # Add workstation type annotation to column names
    col_names = []
    for col in pivot_data.columns:
        if col in region_categories['Bread Workstation']:
            col_names.append(f"{col} (B)")
        elif col in region_categories['Cake Workstation']:
            col_names.append(f"{col} (C)")
        else:
            col_names.append(col)
    pivot_data.columns = col_names
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    
    # Create custom colormap from blue to red
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create the heatmap
    heatmap = sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap=cmap, 
                         linewidths=.5, vmin=0, vmax=1, center=0.5,
                         cbar_kws={'label': 'Proportion of "Handle up" Position'})
    
    plt.title('"Handle up" Proportion by Employee and Region', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(output_dir / 'employee_region_heatmap.png', dpi=300)
    plt.close()
    print(f"Saved visualization to {output_dir}/employee_region_heatmap.png")
    
    # 4. Boxplot showing variance in handling positions by region and employee
    plt.figure(figsize=(12, 6))
    
    # Filter for important regions
    boxplot_data = agg_data[agg_data['region'].isin(important_regions)]
    
    # Add region category for coloring
    boxplot_data['region_category'] = boxplot_data['region'].apply(
        lambda r: 'Bread' if r in region_categories['Bread Workstation'] 
                 else ('Cake' if r in region_categories['Cake Workstation'] else 'Other')
    )
    
    # Create the boxplot
    plt.figure(figsize=(14, 8))
    
    # Create a custom palette
    palette = {'Bread': '#8c6464', 'Cake': '#ffd700'}
    
    # Create a grouped boxplot
    ax = sns.boxplot(x='region', y='proportion', hue='activity', data=boxplot_data,
                   palette={'Handle up': '#ff9999', 'Handle center': '#99cc99', 'Handle down': '#9999ff'})
    
    # Color the boxes based on workstation type
    for i, box in enumerate(ax.artists):
        region_idx = i // 3  # 3 activities per region
        region = boxplot_data['region'].unique()[region_idx]
        region_type = 'Bread' if region in region_categories['Bread Workstation'] else 'Cake'
        box.set_edgecolor(palette[region_type])
        box.set_linewidth(2)
    
    plt.title('Variation in Handling Positions by Region', fontsize=16)
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Proportion of Time', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / 'position_variation_by_region.png', dpi=300)
    plt.close()
    print(f"Saved visualization to {output_dir}/position_variation_by_region.png")
    
    print(f"\nAll visualizations have been saved to the {output_dir} directory")

def create_department_visualizations(dept_summaries, region_categories):
    """Create visualizations for department-based analysis"""
    
    # Create directory for visualizations
    output_dir = Path('output/visualizations')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    dept_vis_dir = output_dir / 'department_analysis'
    dept_vis_dir.mkdir(exist_ok=True, parents=True)
    
    if not dept_summaries:
        print("No department data available for visualization")
        return
    
    # 1. Department comparison bar chart
    if 'Bread' in dept_summaries and 'Cake' in dept_summaries:
        # Calculate averages
        bread_summary = dept_summaries['Bread']
        cake_summary = dept_summaries['Cake']
        
        bread_avg_up = bread_summary['up_pct'].mean()
        bread_avg_center = bread_summary['center_pct'].mean()
        bread_avg_down = bread_summary['down_pct'].mean()
        
        cake_avg_up = cake_summary['up_pct'].mean()
        cake_avg_center = cake_summary['center_pct'].mean()
        cake_avg_down = cake_summary['down_pct'].mean()
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        positions = np.arange(3)
        width = 0.35
        
        bread_values = [bread_avg_up, bread_avg_center, bread_avg_down]
        cake_values = [cake_avg_up, cake_avg_center, cake_avg_down]
        
        rects1 = ax.bar(positions - width/2, bread_values, width, label='Bread Department', color='#8c6464')
        rects2 = ax.bar(positions + width/2, cake_values, width, label='Cake Department', color='#ffd700')
        
        # Add labels and formatting
        ax.set_title('Bread vs. Cake Department Handling Patterns\n(In Their Specialized Workstations)', fontsize=16)
        ax.set_ylabel('Percentage of Time (%)', fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Handle up', 'Handle center', 'Handle down'])
        ax.legend()
        
        # Add value labels
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        add_labels(rects1)
        add_labels(rects2)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(dept_vis_dir / 'department_comparison.png', dpi=300)
        plt.close()
        print(f"Saved department comparison to {dept_vis_dir}/department_comparison.png")
    
    # 2. Department individual employee variability
    for dept_name, dept_summary in dept_summaries.items():
        if len(dept_summary) > 1:  # Need at least 2 employees for comparison
            # Create radar chart for department employees
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            
            # Set up the radar chart
            categories = ['Handle up', 'Handle center', 'Handle down']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the polygon
            
            # Calculate department average for reference
            dept_avg_up = dept_summary['up_pct'].mean()
            dept_avg_center = dept_summary['center_pct'].mean()
            dept_avg_down = dept_summary['down_pct'].mean()
            dept_avg = [dept_avg_up, dept_avg_center, dept_avg_down]
            dept_avg += dept_avg[:1]  # Close the polygon
            
            # Plot department average first
            ax.plot(angles, dept_avg, 'k--', linewidth=2, label='Department Avg')
            
            # Plot each employee
            for _, row in dept_summary.iterrows():
                values = [row['up_pct'], row['center_pct'], row['down_pct']]
                values += values[:1]  # Close the polygon
                ax.plot(angles, values, linewidth=1, label=row['employee'])
                ax.fill(angles, values, alpha=0.1)
            
            # Set chart properties
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title(f'{dept_name} Department Employee Patterns', fontsize=16)
            
            # Set y limit based on max value
            max_val = max(dept_summary[['up_pct', 'center_pct', 'down_pct']].max()) * 1.2
            ax.set_ylim(0, max_val)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Save figure
            plt.tight_layout()
            plt.savefig(dept_vis_dir / f'{dept_name.lower()}_department_employees.png', dpi=300)
            plt.close()
            print(f"Saved {dept_name} department employee comparison to {dept_vis_dir}/{dept_name.lower()}_department_employees.png")
    
    print(f"All department visualizations saved to {dept_vis_dir}")


def main():
    # Use standard data path from project
    data_path = 'data/raw/processed_sensor_data.csv'
    
    # Try alternate paths if main path doesn't exist
    if not Path(data_path).exists():
        alternate_paths = ['data/processed_sensor_data.csv', 'processed_sensor_data.csv']
        for alt_path in alternate_paths:
            if Path(alt_path).exists():
                data_path = alt_path
                break
    
    print(f"Loading data from {data_path}")
    try:
        data = load_sensor_data(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {data_path}")
        print("Please run this script from the root directory of the project.")
        sys.exit(1)
    
    # Add department classification
    data = classify_departments(data)
    
    # Filter out pause data if column exists
    if 'isPauseData' in data.columns:
        data = data[data['isPauseData'] == False]
    
    # Step 1: Analyze handling time in top regions
    pivot_table, employee_top_regions, results_df = analyze_handling_time_in_top_regions(data)
    
    # Step 2: Analyze by workstation category
    category_df, region_categories = analyze_workstation_categories(data)
    
    # Step 3: Statistical analysis of handling positions
    comparison_df = analyze_handling_position_patterns(data, region_categories)
    
    # Step 4: Department-based analysis
    model_results, dept_summaries = run_regression_analysis(data, region_categories)
    
    # Step 5: Create visualizations
    create_visualizations(pivot_table, category_df, model_results, agg_data, region_categories)
    
    # Step 6: Create department-based visualizations
    create_department_visualizations(dept_summaries, region_categories)

if __name__ == "__main__":
    main()