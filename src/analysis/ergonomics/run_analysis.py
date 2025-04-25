"""
Ergonomics Run Analysis Module

Main entry point for running the ergonomic analysis workflow.
"""

from pathlib import Path

from src.utils.file_utils import ensure_dir_exists
from src.visualization.base import get_text

from .data_analysis import (
    analyze_handling_time_in_top_regions,
    analyze_workstation_categories,
    analyze_handling_position_patterns
)

from .visualizations import create_handling_position_visualizations
from .department_analysis import (
    compare_employees_within_department,
    compare_departments_in_specializations
)

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
    
    # Step 5: Employee comparison within departments
    print(f"\n=== {get_text('Step 5: Comparing Employees Within Departments', language)} ===")
    department_comparisons = compare_employees_within_department(data, output_dir, language)
    
    # Step 6: Department specialization comparison
    print(f"\n=== {get_text('Step 6: Comparing Departments in Specialized Environments', language)} ===")
    specialization_results = compare_departments_in_specializations(data, output_dir, language)
    
    print(f"\n=== {get_text('Handling Time Analysis Complete', language)} ===")
    
    return {
        'top_regions': {
            'pivot_table': pivot_table,
            'employee_top_regions': employee_top_regions
        },
        'workstation_categories': {
            'category_data': category_df,
            'region_categories': region_categories
        },
        'position_analysis': comparison_df,
        'department_comparisons': department_comparisons,
        'specialization_results': specialization_results
    }