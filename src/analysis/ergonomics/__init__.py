"""
Ergonomics Analysis Package

Analyzes ergonomic patterns and generates employee and region ergonomic scores with visualizations.
"""

# Import main functions for direct access from ergonomics package
from .data_analysis import (
    analyze_activity_durations,
    analyze_handling_time_in_top_regions,
    analyze_workstation_categories,
    analyze_handling_position_patterns,
    analyze_region_ergonomics,
    calculate_ergonomic_score
)

from .utilities import (
    create_region_categories,
    create_region_clusters,
    export_duration_factor_thresholds
)

from .visualizations import (
    create_handling_position_visualizations
)

from .reports import (
    generate_ergonomic_report,
    generate_region_ergonomic_report,
    generate_all_employees_comparison,
    generate_all_regions_comparison,
    generate_circular_packing_visualization,
    generate_employee_handling_comparison_by_region
)

from .department_analysis import (
    compare_employees_within_department,
    compare_departments_in_specializations
)

# Main entry point
from .run_analysis import run_handling_time_analysis

# For backward compatibility with code using the previous flat structure
__all__ = [
    # Data Analysis
    'analyze_activity_durations',
    'analyze_handling_time_in_top_regions',
    'analyze_workstation_categories',
    'analyze_handling_position_patterns',
    'analyze_region_ergonomics',
    'calculate_ergonomic_score',
    
    # Utilities
    'create_region_categories',
    'create_region_clusters',
    'export_duration_factor_thresholds',
    
    # Visualizations
    'create_handling_position_visualizations',
    
    # Reports
    'generate_ergonomic_report',
    'generate_region_ergonomic_report',
    'generate_all_employees_comparison',
    'generate_all_regions_comparison',
    'generate_circular_packing_visualization',
    'generate_employee_handling_comparison_by_region',
    
    # Department Analysis
    'compare_employees_within_department',
    'compare_departments_in_specializations',
    
    # Main
    'run_handling_time_analysis'
]