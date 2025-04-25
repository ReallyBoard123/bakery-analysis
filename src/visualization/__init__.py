"""
Visualization Package

Contains modules for creating data visualizations of employee movement and activities.
"""

# Import key modules for convenient access
from .base import (
    set_visualization_style,
    save_figure,
    get_activity_colors,
    get_activity_order,
    get_text
)

from .activity_vis import (
    plot_activity_distribution,
    plot_activity_distribution_by_employee,
    plot_hourly_activity_patterns,
    analyze_activities_by_region
)

from .employee_region_vis import (
    create_employee_region_heatmap
)

from .floor_plan_vis import (
    create_region_heatmap,
    create_movement_flow
)

__all__ = [
    'activity_vis',
    'base',
    'employee_region_vis',
    'floor_plan_vis'
]