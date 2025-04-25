"""
Analysis Package

Contains modules for statistical analysis and ergonomics analysis of employee data.
"""

# Import key modules for convenient access
from .statistical import (
    analyze_employee_differences,
    analyze_ergonomic_patterns,
    compare_shifts
)

# Import ergonomics module
from . import ergonomics

__all__ = [
    'statistical',
    'ergonomics'
]