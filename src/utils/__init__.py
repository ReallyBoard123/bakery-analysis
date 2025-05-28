"""
Utilities Package

Contains utility functions for file operations, data processing, time formatting, etc.
"""

# Import key modules for convenient access
from .file_utils import (
    check_file_exists,
    ensure_dir_exists,
    check_required_files
)

from .data_utils import (
    load_sensor_data,
    load_floor_plan_data,
    filter_data,
    classify_departments,
    aggregate_by_shift,
    aggregate_by_activity
)

from .time_utils import (
    format_seconds_to_hms,
    format_seconds_to_minutes,
    format_seconds_to_hours,
    format_duration
)

from .translation_utils import (
    get_translation
)

__all__ = [
    'cli_utils',
    'data_utils',
    'file_utils',
    'time_utils',
    'translation_utils'
]