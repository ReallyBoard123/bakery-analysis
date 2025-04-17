"""
Time Utilities Module

Provides standardized functions for time and duration formatting in the bakery analysis system.
"""

from datetime import datetime, timedelta
import pandas as pd

def format_seconds_to_hms(seconds):
    """
    Convert seconds to hh:mm:ss format
    
    Parameters:
    -----------
    seconds : float or int
        Duration in seconds
    
    Returns:
    --------
    str
        Formatted time string in hh:mm:ss format
    """
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00"
    
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def parse_time_to_seconds(time_str):
    """
    Convert hh:mm:ss formatted string to seconds
    
    Parameters:
    -----------
    time_str : str
        Time string in hh:mm:ss format
    
    Returns:
    --------
    float
        Duration in seconds
    """
    if not time_str or not isinstance(time_str, str):
        return 0
    
    try:
        # Try parsing as HH:MM:SS.fraction
        time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
        return total_seconds
    except ValueError:
        try:
            # Try parsing as HH:MM:SS
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
            total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            return total_seconds
        except ValueError:
            return 0

def format_seconds_to_minutes(seconds):
    """Convert seconds to minutes (float with 2 decimal places)"""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return 0.0
    return round(seconds / 60, 2)

def format_seconds_to_hours(seconds):
    """Convert seconds to hours (float with 2 decimal places)"""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return 0.0
    return round(seconds / 3600, 2)

def format_duration(seconds, include_seconds=True):
    """Format seconds as 5h 23m 12s or 5h 23m"""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "0h 0m"
    
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if include_seconds:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{hours}h {minutes}m"

def format_timedelta_series(td_series):
    """Format a pandas Series of timedelta values to hh:mm:ss strings"""
    def format_td(td):
        if pd.isna(td):
            return "00:00:00"
        
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    return td_series.apply(format_td)