"""
File Utilities Module

Provides standardized functions for file and directory operations.
"""

import os
from pathlib import Path

def check_file_exists(file_path):
    """
    Check if file exists
    
    Parameters:
    -----------
    file_path : str or Path
        Path to check
    
    Returns:
    --------
    bool
        True if file exists, False otherwise
    """
    return Path(file_path).exists()

def ensure_dir_exists(dir_path):
    """
    Create directory if it doesn't exist
    
    Parameters:
    -----------
    dir_path : str or Path
        Directory path to create
    
    Returns:
    --------
    Path
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(exist_ok=True, parents=True)
    return path

def get_output_path(base_dir, sub_dir=None, filename=None):
    """
    Generate standardized output path
    
    Parameters:
    -----------
    base_dir : str or Path
        Base output directory
    sub_dir : str, optional
        Subdirectory within base_dir
    filename : str, optional
        Filename to append to the path
    
    Returns:
    --------
    Path
        Complete output path
    """
    path = Path(base_dir)
    
    if sub_dir:
        path = path / sub_dir
        path.mkdir(exist_ok=True, parents=True)
    
    if filename:
        path = path / filename
    
    return path

def check_required_files(required_files):
    """
    Check if all required files exist
    
    Parameters:
    -----------
    required_files : dict
        Dictionary with file descriptions as keys and paths as values
    
    Returns:
    --------
    tuple
        (bool, list) - Success flag and list of missing files
    """
    missing_files = []
    
    for description, path in required_files.items():
        if not check_file_exists(path):
            missing_files.append(f"{description}: {path}")
    
    return len(missing_files) == 0, missing_files