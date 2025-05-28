"""
Command Line Interface Utilities

Provides standardized functions for handling command-line arguments.
"""

import argparse
from pathlib import Path
from .file_utils import check_file_exists

def get_argument_parser():
    """
    Create standardized argument parser for bakery analysis
    
    Returns:
    --------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Analyze bakery employee tracking data')
    
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                      help='Path to the data file')
    parser.add_argument('--layout', type=str, default='data/assets/layout.png',
                      help='Path to the floor plan image')
    parser.add_argument('--metadata', type=str, default='data/assets/process_metadata.json',
                      help='Path to the process metadata file')
    parser.add_argument('--connections', type=str, default='data/assets/floor-plan-connections-2025-04-09.json',
                      help='Path to the floor plan connections file')
    parser.add_argument('--output', type=str, default='output',
                      help='Directory to save output files')
    
    return parser

def validate_args(args):
    """
    Validate command-line arguments
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed arguments
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    required_files = {
        'Data file': args.data,
        'Floor plan image': args.layout,
        'Metadata file': args.metadata,
        'Connections file': args.connections
    }
    
    all_valid = True
    
    for name, path in required_files.items():
        if not check_file_exists(path):
            print(f"Error: {name} not found: {path}")
            all_valid = False
    
    return all_valid

def get_config_from_args(args):
    """
    Convert command-line args to a configuration dictionary
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed arguments
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    config = {
        'paths': {
            'data': args.data,
            'layout': args.layout,
            'metadata': args.metadata,
            'connections': args.connections,
            'output': args.output
        }
    }
    
    return config