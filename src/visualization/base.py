"""
Base Visualization Module

Provides base functionality and styling for all visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Define standard color palette for bakery visualizations
BAKERY_COLORS = ['#2D5F91', '#FFAA00', '#E8655F', '#48B7B4', '#96BEE6', '#8C6464']

def set_visualization_style():
    """
    Set consistent styling for all visualizations
    
    Returns:
    --------
    None
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Set default font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Set default grid style
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

def get_color_palette(n_colors=None):
    """
    Get standardized color palette
    
    Parameters:
    -----------
    n_colors : int, optional
        Number of colors to return
    
    Returns:
    --------
    list
        List of color hex codes
    """
    if n_colors is None or n_colors <= len(BAKERY_COLORS):
        return BAKERY_COLORS[:n_colors]
    else:
        # If more colors needed, cycle through the palette
        return [BAKERY_COLORS[i % len(BAKERY_COLORS)] for i in range(n_colors)]

def get_bakery_cmap():
    """
    Get custom colormap for the bakery theme
    
    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap
    """
    return LinearSegmentedColormap.from_list(
        'bakery', 
        ['#FFFFFF', '#E8EEF7', '#96BEE6', '#2D5F91', '#1A365D']
    )

def save_figure(fig, output_path, dpi=300, bbox_inches='tight'):
    """
    Save figure with standard parameters
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_path : str or Path
        Path to save figure
    dpi : int, optional
        Resolution in dots per inch
    bbox_inches : str, optional
        Bounding box in inches
    
    Returns:
    --------
    None
    """
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Save the figure
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure to {output_path}")

def add_duration_percentage_label(ax, bar, value_seconds, total_seconds, min_percentage=5):
    """
    Add a formatted label showing both percentage and duration to a bar

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the label to
    bar : matplotlib.patches.Rectangle
        The bar to add the label to
    value_seconds : float
        The duration in seconds
    total_seconds : float
        The total duration in seconds (for percentage calculation)
    min_percentage : float, optional
        Minimum percentage to display label

    Returns:
    --------
    None
    """
    from ..utils.time_utils import format_seconds_to_hms

    height = bar.get_height()
    percentage = (value_seconds / total_seconds * 100).round(1) if total_seconds > 0 else 0

    # Format duration as hh:mm:ss
    formatted_duration = format_seconds_to_hms(value_seconds)

    if percentage >= min_percentage:  # Only show labels for significant percentages
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                f'{percentage:.1f}%\n{formatted_duration}',
                ha='center', va='bottom', fontsize=8)