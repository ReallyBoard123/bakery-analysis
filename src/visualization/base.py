"""
Base Visualization Module (Updated with New Activity Colors)

Provides base functionality and styling for all visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from ..utils.translation_utils import get_translation

# Updated standardized activity order per client requirements
ACTIVITY_ORDER = ['Walk', 'Stand', 'Handle center', 'Handle down', 'Handle up']

# Updated standardized colors for activities per client requirements
ACTIVITY_COLORS = {
    'Walk': '#156082',       # Blue (unchanged)
    'Stand': '#A6A6A6',      # Grey (unchanged)
    'Handle center': '#F1A983',  # Light orange (unchanged)
    'Handle down': '#C1341A',  # Updated red
    'Handle up': '#FEC134'   # Updated yellow
}

# Standardized colors for departments
DEPARTMENT_COLORS = {
    'Bread': '#8C6464',  # Brown
    'Cake': '#FFD700'    # Gold
}

# Standardized colors for employees
EMPLOYEE_COLORS = {
    '32-A': '#1f77b4',  # Blue
    '32-C': '#ff7f0e',  # Orange
    '32-D': '#2ca02c',  # Green
    '32-E': '#d62728',  # Red
    '32-F': '#9467bd',  # Purple
    '32-G': '#8c564b',  # Brown
    '32-H': '#e377c2'   # Pink
}

# Legacy color palette for backward compatibility
BAKERY_COLORS = ['#156082', '#A6A6A6', '#F1A983', '#C1341A', '#FEC134', '#8C6464']

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
    Get standardized color palette (legacy function)
    
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

def get_activity_colors():
    """Get standardized color mapping for activities"""
    return ACTIVITY_COLORS

def get_activity_order():
    """Get standardized order for activities"""
    return ACTIVITY_ORDER

def get_department_colors():
    """Get standardized color mapping for departments"""
    return DEPARTMENT_COLORS

def get_employee_colors():
    """Get standardized color mapping for employees"""
    return EMPLOYEE_COLORS

def create_activity_colormap(activity, reverse=False):
    """
    Create a custom colormap for a specific activity using its base color
    
    Parameters:
    -----------
    activity : str
        Activity name (e.g., 'Walk', 'Handle up', etc.)
    reverse : bool, optional
        Whether to reverse the colormap (darker to lighter vs lighter to darker)
    
    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap based on the activity's base color
    """
    base_color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
    
    # Create a range from white to the base color for the colormap
    if reverse:
        colors = [base_color, '#FFFFFF']
    else:
        colors = ['#FFFFFF', base_color]
    
    return LinearSegmentedColormap.from_list(f'{activity}_cmap', colors)

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
        ['#FFFFFF', '#E8EEF7', '#96BEE6', '#156082', '#0A324D']
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
    
def get_text(text, language='en'):
    """
    Get text with translation for specified language
    
    Parameters:
    -----------
    text : str
        Text to translate
    language : str
        Language code ('en' or 'de')
    
    Returns:
    --------
    str
        Translated text
    """
    return get_translation(text, language)

def add_duration_percentage_label(ax, bar, value_seconds, total_seconds, min_percentage=5, language='en'):
    """
    Add a formatted label showing both percentage and duration to a bar with translation

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
    language : str, optional
        Language code ('en' or 'de')

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