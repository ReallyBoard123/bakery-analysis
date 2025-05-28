"""
Ergonomics Utilities Module

Contains helper functions for ergonomic analysis.
"""

from pathlib import Path
from src.utils.file_utils import ensure_dir_exists
from src.utils.time_utils import format_seconds_to_hms

def create_region_categories():
    """
    Group similar regions into functional clusters for handling analysis
    
    Returns:
    --------
    dict
        Dictionary mapping category names to lists of region names
    """
    return {
        'Bread Workstation': ['3_Brotstation', '4_Brotstation', '1_Brotstation', '2_Brotstation'],
        'Cake Workstation': ['1_Konditorei_station', '2_Konditorei_station', '3_Konditorei_station', 'konditorei_deco'],
        'Bread Baking': ['bereitstellen_prepared_goods', 'Etagenofen', 'Rollenhandtuchspend', 'Gärraum'],
        'Cake Baking': ['Fettbacken', 'cookies_packing_hand'],
        'Common Regions': ['brotkisten_regal', 'production_plan', 'leere_Kisten', '1_corridor']
    }

def create_region_clusters():
    """
    Group similar regions into functional clusters
    
    Returns:
    --------
    dict
        Dictionary mapping cluster names to lists of region names
    """
    return {
        'Bread_Stations': ['1_Brotstation', '2_Brotstation', '3_Brotstation', '4_Brotstation'],
        'Pastry_Stations': ['1_Konditorei_station', '2_Konditorei_station', '3_Konditorei_station', 
                           'konditorei_deco', 'konditorei_deco_1', 'konditorei_deco_2'],
        'Ovens': ['Etagenofen', '1_Stickenofen', '2_Stickenofen', 'Gärraum', 'Fettbacken', 
                 'g\u00e4rraum_handle', 'etagenofen'],
        'Storage': ['Lager', '1_lager', '2_lager', '3_lager', 'Silo', 'Lager_eingang', '2_lager_eingang'],
        'Hallways': ['flur_staircase_up', 'flur_eingang_umkleidung_WC', '1_corridor', '2_flur_aufzug', 
                    '1_flur_aufzug', '1_corridor_a', '1_corridor_b', 'flur_staircase']
    }

def export_duration_factor_thresholds(data, output_dir, language='en'):
    """
    Export the duration factor thresholds used in ergonomic scoring
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with employee tracking data
    output_dir : Path
        Directory to save the threshold information
    language : str, optional
        Language code ('en' or 'de')
    
    Returns:
    --------
    None
    """
    from .data_analysis import analyze_activity_durations
    from src.visualization.base import get_text
    
    activity_stats = analyze_activity_durations(data)
    output_path = Path(output_dir)
    
    with open(output_path / 'duration_factor_thresholds.txt', 'w') as f:
        f.write(get_text("Ergonomic Scoring System", language) + "\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(get_text("Deduction points per percentage of time spent in each posture:", language) + "\n")
        f.write(get_text("  Handle up: 0.4 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Handle down: 0.5 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Handle center: 0.05 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Stand: 0.15 points deducted per percentage point of time spent", language) + "\n")
        f.write(get_text("  Walk: 0.05 points deducted per percentage point of time spent", language) + "\n\n")
        
        f.write(get_text("Example: An employee spending 50% of time in Handle down position", language) + "\n")
        f.write(get_text("  would receive a deduction of 50 × 0.5 = 25 points", language) + "\n\n")
        
        f.write(get_text("Score categories:", language) + "\n")
        f.write(get_text("  Good: 75-100", language) + "\n")
        f.write(get_text("  Fair: 60-75", language) + "\n")
        f.write(get_text("  Poor: 0-60", language) + "\n\n")
        
        f.write(get_text("Duration statistics by activity:", language) + "\n")
        for activity, stats in activity_stats.items():
            f.write(f"{get_text(activity, language)}:\n")
            f.write(f"  {get_text('Count', language)}: {stats['count']}\n")
            f.write(f"  {get_text('Mean', language)}: {stats['mean']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('Median', language)}: {stats['median']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('25th percentile', language)}: {stats['p25']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('75th percentile', language)}: {stats['p75']:.1f} {get_text('seconds', language)}\n")
            f.write(f"  {get_text('90th percentile', language)}: {stats['p90']:.1f} {get_text('seconds', language)}\n\n")