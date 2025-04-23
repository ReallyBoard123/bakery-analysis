"""
Translation Utilities Module

Provides translations for text used in visualizations and reports.
"""

# Define translations dictionary
TRANSLATIONS = {
    'en': {
        # Main titles
        'Time Distribution by Activity': 'Time Distribution by Activity',
        'Activity Distribution by Employee': 'Activity Distribution by Employee',
        'Activity Patterns by Hour of Day': 'Activity Patterns by Hour of Day',
        'Region Heatmap': 'Region Heatmap',
        'Region Transitions': 'Region Transitions',
        'Top Movement Transitions': 'Top Movement Transitions',
        'Activity Analysis by Region': 'Activity Analysis by Region',
        'Ergonomic Assessment Report': 'Ergonomic Assessment Report',
        
        # Common labels
        'Duration (hours)': 'Duration (hours)',
        'Hour of Day': 'Hour of Day',
        'Activity': 'Activity',
        'Employee ID': 'Employee ID',
        'Region': 'Region',
        'Transition Count': 'Transition Count',
        'Percentage': 'Percentage',
        'Hours': 'Hours',
        
        # Activity names
        'Walk': 'Walk',
        'Stand': 'Stand',
        'Handle up': 'Handle up',
        'Handle center': 'Handle center',
        'Handle down': 'Handle down',
        
        # Departments
        'Bread': 'Bread',
        'Cake': 'Cake',
        'Bread Department': 'Bread Department',
        'Cake Department': 'Cake Department',
        'All Departments': 'All Departments',
        
        # Other common phrases
        'Employee': 'Employee',
        'Shift': 'Shift',
        'Total Duration': 'Total Duration',
        'Hours Analyzed': 'Hours Analyzed',
        'Ergonomic Score': 'Ergonomic Score',
        'Transition Path': 'Transition Path',
        'Number of Transitions': 'Number of Transitions',
        'Transition Frequency': 'Transition Frequency',
        'Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions':
            'Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions',
        
        # Console outputs and errors
        'Error: Employee {0} not found in the data': 'Error: Employee {0} not found in the data',
        'Saved shift summary to data/shift_summary.csv': 'Saved shift summary to data/shift_summary.csv',
        'Saved activity summary to data/activity_summary.csv': 'Saved activity summary to data/activity_summary.csv',
        'Saved activity distribution by employee to visualizations/activity_distribution_by_employee.png': 
            'Saved activity distribution by employee to visualizations/activity_distribution_by_employee.png',
        'Saved region summary to data/region_summary.csv': 'Saved region summary to data/region_summary.csv',
        'Saved walking patterns analysis to statistics/walking_patterns_by_employee.csv': 
            'Saved walking patterns analysis to statistics/walking_patterns_by_employee.csv',
        'Saved handling patterns analysis to statistics/handling_patterns_by_employee.csv': 
            'Saved handling patterns analysis to statistics/handling_patterns_by_employee.csv',
        'Saved handling position summary to statistics/handling_position_summary.csv': 
            'Saved handling position summary to statistics/handling_position_summary.csv',
        'Saved employee handling positions to statistics/employee_handling_positions.csv': 
            'Saved employee handling positions to statistics/employee_handling_positions.csv',
        'Saved shift metrics to statistics/shift_metrics.csv': 'Saved shift metrics to statistics/shift_metrics.csv',
        'Analysis Complete': 'Analysis Complete',
        'Execution time: {0:.2f} seconds': 'Execution time: {0:.2f} seconds',
        'Results saved to: {0}': 'Results saved to: {0}',
        
        # Ergonomic report specific
        'Ergonomic Deductions by Activity': 'Ergonomic Deductions by Activity',
        'Activity Time Distribution': 'Activity Time Distribution',
        'Deduction Points': 'Deduction Points',
        'Average: {0:.1f} | Min: {1:.1f} | Max: {2:.1f}': 'Average: {0:.1f} | Min: {1:.1f} | Max: {2:.1f}',
        
        # Region report specific
        'Region Ergonomic Score': 'Region Ergonomic Score',
        'Activity Distribution in Region': 'Activity Distribution in Region',
        'Employee Contributions to Region': 'Employee Contributions to Region',
        'Contribution (%)': 'Contribution (%)',
        'Activity Breakdown by Employee in this Region': 'Activity Breakdown by Employee in this Region',
        'Activity Percentage (%)': 'Activity Percentage (%)',
        'Region Ergonomic Assessment': 'Region Ergonomic Assessment',
    },
    'de': {
        # Main titles
        'Time Distribution by Activity': 'Zeitverteilung nach Aktivität',
        'Activity Distribution by Employee': 'Aktivitätsverteilung nach Mitarbeiter',
        'Activity Patterns by Hour of Day': 'Aktivitätsmuster nach Tageszeit',
        'Region Heatmap': 'Bereichs-Heatmap',
        'Region Transitions': 'Bereichsübergänge',
        'Top Movement Transitions': 'Top Bewegungsübergänge',
        'Activity Analysis by Region': 'Aktivitätsanalyse nach Bereich',
        'Ergonomic Assessment Report': 'Ergonomischer Bewertungsbericht',
        
        # Common labels
        'Duration (hours)': 'Dauer (Stunden)',
        'Hour of Day': 'Tageszeit',
        'Activity': 'Aktivität',
        'Employee ID': 'Mitarbeiter-ID',
        'Region': 'Bereich',
        'Transition Count': 'Anzahl Übergänge',
        'Percentage': 'Prozentsatz',
        'Hours': 'Stunden',
        
        # Activity names
        'Walk': 'Gehen',
        'Stand': 'Stehen',
        'Handle up': 'Handhabung oben',
        'Handle center': 'Handhabung Mitte',
        'Handle down': 'Handhabung unten',
        
        # Departments
        'Bread': 'Brot',
        'Cake': 'Kuchen',
        'Bread Department': 'Brotabteilung',
        'Cake Department': 'Kuchenabteilung',
        'All Departments': 'Alle Abteilungen',
        
        # Other common phrases
        'Employee': 'Mitarbeiter',
        'Shift': 'Schicht',
        'Total Duration': 'Gesamtdauer',
        'Hours Analyzed': 'Analysierte Stunden',
        'Ergonomic Score': 'Ergonomie-Punktzahl',
        'Transition Path': 'Übergangspfad',
        'Number of Transitions': 'Anzahl der Übergänge',
        'Transition Frequency': 'Übergangshäufigkeit',
        'Solid lines: Adjacent regions | Dotted lines: Non-adjacent regions':
            'Durchgezogene Linien: Angrenzende Bereiche | Gestrichelte Linien: Nicht angrenzende Bereiche',
        
        # Console outputs and errors
        'Error: Employee {0} not found in the data': 'Fehler: Mitarbeiter {0} nicht in Daten gefunden',
        'Saved shift summary to data/shift_summary.csv': 'Schichtzusammenfassung gespeichert in data/shift_summary.csv',
        'Saved activity summary to data/activity_summary.csv': 'Aktivitätszusammenfassung gespeichert in data/activity_summary.csv',
        'Saved activity distribution by employee to visualizations/activity_distribution_by_employee.png': 
            'Aktivitätsverteilung nach Mitarbeiter gespeichert in visualizations/activity_distribution_by_employee.png',
        'Saved region summary to data/region_summary.csv': 'Bereichszusammenfassung gespeichert in data/region_summary.csv',
        'Saved walking patterns analysis to statistics/walking_patterns_by_employee.csv': 
            'Analyse der Laufmuster gespeichert in statistics/walking_patterns_by_employee.csv',
        'Saved handling patterns analysis to statistics/handling_patterns_by_employee.csv': 
            'Analyse der Handhabungsmuster gespeichert in statistics/handling_patterns_by_employee.csv',
        'Saved handling position summary to statistics/handling_position_summary.csv': 
            'Zusammenfassung der Handhabungspositionen gespeichert in statistics/handling_position_summary.csv',
        'Saved employee handling positions to statistics/employee_handling_positions.csv': 
            'Handhabungspositionen der Mitarbeiter gespeichert in statistics/employee_handling_positions.csv',
        'Saved shift metrics to statistics/shift_metrics.csv': 'Schichtmetriken gespeichert in statistics/shift_metrics.csv',
        'Analysis Complete': 'Analyse abgeschlossen',
        'Execution time: {0:.2f} seconds': 'Ausführungszeit: {0:.2f} Sekunden',
        'Results saved to: {0}': 'Ergebnisse gespeichert in: {0}',
        
        # Ergonomic report specific
        'Ergonomic Deductions by Activity': 'Ergonomische Abzüge nach Aktivität',
        'Activity Time Distribution': 'Aktivitätszeitverteilung',
        'Deduction Points': 'Abzugspunkte',
        'Average: {0:.1f} | Min: {1:.1f} | Max: {2:.1f}': 'Durchschnitt: {0:.1f} | Min: {1:.1f} | Max: {2:.1f}',
        'Ergonomic Score: {0}/100': 'Ergonomie-Punktzahl: {0}/100',
        'Employee: {0} | Hours Analyzed: {1:.1f}': 'Mitarbeiter: {0} | Analysierte Stunden: {1:.1f}',
        'Duration (hours)': 'Dauer (Stunden)',
        'Employee {0}': 'Mitarbeiter {0}',
        'Activity Time Distribution': 'Aktivitätszeitverteilung',
        'Ergonomic Assessment Report - Employee {0}': 'Ergonomischer Bewertungsbericht - Mitarbeiter {0}',
        'Percentage': 'Prozentsatz',
        'Duration': 'Dauer',
        'Deduction': 'Abzug',
        'pts': 'Pkt.',
        'of total': 'der Gesamtzeit',
        'No transitions with at least {0} occurrences for {1}': 'Keine Übergänge mit mindestens {0} Vorkommen für {1}',
        'Region: {0} | Hours Analyzed: {1:.1f} | Employees: {2}': 'Region: {0} | Analysierte Stunden: {1:.1f} | Mitarbeiter: {2}',
        
        # Region report specific
        'Region Ergonomic Score': 'Ergonomie-Punktzahl für Bereich',
        'Activity Distribution in Region': 'Aktivitätsverteilung im Bereich',
        'Employee Contributions to Region': 'Mitarbeiterbeiträge zum Bereich',
        'Contribution (%)': 'Beitrag (%)',
        'Activity Breakdown by Employee in this Region': 'Aktivitätsaufschlüsselung nach Mitarbeiter in diesem Bereich',
        'Activity Percentage (%)': 'Aktivitätsprozentsatz (%)',
        'Region Ergonomic Assessment': 'Ergonomische Bewertung des Bereichs'
    }
}

def get_translation(text, language='en'):
    """
    Get translation for text in specified language
    
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
    if language not in TRANSLATIONS:
        return text
    
    return TRANSLATIONS[language].get(text, text)

def translate_format_string(format_string, values, language='en'):
    """
    Translate a format string and format it with the provided values
    
    Parameters:
    -----------
    format_string : str
        Format string to translate
    values : tuple
        Values to format the string with
    language : str
        Language code ('en' or 'de')
    
    Returns:
    --------
    str
        Translated and formatted string
    """
    translated_format = get_translation(format_string, language)
    return translated_format.format(*values)