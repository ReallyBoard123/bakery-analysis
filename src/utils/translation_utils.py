"""
Translation Utilities Module

Provides translations for text used in visualizations and reports.
"""

# Define translations dictionary
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
        'Date': 'Date',
        
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
        'Department': 'Department',
        
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
        'of total': 'of total',
        
        # New visualization titles
        'Total Activity Duration by Day': 'Total Activity Duration by Day',
        'Daily Activities - Employee {0}': 'Daily Activities - Employee {0}',
        'Daily Activities - All Employees': 'Daily Activities - All Employees',
        'Activity Breakdown - {0}: {1}': 'Activity Breakdown - {0}: {1}',
        'Activity Breakdown - All Data': 'Activity Breakdown - All Data',
        'Total Duration by Employee': 'Total Duration by Employee',
        'Activity Distribution by Department': 'Activity Distribution by Department',
        'Activity Distribution by Shift': 'Activity Distribution by Shift',
        'Activity Distribution Across Top Regions': 'Activity Distribution Across Top Regions',
        'Common Visualizations': 'Common Visualizations',
        
        # New visualization messages
        'Generating activity by day visualization...': 'Generating activity by day visualization...',
        'Generating employee summary visualization...': 'Generating employee summary visualization...',
        'Generating department comparison visualization...': 'Generating department comparison visualization...',
        'Generating shift comparison visualization...': 'Generating shift comparison visualization...',
        'Generating overall activity breakdown...': 'Generating overall activity breakdown...',
        'Generating visualizations for employee {0}...': 'Generating visualizations for employee {0}...',
        'Generating visualizations for {0} department...': 'Generating visualizations for {0} department...',
        'Generating region activity heatmap...': 'Generating region activity heatmap...',
        'Saved common visualizations to visualizations/common/': 'Saved common visualizations to visualizations/common/',
        'Common visualizations': 'Common visualizations',
        
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
        
        # Quality indicators
        'Poor': 'Poor',
        'Fair': 'Fair',
        'Good': 'Good',
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
        'Date': 'Datum',
        
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
        'Department': 'Abteilung',
        
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
        'of total': 'der Gesamtzeit',
        
        # New visualization titles
        'Total Activity Duration by Day': 'Gesamtaktivitätsdauer nach Tag',
        'Daily Activities - Employee {0}': 'Tägliche Aktivitäten - Mitarbeiter {0}',
        'Daily Activities - All Employees': 'Tägliche Aktivitäten - Alle Mitarbeiter',
        'Activity Breakdown - {0}: {1}': 'Aktivitätsaufschlüsselung - {0}: {1}',
        'Activity Breakdown - All Data': 'Aktivitätsaufschlüsselung - Alle Daten',
        'Total Duration by Employee': 'Gesamtdauer nach Mitarbeiter',
        'Activity Distribution by Department': 'Aktivitätsverteilung nach Abteilung',
        'Activity Distribution by Shift': 'Aktivitätsverteilung nach Schicht',
        'Activity Distribution Across Top Regions': 'Aktivitätsverteilung über die wichtigsten Regionen',
        'Common Visualizations': 'Allgemeine Visualisierungen',
        
        # New visualization messages
        'Generating activity by day visualization...': 'Generiere Aktivität nach Tag Visualisierung...',
        'Generating employee summary visualization...': 'Generiere Mitarbeiterzusammenfassung Visualisierung...',
        'Generating department comparison visualization...': 'Generiere Abteilungsvergleich Visualisierung...',
        'Generating shift comparison visualization...': 'Generiere Schichtvergleich Visualisierung...',
        'Generating overall activity breakdown...': 'Generiere Gesamtaktivitätsaufschlüsselung...',
        'Generating visualizations for employee {0}...': 'Generiere Visualisierungen für Mitarbeiter {0}...',
        'Generating visualizations for {0} department...': 'Generiere Visualisierungen für Abteilung {0}...',
        'Generating region activity heatmap...': 'Generiere Regionsaktivitäts-Heatmap...',
        'Saved common visualizations to visualizations/common/': 'Allgemeine Visualisierungen gespeichert in visualizations/common/',
        'Common visualizations': 'Allgemeine Visualisierungen',
        
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
        'Region Ergonomic Assessment': 'Ergonomische Bewertung des Bereichs',
        
        # Quality indicators
        'Poor': 'Schlecht',
        'Fair': 'Mäßig',
        'Good': 'Gut',
        
        # Other specialized terms
        'Score': 'Punktzahl',
        'Total Hours': 'Gesamtstunden',
        'Employees': 'Mitarbeiter',
        'Total': 'Gesamt',
        'Employee Handling Activities in Top Regions': 'Handhabungsaktivitäten der Mitarbeiter in Top-Bereichen',
        'Generated employee handling comparison for region': 'Mitarbeiter-Handhabungsvergleich erstellt für Bereich',
        'Generated combined handling comparison for all regions': 'Kombinierter Handhabungsvergleich für alle Bereiche erstellt',
        'Region Usage vs. Ergonomic Score': 'Bereichsnutzung vs. Ergonomie-Punktzahl',
        'Usage (hours)': 'Nutzung (Stunden)',
        'Percentage of Time (%)': 'Prozentsatz der Zeit (%)',
        'Employee Handling Activities': 'Handhabungsaktivitäten der Mitarbeiter',
        'Saved employee handling comparison by region to ergonomic_analysis/region_employee_comparisons/': 'Mitarbeiter-Handhabungsvergleich nach Bereich gespeichert in ergonomic_analysis/region_employee_comparisons/',
        'Generating employee handling comparison by region...': 'Erstelle Mitarbeiter-Handhabungsvergleich nach Bereich...',
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