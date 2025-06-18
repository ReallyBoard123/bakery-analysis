#!/usr/bin/env python3
"""
Simple Region Activity Analyzer
===============================

Analyzes activity frequency within a specific region by employee and shift.

For each employee/shift combination:
1. Finds meaningful visits to the region (duration > threshold)
2. Counts target activity occurrences within each visit
3. Calculates frequency and patterns

Usage:
    python simple_region_activity_analyzer.py --region "7e_Brotwagon_und_kisten" --activity "Handle down" --threshold 10
    python simple_region_activity_analyzer.py --region "3_Konditorei_deco_raum" --activity "Handle center" --threshold 15
    python simple_region_activity_analyzer.py --region "7e_Brotwagon_und_kisten" --activity "Handle down" --temporal-detail --comprehensive-viz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import sys

# Import utilities from the existing codebase
try:
    from src.utils.time_utils import format_seconds_to_hms
    from src.utils.data_utils import load_sensor_data, classify_departments
    from src.visualization.base import save_figure, set_visualization_style, get_employee_colors
except ImportError:
    print("Warning: Could not import from src modules. Some features may be limited.")
    
    def format_seconds_to_hms(seconds):
        """Fallback time formatting function"""
        if seconds is None or seconds < 0:
            return "00:00:00"
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class SimpleRegionActivityAnalyzer:
    """Simple analyzer for activity frequency within a specific region"""
    
    def __init__(self, data, region_of_interest, target_activity, visit_threshold_seconds=10):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Employee tracking data
        region_of_interest : str
            The region to analyze
        target_activity : str
            The specific activity to count (e.g., "Handle down")
        visit_threshold_seconds : int
            Minimum duration to count as a meaningful visit
        """
        self.data = data.copy()
        self.region_of_interest = region_of_interest
        self.target_activity = target_activity
        self.visit_threshold = visit_threshold_seconds
        self.results = None
        
        # Prepare the data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and sort data for analysis"""
        # Ensure data is sorted by employee and time
        self.data = self.data.sort_values(['id', 'startTime']).reset_index(drop=True)
        
        # Filter for the region of interest
        self.region_data = self.data[self.data['region'] == self.region_of_interest].copy()
        
        print(f"📍 Region: {self.region_of_interest}")
        print(f"🎯 Target Activity: {self.target_activity}")
        print(f"⏱️  Visit Threshold: {self.visit_threshold} seconds")
        print(f"📊 Total records in region: {len(self.region_data):,}")
        
        # Get unique employees and shifts
        self.employees = sorted(self.region_data['id'].unique())
        self.shifts = sorted(self.region_data['shift'].unique()) if 'shift' in self.region_data.columns else [1]
        
        print(f"👥 Employees: {', '.join(self.employees)}")
        print(f"🔄 Shifts: {', '.join(map(str, self.shifts))}")
        
    def analyze_region_activity(self):
        """
        Perform the main analysis
        
        Returns:
        --------
        dict
            Analysis results
        """
        print(f"\n🔍 Analyzing activity patterns...")
        
        all_visits = []
        employee_shift_summary = defaultdict(lambda: {
            'visit_count': 0,
            'total_visit_duration': 0,
            'target_activity_occurrences': 0,
            'total_target_duration': 0,
            'visits': []
        })
        
        # Process each employee
        for employee_id in self.employees:
            emp_data = self.region_data[self.region_data['id'] == employee_id]
            
            # Process each shift for this employee
            for shift in self.shifts:
                shift_data = emp_data[emp_data['shift'] == shift] if 'shift' in emp_data.columns else emp_data
                
                if shift_data.empty:
                    continue
                
                # Find visits for this employee/shift combination
                visits = self._find_meaningful_visits(shift_data, employee_id, shift)
                
                # Update summary
                key = (employee_id, shift)
                summary = employee_shift_summary[key]
                summary['visit_count'] = len(visits)
                summary['visits'] = visits
                
                if visits:
                    summary['total_visit_duration'] = sum(v['total_duration'] for v in visits)
                    summary['target_activity_occurrences'] = sum(v['target_activity_count'] for v in visits)
                    summary['total_target_duration'] = sum(v['target_activity_duration'] for v in visits)
                    summary['avg_target_per_visit'] = summary['target_activity_occurrences'] / len(visits)
                    summary['avg_visit_duration'] = summary['total_visit_duration'] / len(visits)
                
                all_visits.extend(visits)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(all_visits, employee_shift_summary)
        
        self.results = {
            'all_visits': all_visits,
            'employee_shift_summary': dict(employee_shift_summary),
            'overall_stats': overall_stats,
            'region_of_interest': self.region_of_interest,
            'target_activity': self.target_activity,
            'visit_threshold': self.visit_threshold
        }
        
        return self.results
    
    def _find_meaningful_visits(self, shift_data, employee_id, shift):
        """
        Find meaningful visits (consecutive records with total duration > threshold)
        
        Parameters:
        -----------
        shift_data : pandas.DataFrame
            Data for one employee/shift combination
        employee_id : str
            Employee ID
        shift : int
            Shift number
            
        Returns:
        --------
        list
            List of visit dictionaries
        """
        if shift_data.empty:
            return []
        
        visits = []
        shift_data = shift_data.sort_values('startTime').reset_index(drop=True)
        
        i = 0
        while i < len(shift_data):
            # Start a new visit
            visit_records = [shift_data.iloc[i]]
            visit_start_time = shift_data.iloc[i]['startTime']
            
            # Look for consecutive records (allowing small gaps)
            j = i + 1
            while j < len(shift_data):
                current_record = shift_data.iloc[j]
                prev_record = shift_data.iloc[j-1]
                
                # Check if this record continues the visit (small time gap = same visit)
                time_gap = current_record['startTime'] - prev_record['endTime']
                
                if time_gap <= 60:  # Less than 1 minute gap = same visit
                    visit_records.append(current_record)
                    j += 1
                else:
                    break
            
            # Calculate visit statistics
            visit_end_time = visit_records[-1]['endTime']
            total_duration = sum(record['duration'] for record in visit_records)
            
            # Only include if duration exceeds threshold
            if total_duration >= self.visit_threshold:
                # Count target activity occurrences and duration
                target_records = [r for r in visit_records if r['activity'] == self.target_activity]
                target_activity_count = len(target_records)
                target_activity_duration = sum(r['duration'] for r in target_records)
                
                # Get all activities in this visit
                activities = Counter([r['activity'] for r in visit_records])
                
                visit = {
                    'employee_id': employee_id,
                    'shift': shift,
                    'date': visit_records[0]['date'] if 'date' in visit_records[0] else 'unknown',
                    'start_time': visit_start_time,
                    'end_time': visit_end_time,
                    'total_duration': total_duration,
                    'record_count': len(visit_records),
                    'target_activity_count': target_activity_count,
                    'target_activity_duration': target_activity_duration,
                    'target_activity_percentage': (target_activity_duration / total_duration * 100) if total_duration > 0 else 0,
                    'all_activities': dict(activities),
                    'records': visit_records
                }
                
                visits.append(visit)
            
            # Move to the next unprocessed record
            i = j if j > i + 1 else i + 1
        
        return visits
    
    def _calculate_overall_stats(self, all_visits, employee_shift_summary):
        """Calculate overall statistics across all visits"""
        if not all_visits:
            return {
                'total_visits': 0,
                'total_employees': len(self.employees),
                'total_shifts': len(self.shifts),
                'avg_target_per_visit': 0,
                'total_target_occurrences': 0
            }
        
        total_target_occurrences = sum(visit['target_activity_count'] for visit in all_visits)
        total_visits = len(all_visits)
        active_combinations = len([k for k, v in employee_shift_summary.items() if v['visit_count'] > 0])
        
        # Calculate employee rankings
        employee_totals = defaultdict(lambda: {'visits': 0, 'target_count': 0, 'target_duration': 0})
        for (emp_id, shift), summary in employee_shift_summary.items():
            employee_totals[emp_id]['visits'] += summary['visit_count']
            employee_totals[emp_id]['target_count'] += summary['target_activity_occurrences']
            employee_totals[emp_id]['target_duration'] += summary['total_target_duration']
        
        # Sort employees by target activity count
        top_employees = sorted(employee_totals.items(), 
                             key=lambda x: x[1]['target_count'], reverse=True)[:5]
        
        return {
            'total_visits': total_visits,
            'total_employees': len(self.employees),
            'total_shifts': len(self.shifts),
            'active_employee_shift_combinations': active_combinations,
            'avg_target_per_visit': total_target_occurrences / total_visits if total_visits > 0 else 0,
            'total_target_occurrences': total_target_occurrences,
            'total_target_duration': sum(visit['target_activity_duration'] for visit in all_visits),
            'avg_visit_duration': np.mean([visit['total_duration'] for visit in all_visits]),
            'median_visit_duration': np.median([visit['total_duration'] for visit in all_visits]),
            'top_employees': top_employees
        }
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis"""
        if not self.results:
            print("❌ No analysis results available. Run analyze_region_activity() first.")
            return
        
        stats = self.results['overall_stats']
        summary = self.results['employee_shift_summary']
        
        print(f"\n{'='*60}")
        print(f"📊 REGION ACTIVITY ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"📍 Region: {self.region_of_interest}")
        print(f"🎯 Target Activity: {self.target_activity}")
        print(f"⏱️  Visit Threshold: {self.visit_threshold} seconds")
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   • Total meaningful visits: {stats['total_visits']}")
        print(f"   • Active employee/shift combinations: {stats['active_employee_shift_combinations']}")
        print(f"   • Total {self.target_activity} occurrences: {stats['total_target_occurrences']}")
        print(f"   • Average {self.target_activity} per visit: {stats['avg_target_per_visit']:.1f}")
        print(f"   • Total {self.target_activity} duration: {format_seconds_to_hms(stats['total_target_duration'])}")
        print(f"   • Average visit duration: {format_seconds_to_hms(stats['avg_visit_duration'])}")
        
        print(f"\n🏆 TOP EMPLOYEES BY {self.target_activity.upper()} COUNT:")
        for i, (emp_id, emp_stats) in enumerate(stats['top_employees'], 1):
            print(f"   {i}. {emp_id}: {emp_stats['target_count']} occurrences "
                  f"({emp_stats['visits']} visits, {format_seconds_to_hms(emp_stats['target_duration'])})")
        
        print(f"\n👥 DETAILED EMPLOYEE/SHIFT BREAKDOWN:")
        
        # Sort by employee ID for consistent display
        sorted_summary = sorted(summary.items(), key=lambda x: (x[0][0], x[0][1]))
        
        for (emp_id, shift), emp_stats in sorted_summary:
            if emp_stats['visit_count'] > 0:
                avg_target = emp_stats.get('avg_target_per_visit', 0)
                avg_duration = emp_stats.get('avg_visit_duration', 0)
                
                print(f"   • {emp_id}, Shift {shift}: {emp_stats['visit_count']} visits, "
                      f"{emp_stats['target_activity_occurrences']} {self.target_activity} "
                      f"(avg {avg_target:.1f} per visit, avg visit: {format_seconds_to_hms(avg_duration)})")
    
    def get_detailed_visit_data(self):
        """
        Get detailed visit data for export or further analysis
        
        Returns:
        --------
        pandas.DataFrame
            Detailed visit data
        """
        if not self.results:
            print("❌ No analysis results available. Run analyze_region_activity() first.")
            return None
        
        visit_data = []
        for visit in self.results['all_visits']:
            visit_data.append({
                'employee_id': visit['employee_id'],
                'shift': visit['shift'],
                'date': visit['date'],
                'start_time': visit['start_time'],
                'end_time': visit['end_time'],
                'total_duration_seconds': visit['total_duration'],
                'total_duration_formatted': format_seconds_to_hms(visit['total_duration']),
                'record_count': visit['record_count'],
                'target_activity_count': visit['target_activity_count'],
                'target_activity_duration_seconds': visit['target_activity_duration'],
                'target_activity_duration_formatted': format_seconds_to_hms(visit['target_activity_duration']),
                'target_activity_percentage': round(visit['target_activity_percentage'], 1),
                'all_activities': str(visit['all_activities'])
            })
        
        return pd.DataFrame(visit_data)
    
    def get_employee_summary_data(self):
        """
        Get employee summary data for export
        
        Returns:
        --------
        pandas.DataFrame
            Employee summary data
        """
        if not self.results:
            print("❌ No analysis results available. Run analyze_region_activity() first.")
            return None
        
        summary_data = []
        for (emp_id, shift), stats in self.results['employee_shift_summary'].items():
            if stats['visit_count'] > 0:
                summary_data.append({
                    'employee_id': emp_id,
                    'shift': shift,
                    'visit_count': stats['visit_count'],
                    'total_visit_duration_seconds': stats['total_visit_duration'],
                    'total_visit_duration_formatted': format_seconds_to_hms(stats['total_visit_duration']),
                    'target_activity_occurrences': stats['target_activity_occurrences'],
                    'total_target_duration_seconds': stats['total_target_duration'],
                    'total_target_duration_formatted': format_seconds_to_hms(stats['total_target_duration']),
                    'avg_target_per_visit': round(stats.get('avg_target_per_visit', 0), 1),
                    'avg_visit_duration_seconds': round(stats.get('avg_visit_duration', 0), 1),
                    'avg_visit_duration_formatted': format_seconds_to_hms(stats.get('avg_visit_duration', 0))
                })
        
        return pd.DataFrame(summary_data)
    
    def create_visualizations(self, output_dir=None):
        """Create visualizations of the analysis"""
        if not self.results:
            print("❌ No analysis results available. Run analyze_region_activity() first.")
            return
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            set_visualization_style()
        except:
            plt.style.use('default')
        
        # 1. Employee activity frequency
        self._create_employee_frequency_chart(output_dir)
        
        # 2. Visit patterns by shift
        self._create_shift_patterns_chart(output_dir)
        
        # 3. Activity count distribution
        self._create_activity_distribution_chart(output_dir)
    
    def _create_employee_frequency_chart(self, output_dir):
        """Create employee activity frequency chart"""
        summary = self.results['employee_shift_summary']
        
        # Aggregate by employee (sum across shifts)
        employee_totals = defaultdict(lambda: {'visits': 0, 'target_count': 0})
        for (emp_id, shift), stats in summary.items():
            employee_totals[emp_id]['visits'] += stats['visit_count']
            employee_totals[emp_id]['target_count'] += stats['target_activity_occurrences']
        
        # Filter employees with visits
        active_employees = {k: v for k, v in employee_totals.items() if v['visits'] > 0}
        
        if not active_employees:
            print("⚠️  No active employees to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        employees = list(active_employees.keys())
        visit_counts = [active_employees[emp]['visits'] for emp in employees]
        target_counts = [active_employees[emp]['target_count'] for emp in employees]
        
        try:
            employee_colors = get_employee_colors()
            colors = [employee_colors.get(emp, '#cccccc') for emp in employees]
        except:
            colors = plt.cm.Set3(np.linspace(0, 1, len(employees)))
        
        # Chart 1: Visit counts
        bars1 = ax1.bar(employees, visit_counts, color=colors)
        ax1.set_title('Number of Visits by Employee', fontweight='bold')
        ax1.set_xlabel('Employee ID')
        ax1.set_ylabel('Visit Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars1, visit_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Target activity counts
        bars2 = ax2.bar(employees, target_counts, color=colors)
        ax2.set_title(f'{self.target_activity} Count by Employee', fontweight='bold')
        ax2.set_xlabel('Employee ID')
        ax2.set_ylabel('Activity Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars2, target_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_{self.target_activity.replace(" ", "_")}_employee_frequency.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_shift_patterns_chart(self, output_dir):
        """Create shift pattern analysis chart"""
        summary = self.results['employee_shift_summary']
        
        # Organize data by shift
        shift_data = defaultdict(lambda: {'employees': [], 'visits': [], 'target_counts': []})
        
        for (emp_id, shift), stats in summary.items():
            if stats['visit_count'] > 0:
                shift_data[shift]['employees'].append(emp_id)
                shift_data[shift]['visits'].append(stats['visit_count'])
                shift_data[shift]['target_counts'].append(stats['target_activity_occurrences'])
        
        if not shift_data:
            print("⚠️  No shift data to visualize")
            return
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        shifts = sorted(shift_data.keys())
        width = 0.35
        
        # Get all unique employees for consistent coloring
        all_employees = sorted(set(emp for (emp, _) in summary.keys()))
        try:
            employee_colors = get_employee_colors()
            color_map = {emp: employee_colors.get(emp, '#cccccc') for emp in all_employees}
        except:
            color_map = {emp: plt.cm.Set3(i/len(all_employees)) for i, emp in enumerate(all_employees)}
        
        # Create stacked bars for each shift
        bottom_visits = [0] * len(shifts)
        bottom_targets = [0] * len(shifts)
        
        legend_handles = []
        legend_labels = []
        
        for emp in all_employees:
            emp_visits = []
            emp_targets = []
            
            for shift in shifts:
                # Find this employee's data for this shift
                emp_data = summary.get((emp, shift), {'visit_count': 0, 'target_activity_occurrences': 0})
                emp_visits.append(emp_data['visit_count'])
                emp_targets.append(emp_data['target_activity_occurrences'])
            
            # Only plot if employee has any activity
            if sum(emp_visits) > 0 or sum(emp_targets) > 0:
                x_pos = np.arange(len(shifts))
                
                # Plot visits
                bars1 = ax.bar(x_pos - width/2, emp_visits, width, bottom=bottom_visits, 
                              color=color_map[emp], alpha=0.7, label=f'{emp} (visits)')
                
                # Plot targets
                bars2 = ax.bar(x_pos + width/2, emp_targets, width, bottom=bottom_targets, 
                              color=color_map[emp], alpha=1.0, label=f'{emp} ({self.target_activity})')
                
                # Update bottoms for stacking
                for i in range(len(shifts)):
                    bottom_visits[i] += emp_visits[i]
                    bottom_targets[i] += emp_targets[i]
                
                # Add to legend (only once per employee)
                if emp_visits != [0] * len(shifts) or emp_targets != [0] * len(shifts):
                    legend_handles.append(bars2[0])  # Use the target activity bar for legend
                    legend_labels.append(emp)
        
        ax.set_xlabel('Shift')
        ax.set_ylabel('Count')
        ax.set_title(f'Visit and {self.target_activity} Patterns by Shift', fontweight='bold')
        ax.set_xticks(np.arange(len(shifts)))
        ax.set_xticklabels([f'Shift {s}' for s in shifts])
        
        # Add legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, title='Employee', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add labels to distinguish visit vs target bars
        ax.text(-0.4, max(bottom_visits) * 0.9, 'Visits', rotation=90, ha='center', va='center', 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(len(shifts)-0.6, max(bottom_targets) * 0.9, self.target_activity, rotation=90, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_shift_patterns.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_activity_distribution_chart(self, output_dir):
        """Create activity count distribution chart"""
        all_visits = self.results['all_visits']
        
        if not all_visits:
            print("⚠️  No visits to analyze")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Chart 1: Distribution of target activity counts per visit
        target_counts = [visit['target_activity_count'] for visit in all_visits]
        
        ax1.hist(target_counts, bins=max(1, max(target_counts) + 1), alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'Distribution of {self.target_activity} Count per Visit', fontweight='bold')
        ax1.set_xlabel(f'{self.target_activity} Count')
        ax1.set_ylabel('Number of Visits')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Visit duration vs target activity count scatter
        durations = [visit['total_duration']/60 for visit in all_visits]  # Convert to minutes
        
        scatter = ax2.scatter(target_counts, durations, alpha=0.6, c=target_counts, cmap='viridis')
        ax2.set_title(f'Visit Duration vs {self.target_activity} Count', fontweight='bold')
        ax2.set_xlabel(f'{self.target_activity} Count')
        ax2.set_ylabel('Visit Duration (minutes)')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax2, label=f'{self.target_activity} Count')
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_activity_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def get_temporal_analysis_data(self):
        """
        Extract comprehensive temporal analysis data similar to region transit analysis
        
        Returns:
        --------
        dict
            Dictionary containing temporal analysis data
        """
        if not self.results:
            print("❌ No analysis results available. Run analyze_region_activity() first.")
            return None
        
        all_visits = self.results['all_visits']
        
        if not all_visits:
            print("⚠️  No visits data available for temporal analysis")
            return None
        
        print(f"🔍 Processing {len(all_visits)} visits for temporal analysis...")
        
        # Convert to DataFrame for easier analysis
        try:
            visits_df = pd.DataFrame([{
                'employee_id': visit['employee_id'],
                'shift': visit['shift'],
                'date': visit.get('date', 'unknown'),
                'start_time': visit['start_time'],
                'total_duration': visit['total_duration'],
                'target_activity_count': visit['target_activity_count'],
                'target_activity_duration': visit['target_activity_duration']
            } for visit in all_visits])
            
            # Convert start_time to datetime and extract components
            visits_df['datetime'] = pd.to_datetime(visits_df['start_time'], unit='s')
            visits_df['hour'] = visits_df['datetime'].dt.hour
            visits_df['day_of_week'] = visits_df['datetime'].dt.day_name()
            visits_df['date_only'] = visits_df['datetime'].dt.date
            
            print(f"✅ Successfully processed temporal data for {len(visits_df)} visits")
            
        except Exception as e:
            print(f"❌ Error processing temporal data: {e}")
            return None
        
        # Hourly analysis
        hourly_data = []
        for hour in range(24):
            hour_visits = visits_df[visits_df['hour'] == hour]
            
            total_visits = len(hour_visits)
            target_occurrences = hour_visits['target_activity_count'].sum()
            target_duration = hour_visits['target_activity_duration'].sum()
            visit_duration = hour_visits['total_duration'].sum()
            
            avg_target_per_visit = target_occurrences / total_visits if total_visits > 0 else 0
            avg_visit_duration = visit_duration / total_visits if total_visits > 0 else 0
            
            hourly_data.append({
                'hour': hour,
                'total_visits': total_visits,
                'target_activity_occurrences': target_occurrences,
                'target_activity_duration_seconds': target_duration,
                'total_visit_duration_seconds': visit_duration,
                'target_activity_duration_formatted': format_seconds_to_hms(target_duration),
                'total_visit_duration_formatted': format_seconds_to_hms(visit_duration),
                'avg_target_per_visit': round(avg_target_per_visit, 1),
                'avg_visit_duration_seconds': round(avg_visit_duration, 1),
                'avg_visit_duration_formatted': format_seconds_to_hms(avg_visit_duration),
                'target_percentage_of_time': round((target_duration / visit_duration * 100) if visit_duration > 0 else 0, 1)
            })
        
        # Daily analysis
        daily_data = []
        unique_dates = visits_df['date_only'].unique()
        # Convert to list and filter out any NaN values, then sort
        valid_dates = [d for d in unique_dates if pd.notna(d)]
        
        for date in sorted(valid_dates):
            day_visits = visits_df[visits_df['date_only'] == date]
            
            total_visits = len(day_visits)
            target_occurrences = day_visits['target_activity_count'].sum()
            target_duration = day_visits['target_activity_duration'].sum()
            visit_duration = day_visits['total_duration'].sum()
            unique_employees = day_visits['employee_id'].nunique()
            
            daily_data.append({
                'date': str(date),
                'day_of_week': day_visits['day_of_week'].iloc[0] if total_visits > 0 else '',
                'total_visits': total_visits,
                'unique_employees': unique_employees,
                'target_activity_occurrences': target_occurrences,
                'target_activity_duration_seconds': target_duration,
                'total_visit_duration_seconds': visit_duration,
                'target_activity_duration_formatted': format_seconds_to_hms(target_duration),
                'total_visit_duration_formatted': format_seconds_to_hms(visit_duration),
                'avg_target_per_visit': round(target_occurrences / total_visits if total_visits > 0 else 0, 1),
                'target_percentage_of_time': round((target_duration / visit_duration * 100) if visit_duration > 0 else 0, 1)
            })
        
        # Day of week analysis
        dow_data = []
        for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            dow_visits = visits_df[visits_df['day_of_week'] == dow]
            
            total_visits = len(dow_visits)
            target_occurrences = dow_visits['target_activity_count'].sum()
            target_duration = dow_visits['target_activity_duration'].sum()
            visit_duration = dow_visits['total_duration'].sum()
            
            dow_data.append({
                'day_of_week': dow,
                'total_visits': total_visits,
                'target_activity_occurrences': target_occurrences,
                'target_activity_duration_seconds': target_duration,
                'total_visit_duration_seconds': visit_duration,
                'target_activity_duration_formatted': format_seconds_to_hms(target_duration),
                'total_visit_duration_formatted': format_seconds_to_hms(visit_duration),
                'avg_target_per_visit': round(target_occurrences / total_visits if total_visits > 0 else 0, 1),
                'target_percentage_of_time': round((target_duration / visit_duration * 100) if visit_duration > 0 else 0, 1)
            })
        
        # Shift analysis
        shift_data = []
        if 'shift' in visits_df.columns:
            for shift in sorted(visits_df['shift'].unique()):
                shift_visits = visits_df[visits_df['shift'] == shift]
                
                total_visits = len(shift_visits)
                target_occurrences = shift_visits['target_activity_count'].sum()
                target_duration = shift_visits['target_activity_duration'].sum()
                visit_duration = shift_visits['total_duration'].sum()
                unique_employees = shift_visits['employee_id'].nunique()
                
                shift_data.append({
                    'shift': shift,
                    'total_visits': total_visits,
                    'unique_employees': unique_employees,
                    'target_activity_occurrences': target_occurrences,
                    'target_activity_duration_seconds': target_duration,
                    'total_visit_duration_seconds': visit_duration,
                    'target_activity_duration_formatted': format_seconds_to_hms(target_duration),
                    'total_visit_duration_formatted': format_seconds_to_hms(visit_duration),
                    'avg_target_per_visit': round(target_occurrences / total_visits if total_visits > 0 else 0, 1),
                    'target_percentage_of_time': round((target_duration / visit_duration * 100) if visit_duration > 0 else 0, 1)
                })
        
        # Peak analysis
        hourly_df = pd.DataFrame(hourly_data)
        daily_df = pd.DataFrame(daily_data)
        
        peak_visit_hour = None
        peak_target_hour = None  
        most_active_day = None
        
        if not hourly_df.empty and hourly_df['total_visits'].max() > 0:
            peak_visit_hour = hourly_df.loc[hourly_df['total_visits'].idxmax()]
            
        if not hourly_df.empty and hourly_df['target_activity_occurrences'].max() > 0:
            peak_target_hour = hourly_df.loc[hourly_df['target_activity_occurrences'].idxmax()]
            
        if not daily_df.empty and daily_df['total_visits'].max() > 0:
            most_active_day = daily_df.loc[daily_df['total_visits'].idxmax()]
        
        # Employee patterns
        employee_patterns = []
        for emp_id in visits_df['employee_id'].unique():
            emp_visits = visits_df[visits_df['employee_id'] == emp_id]
            
            # Daily patterns for this employee
            emp_daily = emp_visits.groupby('date_only').agg({
                'target_activity_count': 'sum',
                'total_duration': 'sum',
                'start_time': 'count'  # visit count
            }).reset_index()
            
            # Hourly patterns for this employee  
            emp_hourly = emp_visits.groupby('hour').agg({
                'target_activity_count': 'sum',
                'total_duration': 'sum',
                'start_time': 'count'  # visit count
            }).reset_index()
            
            # Find most active hour and day
            most_active_hour = None
            most_active_day = None
            
            if not emp_hourly.empty and emp_hourly['target_activity_count'].max() > 0:
                most_active_hour = emp_hourly.loc[emp_hourly['target_activity_count'].idxmax()]['hour']
                
            if not emp_daily.empty and emp_daily['target_activity_count'].max() > 0:
                most_active_day = emp_daily.loc[emp_daily['target_activity_count'].idxmax()]['date_only']
            
            employee_patterns.append({
                'employee_id': emp_id,
                'total_visits': len(emp_visits),
                'total_target_occurrences': emp_visits['target_activity_count'].sum(),
                'avg_target_per_visit': emp_visits['target_activity_count'].mean(),
                'most_active_hour': most_active_hour,
                'most_active_day': str(most_active_day) if most_active_day is not None else None,
                'daily_patterns': emp_daily.to_dict('records'),
                'hourly_patterns': emp_hourly.to_dict('records')
            })
        
        # Convert peak analysis results to dictionaries safely
        def safe_to_dict(obj):
            """Safely convert pandas Series or single values to dict"""
            if obj is None:
                return None
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            else:
                # If it's a single value, return as is
                return obj
        
        # Define safe_to_dict function first
        def safe_to_dict(obj):
            """Safely convert pandas Series or single values to dict"""
            if obj is None:
                return None
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            else:
                # If it's a single value, return as is
                return obj
                
        return {
            'hourly_analysis': hourly_data,
            'daily_analysis': daily_data,
            'day_of_week_analysis': dow_data,
            'shift_analysis': shift_data,
            'employee_patterns': employee_patterns,
            'peak_analysis': {
                'peak_visit_hour': safe_to_dict(peak_visit_hour),
                'peak_target_hour': safe_to_dict(peak_target_hour),
                'most_active_day': safe_to_dict(most_active_day)
            },
            'region_of_interest': self.region_of_interest,
            'target_activity': self.target_activity,
            'visit_threshold': self.visit_threshold,
            'analysis_period': {
                'start_date': str(min(valid_dates)) if valid_dates else None,
                'end_date': str(max(valid_dates)) if valid_dates else None,
                'total_days': len(valid_dates) if valid_dates else 0,
                'total_employees': visits_df['employee_id'].nunique() if not visits_df.empty else 0
            }
        }
    
    def print_temporal_summary(self):
        """Print detailed temporal analysis summary similar to region transit analysis"""
        print("🔍 Attempting to generate temporal analysis...")
        
        temporal_data = self.get_temporal_analysis_data()
        if not temporal_data:
            print("❌ Could not generate temporal data - check visit data format")
            return
        
        print(f"\n{'='*60}")
        print(f"🕐 DETAILED TEMPORAL ANALYSIS")
        print(f"{'='*60}")
        print(f"📍 Region: {temporal_data['region_of_interest']}")
        print(f"🎯 Target Activity: {temporal_data['target_activity']}")
        print(f"⏱️  Visit Threshold: {temporal_data['visit_threshold']} seconds")
        
        # Analysis period
        period = temporal_data['analysis_period']
        if period['start_date'] and period['end_date']:
            print(f"📅 Analysis Period: {period['start_date']} to {period['end_date']} ({period['total_days']} days)")
            print(f"👥 Total Employees: {period['total_employees']}")
        
        # Peak analysis
        print(f"\n📊 PEAK PATTERNS:")
        peak_data = temporal_data['peak_analysis']
        
        if peak_data.get('peak_visit_hour'):
            pv = peak_data['peak_visit_hour']
            print(f"   🏢 Most Visits: {pv['hour']}:00 ({pv['total_visits']} visits)")
        
        if peak_data.get('peak_target_hour'):
            pt = peak_data['peak_target_hour']
            print(f"   🎯 Most {temporal_data['target_activity']}: {pt['hour']}:00 ({pt['target_activity_occurrences']} occurrences)")
        
        if peak_data.get('most_active_day'):
            md = peak_data['most_active_day']
            if isinstance(md, dict) and 'date' in md:
                print(f"   📅 Most Active Day: {md['date']} ({md['total_visits']} visits, {md['target_activity_occurrences']} {temporal_data['target_activity']})")
        
        # Top 5 busiest hours
        print(f"\n⏰ TOP 5 BUSIEST HOURS:")
        hourly_df = pd.DataFrame(temporal_data['hourly_analysis'])
        if not hourly_df.empty:
            top_hours = hourly_df.nlargest(5, 'total_visits')
            
            for _, hour_data in top_hours.iterrows():
                if hour_data['total_visits'] > 0:
                    print(f"   • {hour_data['hour']:02d}:00 - {hour_data['total_visits']} visits, "
                          f"{hour_data['target_activity_occurrences']} {temporal_data['target_activity']} "
                          f"(avg {hour_data['avg_target_per_visit']:.1f} per visit)")
        
        # Day of week patterns
        print(f"\n📅 DAY OF WEEK PATTERNS:")
        dow_df = pd.DataFrame(temporal_data['day_of_week_analysis'])
        dow_df_active = dow_df[dow_df['total_visits'] > 0]  # Only show days with data
        
        if not dow_df_active.empty:
            for _, day_data in dow_df_active.iterrows():
                print(f"   • {day_data['day_of_week']}: {day_data['total_visits']} visits, "
                      f"{day_data['target_activity_occurrences']} {temporal_data['target_activity']} "
                      f"(avg {day_data['avg_target_per_visit']:.1f} per visit)")
        
        # Shift patterns
        if temporal_data.get('shift_analysis'):
            print(f"\n👥 SHIFT PATTERNS:")
            for shift_data in temporal_data['shift_analysis']:
                if shift_data['total_visits'] > 0:
                    print(f"   • Shift {shift_data['shift']}: {shift_data['total_visits']} visits, "
                          f"{shift_data['target_activity_occurrences']} {temporal_data['target_activity']} "
                          f"(avg {shift_data['avg_target_per_visit']:.1f} per visit)")
        
        # Employee insights
        print(f"\n👤 EMPLOYEE INSIGHTS:")
        for emp_pattern in temporal_data['employee_patterns'][:5]:  # Top 5 employees
            emp_id = emp_pattern['employee_id']
            most_active_hour = emp_pattern.get('most_active_hour')
            avg_target = emp_pattern.get('avg_target_per_visit', 0)
            if most_active_hour is not None:
                print(f"   • {emp_id}: Peak hour {most_active_hour}:00, "
                      f"avg {avg_target:.1f} {temporal_data['target_activity']} per visit")
            else:
                print(f"   • {emp_id}: avg {avg_target:.1f} {temporal_data['target_activity']} per visit")
    
    def create_comprehensive_visualizations(self, output_dir=None):
        """Create comprehensive visualizations similar to region transit analysis"""
        if not self.results:
            print("❌ No analysis results available. Run analyze_region_activity() first.")
            return
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            set_visualization_style()
        except:
            plt.style.use('default')
        
        # Get temporal data
        temporal_data = self.get_temporal_analysis_data()
        if not temporal_data:
            print("⚠️  No temporal data for comprehensive visualizations")
            return
        
        # 1. Enhanced temporal patterns
        self._create_enhanced_temporal_chart(temporal_data, output_dir)
        
        # 2. Employee comparison donut charts
        self._create_employee_donut_charts(temporal_data, output_dir)
        
        # 3. Activity intensity heatmap
        self._create_activity_intensity_heatmap(temporal_data, output_dir)
        
        # 4. Daily and weekly patterns
        self._create_weekly_patterns_chart(temporal_data, output_dir)
        
        # 5. Visit characteristics analysis
        self._create_visit_characteristics_chart(output_dir)
    
    def _create_enhanced_temporal_chart(self, temporal_data, output_dir):
        """Create enhanced temporal analysis chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Hourly visit patterns
        hourly_df = pd.DataFrame(temporal_data['hourly_analysis'])
        hours = hourly_df['hour']
        visits = hourly_df['total_visits']
        target_counts = hourly_df['target_activity_occurrences']
        
        ax1.bar(hours, visits, alpha=0.7, color='skyblue', label='Total Visits')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(hours, target_counts, 'ro-', linewidth=2, markersize=4, 
                     label=f'{temporal_data["target_activity"]} Count', color='red')
        
        ax1.set_title('Hourly Visit and Activity Patterns', fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Visits', color='blue')
        ax1_twin.set_ylabel(f'{temporal_data["target_activity"]} Count', color='red')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Daily patterns
        daily_df = pd.DataFrame(temporal_data['daily_analysis'])
        if not daily_df.empty:
            dates = range(len(daily_df))
            daily_visits = daily_df['total_visits']
            daily_targets = daily_df['target_activity_occurrences']
            
            ax2.plot(dates, daily_visits, 'b-o', linewidth=2, markersize=4, label='Visits')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(dates, daily_targets, 'r-s', linewidth=2, markersize=4, 
                         label=f'{temporal_data["target_activity"]}', color='red')
            
            ax2.set_title('Daily Activity Patterns', fontweight='bold')
            ax2.set_xlabel('Days')
            ax2.set_ylabel('Visits', color='blue')
            ax2_twin.set_ylabel(f'{temporal_data["target_activity"]} Count', color='red')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        # Day of week patterns
        dow_df = pd.DataFrame(temporal_data['day_of_week_analysis'])
        dow_df = dow_df[dow_df['total_visits'] > 0]
        
        if not dow_df.empty:
            days = dow_df['day_of_week']
            dow_visits = dow_df['total_visits']
            dow_targets = dow_df['target_activity_occurrences']
            
            x_pos = range(len(days))
            width = 0.35
            
            bars1 = ax3.bar([x - width/2 for x in x_pos], dow_visits, width, 
                           label='Visits', color='lightblue')
            bars2 = ax3.bar([x + width/2 for x in x_pos], dow_targets, width, 
                           label=f'{temporal_data["target_activity"]}', color='lightcoral')
            
            ax3.set_title('Day of Week Patterns', fontweight='bold')
            ax3.set_xlabel('Day of Week')
            ax3.set_ylabel('Count')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(days, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Average activity per visit by hour
        avg_per_visit = hourly_df['avg_target_per_visit']
        ax4.plot(hours, avg_per_visit, 'go-', linewidth=2, markersize=6)
        ax4.fill_between(hours, avg_per_visit, alpha=0.3, color='green')
        ax4.set_title(f'Average {temporal_data["target_activity"]} per Visit by Hour', fontweight='bold')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel(f'Avg {temporal_data["target_activity"]} per Visit')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_enhanced_temporal_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved enhanced temporal analysis: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_employee_donut_charts(self, temporal_data, output_dir):
        """Create employee comparison donut charts"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Employee visit distribution
        employee_visits = {}
        employee_targets = {}
        
        for emp_pattern in temporal_data['employee_patterns']:
            emp_id = emp_pattern['employee_id']
            employee_visits[emp_id] = emp_pattern['total_visits']
            employee_targets[emp_id] = emp_pattern['total_target_occurrences']
        
        if employee_visits:
            # Visits donut
            labels = list(employee_visits.keys())
            sizes = list(employee_visits.values())
            
            try:
                employee_colors = get_employee_colors()
                colors = [employee_colors.get(emp, '#cccccc') for emp in labels]
            except:
                colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90, pctdistance=0.85)
            
            # Create donut by adding white circle
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            ax1.add_artist(centre_circle)
            ax1.set_title('Visit Distribution by Employee', fontweight='bold')
            
            # Add center text
            total_visits = sum(sizes)
            ax1.text(0, 0, f"Total\n{total_visits}\nVisits", ha='center', va='center', 
                    fontsize=14, fontweight='bold')
        
        if employee_targets:
            # Target activity donut
            target_labels = list(employee_targets.keys())
            target_sizes = list(employee_targets.values())
            
            # Filter out zero values
            filtered_data = [(label, size) for label, size in zip(target_labels, target_sizes) if size > 0]
            if filtered_data:
                target_labels, target_sizes = zip(*filtered_data)
                
                try:
                    colors = [employee_colors.get(emp, '#cccccc') for emp in target_labels]
                except:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(target_labels)))
                
                wedges, texts, autotexts = ax2.pie(target_sizes, labels=target_labels, colors=colors,
                                                  autopct='%1.1f%%', startangle=90, pctdistance=0.85)
                
                # Create donut
                centre_circle = plt.Circle((0,0), 0.70, fc='white')
                ax2.add_artist(centre_circle)
                ax2.set_title(f'{temporal_data["target_activity"]} Distribution by Employee', fontweight='bold')
                
                # Add center text
                total_targets = sum(target_sizes)
                ax2.text(0, 0, f"Total\n{total_targets}\n{temporal_data['target_activity']}", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_employee_donut_charts.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved employee donut charts: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_activity_intensity_heatmap(self, temporal_data, output_dir):
        """Create activity intensity heatmap"""
        # Create employee x hour heatmap
        employees = [emp['employee_id'] for emp in temporal_data['employee_patterns']]
        
        if not employees:
            print("⚠️  No employee data for heatmap")
            return
        
        # Create matrix: employees x hours
        intensity_matrix = np.zeros((len(employees), 24))
        
        for i, emp_pattern in enumerate(temporal_data['employee_patterns']):
            for hour_data in emp_pattern['hourly_patterns']:
                hour = hour_data['hour']
                target_count = hour_data['target_activity_count']
                intensity_matrix[i, hour] = target_count
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(intensity_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{h}:00' for h in range(24)])
        ax.set_yticks(range(len(employees)))
        ax.set_yticklabels(employees)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{temporal_data["target_activity"]} Count')
        
        # Add text annotations
        for i in range(len(employees)):
            for j in range(24):
                if intensity_matrix[i, j] > 0:
                    text = ax.text(j, i, int(intensity_matrix[i, j]), 
                                 ha='center', va='center', color='black', fontweight='bold')
        
        ax.set_title(f'{temporal_data["target_activity"]} Intensity Heatmap\n(Employee × Hour)', 
                    fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Employee')
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_activity_intensity_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved activity intensity heatmap: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_weekly_patterns_chart(self, temporal_data, output_dir):
        """Create weekly and daily pattern analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Weekly pattern (day of week)
        dow_df = pd.DataFrame(temporal_data['day_of_week_analysis'])
        dow_df = dow_df[dow_df['total_visits'] > 0]
        
        if not dow_df.empty:
            days = dow_df['day_of_week']
            visits = dow_df['total_visits']
            targets = dow_df['target_activity_occurrences']
            percentages = dow_df['target_percentage_of_time']
            
            x_pos = range(len(days))
            
            # Stacked bar chart
            ax1.bar(x_pos, targets, color='lightcoral', label=f'{temporal_data["target_activity"]} Time', alpha=0.8)
            
            # Add visit counts as line
            ax1_twin = ax1.twinx()
            ax1_twin.plot(x_pos, visits, 'bo-', linewidth=2, markersize=6, label='Total Visits')
            
            ax1.set_title('Weekly Activity Patterns', fontweight='bold')
            ax1.set_xlabel('Day of Week')
            ax1.set_ylabel(f'{temporal_data["target_activity"]} Duration (seconds)', color='red')
            ax1_twin.set_ylabel('Number of Visits', color='blue')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(days, rotation=45)
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        
        # Daily progression
        daily_df = pd.DataFrame(temporal_data['daily_analysis'])
        if not daily_df.empty:
            dates = range(len(daily_df))
            daily_visits = daily_df['total_visits']
            daily_targets = daily_df['target_activity_occurrences']
            
            ax2.fill_between(dates, daily_visits, alpha=0.5, color='skyblue', label='Visits')
            ax2.plot(dates, daily_visits, 'b-', linewidth=2, label='Visit Trend')
            
            ax2_twin = ax2.twinx()
            ax2_twin.fill_between(dates, daily_targets, alpha=0.5, color='lightcoral', 
                                 label=f'{temporal_data["target_activity"]}')
            ax2_twin.plot(dates, daily_targets, 'r-', linewidth=2, 
                         label=f'{temporal_data["target_activity"]} Trend')
            
            ax2.set_title('Daily Activity Progression', fontweight='bold')
            ax2.set_xlabel('Days in Analysis Period')
            ax2.set_ylabel('Number of Visits', color='blue')
            ax2_twin.set_ylabel(f'{temporal_data["target_activity"]} Count', color='red')
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_weekly_patterns.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved weekly patterns: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_visit_characteristics_chart(self, output_dir):
        """Create visit characteristics analysis"""
        all_visits = self.results['all_visits']
        
        if not all_visits:
            print("⚠️  No visits to analyze characteristics")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Visit duration distribution
        durations = [visit['total_duration'] for visit in all_visits]
        ax1.hist(durations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(durations), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(durations):.1f}s')
        ax1.axvline(np.median(durations), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(durations):.1f}s')
        ax1.set_title('Visit Duration Distribution', fontweight='bold')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Target activity count distribution
        target_counts = [visit['target_activity_count'] for visit in all_visits]
        ax2.hist(target_counts, bins=max(1, max(target_counts) + 1), alpha=0.7, 
                color='lightcoral', edgecolor='black')
        ax2.axvline(np.mean(target_counts), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(target_counts):.1f}')
        ax2.set_title(f'{self.target_activity} Count Distribution', fontweight='bold')
        ax2.set_xlabel(f'{self.target_activity} Count per Visit')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Duration vs Activity Count scatter
        ax3.scatter(target_counts, durations, alpha=0.6, c=target_counts, cmap='viridis')
        ax3.set_title(f'Visit Duration vs {self.target_activity} Count', fontweight='bold')
        ax3.set_xlabel(f'{self.target_activity} Count')
        ax3.set_ylabel('Visit Duration (seconds)')
        ax3.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(target_counts, durations)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Activity percentage in visits
        target_percentages = [visit['target_activity_percentage'] for visit in all_visits]
        ax4.hist(target_percentages, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(np.mean(target_percentages), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(target_percentages):.1f}%')
        ax4.set_title(f'{self.target_activity} Percentage of Visit Time', fontweight='bold')
        ax4.set_xlabel('Percentage (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_visit_characteristics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved visit characteristics: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def export_results(self, output_dir):
        """Export all results to CSV files"""
        if not self.results:
            print("❌ No analysis results available. Run analyze_region_activity() first.")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export detailed visit data
        visit_df = self.get_detailed_visit_data()
        if visit_df is not None and not visit_df.empty:
            visit_path = output_dir / f'{self.region_of_interest}_detailed_visits.csv'
            visit_df.to_csv(visit_path, index=False)
            print(f"✅ Exported detailed visits: {visit_path}")
        
        # 2. Export employee summary
        summary_df = self.get_employee_summary_data()
        if summary_df is not None and not summary_df.empty:
            summary_path = output_dir / f'{self.region_of_interest}_employee_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"✅ Exported employee summary: {summary_path}")
        
        # 3. Export overall statistics
        stats = self.results['overall_stats']
        stats_data = [{
            'metric': 'Total Visits',
            'value': stats['total_visits']
        }, {
            'metric': 'Total Target Activity Occurrences',
            'value': stats['total_target_occurrences']
        }, {
            'metric': 'Average Target Activity per Visit',
            'value': round(stats['avg_target_per_visit'], 2)
        }, {
            'metric': 'Average Visit Duration (seconds)',
            'value': round(stats['avg_visit_duration'], 1)
        }, {
            'metric': 'Total Target Activity Duration (seconds)',
            'value': stats['total_target_duration']
        }]
        
        stats_df = pd.DataFrame(stats_data)
        stats_path = output_dir / f'{self.region_of_interest}_overall_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f"✅ Exported overall statistics: {stats_path}")
        
        # 4. Export temporal analysis data (similar to region transit analysis)
        print("🕐 Generating temporal analysis data...")
        temporal_data = self.get_temporal_analysis_data()
        
        if temporal_data:
            # Export hourly analysis
            hourly_df = pd.DataFrame(temporal_data['hourly_analysis'])
            hourly_path = output_dir / f'{self.region_of_interest}_hourly_analysis.csv'
            hourly_df.to_csv(hourly_path, index=False)
            print(f"✅ Exported hourly analysis: {hourly_path}")
            
            # Export daily analysis
            if temporal_data['daily_analysis']:
                daily_df = pd.DataFrame(temporal_data['daily_analysis'])
                daily_path = output_dir / f'{self.region_of_interest}_daily_analysis.csv'
                daily_df.to_csv(daily_path, index=False)
                print(f"✅ Exported daily analysis: {daily_path}")
            
            # Export day of week analysis
            dow_df = pd.DataFrame(temporal_data['day_of_week_analysis'])
            dow_path = output_dir / f'{self.region_of_interest}_day_of_week_analysis.csv'
            dow_df.to_csv(dow_path, index=False)
            print(f"✅ Exported day of week analysis: {dow_path}")
            
            # Export shift analysis (if available)
            if temporal_data['shift_analysis']:
                shift_df = pd.DataFrame(temporal_data['shift_analysis'])
                shift_path = output_dir / f'{self.region_of_interest}_shift_analysis.csv'
                shift_df.to_csv(shift_path, index=False)
                print(f"✅ Exported shift analysis: {shift_path}")
            
            # Export employee patterns
            employee_summary = []
            for emp_pattern in temporal_data['employee_patterns']:
                employee_summary.append({
                    'employee_id': emp_pattern['employee_id'],
                    'total_visits': emp_pattern['total_visits'],
                    'total_target_occurrences': emp_pattern['total_target_occurrences'],
                    'avg_target_per_visit': round(emp_pattern['avg_target_per_visit'], 2),
                    'most_active_hour': emp_pattern['most_active_hour'],
                    'most_active_day': str(emp_pattern['most_active_day']) if emp_pattern['most_active_day'] else None
                })
            
            if employee_summary:
                emp_patterns_df = pd.DataFrame(employee_summary)
                emp_patterns_path = output_dir / f'{self.region_of_interest}_employee_patterns.csv'
                emp_patterns_df.to_csv(emp_patterns_path, index=False)
                print(f"✅ Exported employee patterns: {emp_patterns_path}")
            
            # Export peak analysis summary
            if temporal_data['peak_analysis']:
                peak_summary = []
                peak_data = temporal_data['peak_analysis']
                
                if peak_data['peak_visit_hour']:
                    peak_summary.append({
                        'metric': 'Peak Visit Hour',
                        'hour': peak_data['peak_visit_hour']['hour'],
                        'value': peak_data['peak_visit_hour']['total_visits'],
                        'description': f"Hour {peak_data['peak_visit_hour']['hour']} has the most visits ({peak_data['peak_visit_hour']['total_visits']})"
                    })
                
                if peak_data['peak_target_hour']:
                    peak_summary.append({
                        'metric': f'Peak {self.target_activity} Hour',
                        'hour': peak_data['peak_target_hour']['hour'],
                        'value': peak_data['peak_target_hour']['target_activity_occurrences'],
                        'description': f"Hour {peak_data['peak_target_hour']['hour']} has the most {self.target_activity} activities ({peak_data['peak_target_hour']['target_activity_occurrences']})"
                    })
                
                if peak_data['most_active_day']:
                    peak_summary.append({
                        'metric': 'Most Active Day',
                        'hour': 'N/A',
                        'value': peak_data['most_active_day']['total_visits'],
                        'description': f"Date {peak_data['most_active_day']['date']} has the most visits ({peak_data['most_active_day']['total_visits']})"
                    })
                
                if peak_summary:
                    peak_df = pd.DataFrame(peak_summary)
                    peak_path = output_dir / f'{self.region_of_interest}_peak_analysis.csv'
                    peak_df.to_csv(peak_path, index=False)
                    print(f"✅ Exported peak analysis: {peak_path}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Analyze activity frequency within a specific region')
    parser.add_argument('--region', type=str, required=True,
                       help='Region of interest to analyze (e.g., "7e_Brotwagon_und_kisten")')
    parser.add_argument('--activity', type=str, required=True,
                       help='Target activity to count (e.g., "Handle down")')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Visit threshold in seconds (default: 10)')
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                       help='Path to the sensor data CSV file')
    parser.add_argument('--output', type=str, default='region_activity_output',
                       help='Output directory for results and visualizations')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip creating visualizations')
    parser.add_argument('--temporal-detail', action='store_true',
                       help='Show detailed temporal analysis in console output')
    parser.add_argument('--comprehensive-viz', action='store_true',
                       help='Create comprehensive visualizations (includes temporal patterns, heatmaps, etc.)')
    
    args = parser.parse_args()
    
    print("📊 SIMPLE REGION ACTIVITY ANALYZER")
    print("=" * 50)
    
    # Load data
    try:
        print(f"📁 Loading data from: {args.data}")
        if 'load_sensor_data' in globals():
            data = load_sensor_data(args.data)
            data = classify_departments(data)
        else:
            data = pd.read_csv(args.data)
        print(f"✅ Loaded {len(data):,} records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Check if region and activity exist in data
    available_regions = data['region'].unique()
    available_activities = data['activity'].unique()
    
    if args.region not in available_regions:
        print(f"❌ Region '{args.region}' not found in data.")
        print(f"📋 Available regions: {', '.join(sorted(available_regions))}")
        return 1
    
    if args.activity not in available_activities:
        print(f"❌ Activity '{args.activity}' not found in data.")
        print(f"📋 Available activities: {', '.join(sorted(available_activities))}")
        return 1
    
    # Initialize analyzer
    analyzer = SimpleRegionActivityAnalyzer(data, args.region, args.activity, args.threshold)
    
    # Run analysis
    print(f"\n🔍 Starting analysis...")
    results = analyzer.analyze_region_activity()
    
    # Print summary
    analyzer.print_summary()
    
    # Print detailed temporal analysis if requested
    if args.temporal_detail:
        analyzer.print_temporal_summary()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export results
    print(f"\n💾 Exporting results...")
    analyzer.export_results(output_dir)
    
    # Create visualizations
    if not args.no_viz:
        print(f"\n📊 Creating visualizations...")
        
        if args.comprehensive_viz:
            print("🔬 Creating comprehensive visualizations...")
            analyzer.create_comprehensive_visualizations(output_dir)
        else:
            print("📈 Creating basic visualizations...")
            analyzer.create_visualizations(output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)