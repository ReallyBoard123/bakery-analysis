#!/usr/bin/env python3
"""
Region Transit Analysis Tool
============================

Analyzes transitions through a specific region of interest to determine if the region
is being used primarily for transit between other regions.

A transition is defined as: Region A → Region of Interest → Region B
where the time spent in the Region of Interest is below a specified threshold.

Usage:
    python region_transit_analysis.py --region "8_Korridor" --threshold 15
    python region_transit_analysis.py --region "7a_Brotwagon_und_kisten" --threshold 10 --output results/
    python region_transit_analysis.py --region "8_Korridor" --temporal-detail --no-viz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict, Counter
from datetime import datetime
import sys

# Import utilities from the existing codebase
try:
    from src.utils.time_utils import format_seconds_to_hms
    from src.utils.data_utils import load_sensor_data, classify_departments
    from src.visualization.base import save_figure, set_visualization_style, get_region_colors, get_employee_colors
except ImportError:
    print("Warning: Could not import from src modules. Some features may be limited.")
    
    def format_seconds_to_hms(seconds):
        """Fallback time formatting function"""
        if seconds is None or seconds < 0:
            return "00:00:00"
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class RegionTransitAnalyzer:
    """Analyzes transit patterns through a specific region of interest"""
    
    def __init__(self, data, region_of_interest, transit_threshold_seconds=15):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Employee tracking data
        region_of_interest : str
            The region to analyze for transit patterns
        transit_threshold_seconds : int
            Maximum seconds spent in region to be considered transit
        """
        self.data = data.copy()
        self.region_of_interest = region_of_interest
        self.transit_threshold = transit_threshold_seconds
        self.results = None
        
        # Prepare the data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and sort data for analysis"""
        # Ensure data is sorted by employee and time
        self.data = self.data.sort_values(['id', 'startTime']).reset_index(drop=True)
        print(f"📊 Analyzing {len(self.data):,} records for region: {self.region_of_interest}")
        print(f"🔄 Transit threshold: {self.transit_threshold} seconds")
        
    def analyze_transitions(self):
        """
        Analyze all transitions through the region of interest
        
        Returns:
        --------
        dict
            Comprehensive analysis results
        """
        print(f"\n🔍 Analyzing transitions through {self.region_of_interest}...")
        
        all_transitions = []
        employee_stats = defaultdict(lambda: {
            'total_transits': 0,
            'total_visits': 0,
            'transit_time': 0,
            'work_time': 0,
            'transition_patterns': []
        })
        
        # Process each employee separately
        for employee_id in self.data['id'].unique():
            emp_data = self.data[self.data['id'] == employee_id].sort_values('startTime').reset_index(drop=True)
            
            # Find transition sessions (consecutive visits to region of interest)
            sessions = self._find_roi_sessions(emp_data)
            
            for session in sessions:
                prev_region = session['previous_region']
                next_region = session['next_region']
                total_duration = session['total_duration']
                
                # Skip if no valid transition (must come from and go to different regions)
                if prev_region == self.region_of_interest or next_region == self.region_of_interest:
                    continue
                
                is_transit = total_duration <= self.transit_threshold
                
                # Record the transition
                transition = {
                    'employee_id': employee_id,
                    'shift': session['shift'],
                    'date': session['date'],
                    'start_time': session['start_time'],
                    'end_time': session['end_time'],
                    'duration': total_duration,
                    'activities': session['activities'],
                    'previous_region': prev_region,
                    'next_region': next_region,
                    'is_transit': is_transit,
                    'transition_type': self._classify_transition_type(prev_region, next_region),
                    'records_count': session['records_count']
                }
                
                all_transitions.append(transition)
                
                # Update employee statistics
                employee_stats[employee_id]['total_visits'] += 1
                if is_transit:
                    employee_stats[employee_id]['total_transits'] += 1
                    employee_stats[employee_id]['transit_time'] += total_duration
                    employee_stats[employee_id]['transition_patterns'].append(
                        f"{prev_region} → {self.region_of_interest} → {next_region}"
                    )
                else:
                    employee_stats[employee_id]['work_time'] += total_duration
        
        # Convert to DataFrame for easier analysis
        transitions_df = pd.DataFrame(all_transitions)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(transitions_df, employee_stats)
        
        self.results = {
            'transitions_df': transitions_df,
            'employee_stats': dict(employee_stats),
            'summary_stats': summary_stats,
            'region_of_interest': self.region_of_interest,
            'transit_threshold': self.transit_threshold
        }
        
        return self.results
    
    def _find_roi_sessions(self, emp_data):
        """
        Find sessions of consecutive visits to the region of interest.
        Groups consecutive records in ROI and sums their durations.
        
        Parameters:
        -----------
        emp_data : pandas.DataFrame
            Employee data sorted by time
            
        Returns:
        --------
        list
            List of session dictionaries with aggregated information
        """
        sessions = []
        i = 0
        
        while i < len(emp_data):
            row = emp_data.iloc[i]
            
            # Check if this is the start of an ROI session
            if row['region'] == self.region_of_interest:
                # Find previous region (the region before entering ROI)
                prev_region = 'START'
                if i > 0:
                    prev_region = emp_data.iloc[i-1]['region']
                
                # Collect all consecutive ROI records
                session_start = i
                session_records = []
                total_duration = 0
                activities = []
                
                # Continue while we're still in the ROI
                while i < len(emp_data) and emp_data.iloc[i]['region'] == self.region_of_interest:
                    record = emp_data.iloc[i]
                    session_records.append(record)
                    total_duration += record['duration']
                    activities.append(record['activity'])
                    i += 1
                
                # Find next region (the region after leaving ROI)
                next_region = 'END'
                if i < len(emp_data):
                    next_region = emp_data.iloc[i]['region']
                
                # Only create session if we have valid previous and next regions
                # (i.e., this is a true transition through the ROI)
                if prev_region != self.region_of_interest and next_region != self.region_of_interest:
                    session = {
                        'previous_region': prev_region,
                        'next_region': next_region,
                        'total_duration': total_duration,
                        'start_time': session_records[0]['startTime'],
                        'end_time': session_records[-1]['endTime'],
                        'shift': session_records[0]['shift'],
                        'date': session_records[0]['date'],
                        'activities': list(set(activities)),  # Unique activities
                        'records_count': len(session_records),
                        'session_records': session_records
                    }
                    sessions.append(session)
            else:
                i += 1
        
        return sessions
    
    def _classify_transition_type(self, prev_region, next_region):
        """Classify the type of transition"""
        if prev_region == 'START' and next_region == 'END':
            return 'isolated'
        elif prev_region == 'START':
            return 'entry'
        elif next_region == 'END':
            return 'exit'
        elif prev_region == next_region:
            return 'return_loop'
        else:
            return 'through_transit'
    
    def _calculate_summary_statistics(self, transitions_df, employee_stats):
        """Calculate comprehensive summary statistics"""
        if transitions_df.empty:
            return {
                'total_visits': 0,
                'total_transits': 0,
                'transit_percentage': 0,
                'avg_transit_duration': 0,
                'avg_work_duration': 0
            }
        
        total_visits = len(transitions_df)
        total_transits = len(transitions_df[transitions_df['is_transit'] == True])
        transit_percentage = (total_transits / total_visits * 100) if total_visits > 0 else 0
        
        transit_visits = transitions_df[transitions_df['is_transit'] == True]
        work_visits = transitions_df[transitions_df['is_transit'] == False]
        
        avg_transit_duration = transit_visits['duration'].mean() if not transit_visits.empty else 0
        avg_work_duration = work_visits['duration'].mean() if not work_visits.empty else 0
        
        # Most common transition patterns
        common_patterns = Counter()
        for emp_stats in employee_stats.values():
            common_patterns.update(emp_stats['transition_patterns'])
        
        return {
            'total_visits': total_visits,
            'total_transits': total_transits,
            'total_work_visits': total_visits - total_transits,
            'transit_percentage': transit_percentage,
            'work_percentage': 100 - transit_percentage,
            'avg_transit_duration': avg_transit_duration,
            'avg_work_duration': avg_work_duration,
            'total_transit_time': transit_visits['duration'].sum(),
            'total_work_time': work_visits['duration'].sum(),
            'most_common_patterns': common_patterns.most_common(10)
        }
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis"""
        if not self.results:
            print("❌ No analysis results available. Run analyze_transitions() first.")
            return
        
        stats = self.results['summary_stats']
        
        print(f"\n{'='*60}")
        print(f"🎯 TRANSIT ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"📍 Region of Interest: {self.region_of_interest}")
        print(f"⏱️  Transit Threshold: {self.transit_threshold} seconds")
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   • Total visits to region: {stats['total_visits']:,}")
        print(f"   • Transit visits: {stats['total_transits']:,} ({stats['transit_percentage']:.1f}%)")
        print(f"   • Work visits: {stats['total_work_visits']:,} ({stats['work_percentage']:.1f}%)")
        
        print(f"\n⏰ TIME ANALYSIS:")
        print(f"   • Average transit duration: {format_seconds_to_hms(stats['avg_transit_duration'])}")
        print(f"   • Average work duration: {format_seconds_to_hms(stats['avg_work_duration'])}")
        print(f"   • Total transit time: {format_seconds_to_hms(stats['total_transit_time'])}")
        print(f"   • Total work time: {format_seconds_to_hms(stats['total_work_time'])}")
        
        print(f"\n🔄 TOP TRANSITION PATTERNS:")
        for i, (pattern, count) in enumerate(stats['most_common_patterns'][:5], 1):
            print(f"   {i}. {pattern} ({count} times)")
        
        print(f"\n👥 EMPLOYEE BREAKDOWN:")
        for emp_id, emp_stats in self.results['employee_stats'].items():
            transit_pct = (emp_stats['total_transits'] / emp_stats['total_visits'] * 100) if emp_stats['total_visits'] > 0 else 0
            print(f"   • {emp_id}: {emp_stats['total_transits']}/{emp_stats['total_visits']} transits ({transit_pct:.1f}%)")
    
    def create_visualizations(self, output_dir=None):
        """Create comprehensive visualizations of the transit analysis"""
        if not self.results:
            print("❌ No analysis results available. Run analyze_transitions() first.")
            return
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            set_visualization_style()
        except:
            plt.style.use('default')
        
        # 1. Transit vs Work Distribution
        self._create_transit_distribution_chart(output_dir)
        
        # 2. Employee Transit Patterns
        self._create_employee_transit_chart(output_dir)
        
        # 3. Duration Distribution Analysis
        self._create_duration_distribution_chart(output_dir)
        
        # 4. Transition Pattern Heatmap
        self._create_transition_pattern_chart(output_dir)
        
        # 5. Time-based Analysis
        self._create_temporal_analysis_chart(output_dir)
    
    def _create_transit_distribution_chart(self, output_dir):
        """Create a chart showing transit vs work distribution"""
        stats = self.results['summary_stats']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart of visit types
        labels = ['Transit Visits', 'Work Visits']
        sizes = [stats['total_transits'], stats['total_work_visits']]
        colors = ['#ff9999', '#66b3ff']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Visit Type Distribution\n{self.region_of_interest}', fontweight='bold')
        
        # Bar chart of time distribution
        time_labels = ['Transit Time', 'Work Time']
        time_values = [stats['total_transit_time']/3600, stats['total_work_time']/3600]  # Convert to hours
        
        bars = ax2.bar(time_labels, time_values, color=colors)
        ax2.set_title('Time Distribution (Hours)', fontweight='bold')
        ax2.set_ylabel('Hours')
        
        # Add value labels on bars
        for bar, value in zip(bars, time_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.1f}h', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_transit_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_employee_transit_chart(self, output_dir):
        """Create a chart showing transit patterns by employee"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        employees = []
        transit_counts = []
        total_counts = []
        
        for emp_id, stats in self.results['employee_stats'].items():
            employees.append(emp_id)
            transit_counts.append(stats['total_transits'])
            total_counts.append(stats['total_visits'])
        
        # Create stacked bar chart
        work_counts = [total - transit for total, transit in zip(total_counts, transit_counts)]
        
        try:
            employee_colors = get_employee_colors()
            colors = [employee_colors.get(emp, '#cccccc') for emp in employees]
        except:
            colors = plt.cm.Set3(np.linspace(0, 1, len(employees)))
        
        x = np.arange(len(employees))
        width = 0.6
        
        p1 = ax.bar(x, transit_counts, width, label='Transit Visits', color='#ff9999')
        p2 = ax.bar(x, work_counts, width, bottom=transit_counts, label='Work Visits', color='#66b3ff')
        
        ax.set_xlabel('Employee ID')
        ax.set_ylabel('Number of Visits')
        ax.set_title(f'Transit vs Work Visits by Employee\n{self.region_of_interest}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(employees)
        ax.legend()
        
        # Add percentage labels
        for i, (transit, total) in enumerate(zip(transit_counts, total_counts)):
            if total > 0:
                pct = transit / total * 100
                ax.text(i, total + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_employee_patterns.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_duration_distribution_chart(self, output_dir):
        """Create a chart showing duration distributions"""
        transitions_df = self.results['transitions_df']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of all durations with threshold line
        ax1.hist(transitions_df['duration'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=self.transit_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Transit Threshold ({self.transit_threshold}s)')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Duration Distribution of All Visits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparing transit vs work durations
        transit_durations = transitions_df[transitions_df['is_transit']]['duration']
        work_durations = transitions_df[~transitions_df['is_transit']]['duration']
        
        box_data = [transit_durations.dropna(), work_durations.dropna()]
        box_labels = ['Transit', 'Work']
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('#ff9999')
        bp['boxes'][1].set_facecolor('#66b3ff')
        
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Duration Comparison: Transit vs Work')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_duration_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_transition_pattern_chart(self, output_dir):
        """Create a chart showing the most common transition patterns"""
        patterns = self.results['summary_stats']['most_common_patterns'][:10]
        
        if not patterns:
            print("⚠️  No transition patterns to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pattern_names = [pattern.replace(' → ', '\n→\n') for pattern, count in patterns]
        counts = [count for pattern, count in patterns]
        
        bars = ax.barh(range(len(patterns)), counts, color='lightcoral')
        ax.set_yticks(range(len(patterns)))
        ax.set_yticklabels(pattern_names, fontsize=10)
        ax.set_xlabel('Frequency')
        ax.set_title(f'Most Common Transit Patterns\n{self.region_of_interest}', fontweight='bold')
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   str(count), ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_transition_patterns.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def _create_temporal_analysis_chart(self, output_dir):
        """Create temporal analysis showing transit patterns over time"""
        transitions_df = self.results['transitions_df']
        
        if 'date' not in transitions_df.columns:
            print("⚠️  No date information available for temporal analysis")
            return
        
        # Convert start_time to hour of day
        transitions_df = transitions_df.copy()
        transitions_df['hour'] = pd.to_datetime(transitions_df['start_time'], unit='s').dt.hour
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Transit patterns by hour of day
        hourly_transits = transitions_df[transitions_df['is_transit']].groupby('hour').size()
        hourly_total = transitions_df.groupby('hour').size()
        
        hours = range(24)
        transit_counts = [hourly_transits.get(h, 0) for h in hours]
        total_counts = [hourly_total.get(h, 0) for h in hours]
        
        ax1.bar(hours, transit_counts, alpha=0.7, label='Transit Visits', color='#ff9999')
        ax1.bar(hours, [total - transit for total, transit in zip(total_counts, transit_counts)], 
               bottom=transit_counts, alpha=0.7, label='Work Visits', color='#66b3ff')
        
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Visits')
        ax1.set_title('Transit vs Work Visits by Hour of Day')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # Transit percentage by hour
        transit_percentages = []
        for h in hours:
            total = hourly_total.get(h, 0)
            transit = hourly_transits.get(h, 0)
            pct = (transit / total * 100) if total > 0 else 0
            transit_percentages.append(pct)
        
        ax2.plot(hours, transit_percentages, marker='o', linewidth=2, markersize=6, color='darkred')
        ax2.fill_between(hours, transit_percentages, alpha=0.3, color='red')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Transit Percentage (%)')
        ax2.set_title('Transit Percentage by Hour of Day')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f'{self.region_of_interest}_temporal_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def get_temporal_analysis_data(self):
        """
        Extract temporal analysis data for further processing or export
        
        Returns:
        --------
        dict
            Dictionary containing temporal analysis data
        """
        if not self.results:
            print("❌ No analysis results available. Run analyze_transitions() first.")
            return None
        
        transitions_df = self.results['transitions_df']
        
        if transitions_df.empty:
            print("⚠️  No transitions data available for temporal analysis")
            return None
        
        # Convert start_time to datetime components
        transitions_df = transitions_df.copy()
        transitions_df['datetime'] = pd.to_datetime(transitions_df['start_time'], unit='s')
        transitions_df['hour'] = transitions_df['datetime'].dt.hour
        transitions_df['day_of_week'] = transitions_df['datetime'].dt.day_name()
        transitions_df['date_only'] = transitions_df['datetime'].dt.date
        
        # Hourly analysis
        hourly_data = []
        for hour in range(24):
            hour_transitions = transitions_df[transitions_df['hour'] == hour]
            
            total_visits = len(hour_transitions)
            transit_visits = len(hour_transitions[hour_transitions['is_transit']])
            work_visits = total_visits - transit_visits
            
            transit_percentage = (transit_visits / total_visits * 100) if total_visits > 0 else 0
            work_percentage = 100 - transit_percentage
            
            total_time = hour_transitions['duration'].sum()
            transit_time = hour_transitions[hour_transitions['is_transit']]['duration'].sum()
            work_time = total_time - transit_time
            
            avg_duration = hour_transitions['duration'].mean() if total_visits > 0 else 0
            avg_transit_duration = hour_transitions[hour_transitions['is_transit']]['duration'].mean() if transit_visits > 0 else 0
            avg_work_duration = hour_transitions[~hour_transitions['is_transit']]['duration'].mean() if work_visits > 0 else 0
            
            hourly_data.append({
                'hour': hour,
                'total_visits': total_visits,
                'transit_visits': transit_visits,
                'work_visits': work_visits,
                'transit_percentage': round(transit_percentage, 1),
                'work_percentage': round(work_percentage, 1),
                'total_time_seconds': total_time,
                'transit_time_seconds': transit_time,
                'work_time_seconds': work_time,
                'total_time_formatted': format_seconds_to_hms(total_time),
                'transit_time_formatted': format_seconds_to_hms(transit_time),
                'work_time_formatted': format_seconds_to_hms(work_time),
                'avg_duration_seconds': round(avg_duration, 1),
                'avg_transit_duration_seconds': round(avg_transit_duration, 1) if not pd.isna(avg_transit_duration) else 0,
                'avg_work_duration_seconds': round(avg_work_duration, 1) if not pd.isna(avg_work_duration) else 0,
                'avg_duration_formatted': format_seconds_to_hms(avg_duration),
                'avg_transit_duration_formatted': format_seconds_to_hms(avg_transit_duration) if not pd.isna(avg_transit_duration) else "00:00:00",
                'avg_work_duration_formatted': format_seconds_to_hms(avg_work_duration) if not pd.isna(avg_work_duration) else "00:00:00"
            })
        
        # Daily analysis
        daily_data = []
        for date in sorted(transitions_df['date_only'].unique()):
            day_transitions = transitions_df[transitions_df['date_only'] == date]
            
            total_visits = len(day_transitions)
            transit_visits = len(day_transitions[day_transitions['is_transit']])
            work_visits = total_visits - transit_visits
            
            transit_percentage = (transit_visits / total_visits * 100) if total_visits > 0 else 0
            
            total_time = day_transitions['duration'].sum()
            transit_time = day_transitions[day_transitions['is_transit']]['duration'].sum()
            work_time = total_time - transit_time
            
            daily_data.append({
                'date': str(date),
                'day_of_week': day_transitions['day_of_week'].iloc[0] if total_visits > 0 else '',
                'total_visits': total_visits,
                'transit_visits': transit_visits,
                'work_visits': work_visits,
                'transit_percentage': round(transit_percentage, 1),
                'total_time_seconds': total_time,
                'transit_time_seconds': transit_time,
                'work_time_seconds': work_time,
                'total_time_formatted': format_seconds_to_hms(total_time),
                'transit_time_formatted': format_seconds_to_hms(transit_time),
                'work_time_formatted': format_seconds_to_hms(work_time)
            })
        
        # Day of week analysis
        dow_data = []
        for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            dow_transitions = transitions_df[transitions_df['day_of_week'] == dow]
            
            total_visits = len(dow_transitions)
            transit_visits = len(dow_transitions[dow_transitions['is_transit']])
            work_visits = total_visits - transit_visits
            
            transit_percentage = (transit_visits / total_visits * 100) if total_visits > 0 else 0
            
            total_time = dow_transitions['duration'].sum()
            transit_time = dow_transitions[dow_transitions['is_transit']]['duration'].sum()
            work_time = total_time - transit_time
            
            dow_data.append({
                'day_of_week': dow,
                'total_visits': total_visits,
                'transit_visits': transit_visits,
                'work_visits': work_visits,
                'transit_percentage': round(transit_percentage, 1),
                'total_time_seconds': total_time,
                'transit_time_seconds': transit_time,
                'work_time_seconds': work_time,
                'total_time_formatted': format_seconds_to_hms(total_time),
                'transit_time_formatted': format_seconds_to_hms(transit_time),
                'work_time_formatted': format_seconds_to_hms(work_time)
            })
        
        # Shift analysis (if shift data is available)
        shift_data = []
        if 'shift' in transitions_df.columns:
            for shift in sorted(transitions_df['shift'].unique()):
                shift_transitions = transitions_df[transitions_df['shift'] == shift]
                
                total_visits = len(shift_transitions)
                transit_visits = len(shift_transitions[shift_transitions['is_transit']])
                work_visits = total_visits - transit_visits
                
                transit_percentage = (transit_visits / total_visits * 100) if total_visits > 0 else 0
                
                total_time = shift_transitions['duration'].sum()
                transit_time = shift_transitions[shift_transitions['is_transit']]['duration'].sum()
                work_time = total_time - transit_time
                
                shift_data.append({
                    'shift': shift,
                    'total_visits': total_visits,
                    'transit_visits': transit_visits,
                    'work_visits': work_visits,
                    'transit_percentage': round(transit_percentage, 1),
                    'total_time_seconds': total_time,
                    'transit_time_seconds': transit_time,
                    'work_time_seconds': work_time,
                    'total_time_formatted': format_seconds_to_hms(total_time),
                    'transit_time_formatted': format_seconds_to_hms(transit_time),
                    'work_time_formatted': format_seconds_to_hms(work_time)
                })
        
        # Peak analysis
        hourly_df = pd.DataFrame(hourly_data)
        peak_transit_hour = hourly_df.loc[hourly_df['transit_visits'].idxmax()] if not hourly_df.empty else None
        peak_work_hour = hourly_df.loc[hourly_df['work_visits'].idxmax()] if not hourly_df.empty else None
        highest_transit_percentage_hour = hourly_df.loc[hourly_df['transit_percentage'].idxmax()] if not hourly_df.empty else None
        
        return {
            'hourly_analysis': hourly_data,
            'daily_analysis': daily_data,
            'day_of_week_analysis': dow_data,
            'shift_analysis': shift_data,
            'peak_analysis': {
                'peak_transit_hour': peak_transit_hour.to_dict() if peak_transit_hour is not None else None,
                'peak_work_hour': peak_work_hour.to_dict() if peak_work_hour is not None else None,
                'highest_transit_percentage_hour': highest_transit_percentage_hour.to_dict() if highest_transit_percentage_hour is not None else None
            },
            'region_of_interest': self.region_of_interest,
            'transit_threshold': self.transit_threshold,
            'analysis_period': {
                'start_date': str(min(transitions_df['date_only'])) if not transitions_df.empty else None,
                'end_date': str(max(transitions_df['date_only'])) if not transitions_df.empty else None,
                'total_days': len(transitions_df['date_only'].unique()) if not transitions_df.empty else 0
            }
        }
    
    def export_detailed_results(self, output_dir):
        """Export detailed results to CSV files"""
        if not self.results:
            print("❌ No analysis results available. Run analyze_transitions() first.")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export all transitions
        transitions_df = self.results['transitions_df'].copy()
        transitions_df['duration_formatted'] = transitions_df['duration'].apply(format_seconds_to_hms)
        
        # Convert activities list to string for CSV export
        if 'activities' in transitions_df.columns:
            transitions_df['activities_str'] = transitions_df['activities'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        transitions_path = output_dir / f'{self.region_of_interest}_all_transitions.csv'
        # Select columns for export, handling the activities properly
        export_cols = ['employee_id', 'shift', 'date', 'start_time', 'end_time', 'duration', 'duration_formatted',
                      'previous_region', 'next_region', 'is_transit', 'transition_type', 'records_count']
        if 'activities_str' in transitions_df.columns:
            export_cols.append('activities_str')
        
        transitions_df[export_cols].to_csv(transitions_path, index=False)
        print(f"✅ Exported all transitions: {transitions_path}")
        
        # 2. Export transit-only data
        transit_df = transitions_df[transitions_df['is_transit']].copy()
        transit_path = output_dir / f'{self.region_of_interest}_transit_only.csv'
        transit_df[export_cols].to_csv(transit_path, index=False)
        print(f"✅ Exported transit data: {transit_path}")
        
        # 3. Export employee summary
        emp_summary = []
        for emp_id, stats in self.results['employee_stats'].items():
            emp_summary.append({
                'employee_id': emp_id,
                'total_visits': stats['total_visits'],
                'transit_visits': stats['total_transits'],
                'work_visits': stats['total_visits'] - stats['total_transits'],
                'transit_percentage': (stats['total_transits'] / stats['total_visits'] * 100) if stats['total_visits'] > 0 else 0,
                'total_transit_time': format_seconds_to_hms(stats['transit_time']),
                'total_work_time': format_seconds_to_hms(stats['work_time'])
            })
        
        emp_df = pd.DataFrame(emp_summary)
        emp_path = output_dir / f'{self.region_of_interest}_employee_summary.csv'
        emp_df.to_csv(emp_path, index=False)
        print(f"✅ Exported employee summary: {emp_path}")
        
        # 4. Export transition patterns
        patterns_data = []
        for pattern, count in self.results['summary_stats']['most_common_patterns']:
            patterns_data.append({
                'transition_pattern': pattern,
                'frequency': count
            })
        
        if patterns_data:
            patterns_df = pd.DataFrame(patterns_data)
            patterns_path = output_dir / f'{self.region_of_interest}_transition_patterns.csv'
            patterns_df.to_csv(patterns_path, index=False)
            print(f"✅ Exported transition patterns: {patterns_path}")
        
        # 5. Export temporal analysis data
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
            
            # Export peak analysis summary
            if temporal_data['peak_analysis']:
                peak_summary = []
                peak_data = temporal_data['peak_analysis']
                
                if peak_data['peak_transit_hour']:
                    peak_summary.append({
                        'metric': 'Peak Transit Hour',
                        'hour': peak_data['peak_transit_hour']['hour'],
                        'value': peak_data['peak_transit_hour']['transit_visits'],
                        'percentage': peak_data['peak_transit_hour']['transit_percentage'],
                        'description': f"Hour {peak_data['peak_transit_hour']['hour']} has the most transit visits ({peak_data['peak_transit_hour']['transit_visits']})"
                    })
                
                if peak_data['peak_work_hour']:
                    peak_summary.append({
                        'metric': 'Peak Work Hour',
                        'hour': peak_data['peak_work_hour']['hour'],
                        'value': peak_data['peak_work_hour']['work_visits'],
                        'percentage': peak_data['peak_work_hour']['work_percentage'],
                        'description': f"Hour {peak_data['peak_work_hour']['hour']} has the most work visits ({peak_data['peak_work_hour']['work_visits']})"
                    })
                
                if peak_data['highest_transit_percentage_hour']:
                    peak_summary.append({
                        'metric': 'Highest Transit Percentage Hour',
                        'hour': peak_data['highest_transit_percentage_hour']['hour'],
                        'value': peak_data['highest_transit_percentage_hour']['transit_visits'],
                        'percentage': peak_data['highest_transit_percentage_hour']['transit_percentage'],
                        'description': f"Hour {peak_data['highest_transit_percentage_hour']['hour']} has the highest transit percentage ({peak_data['highest_transit_percentage_hour']['transit_percentage']}%)"
                    })
                
                if peak_summary:
                    peak_df = pd.DataFrame(peak_summary)
                    peak_path = output_dir / f'{self.region_of_interest}_peak_analysis.csv'
                    peak_df.to_csv(peak_path, index=False)
                    print(f"✅ Exported peak analysis: {peak_path}")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Analyze transit patterns through a specific region')
    parser.add_argument('--region', type=str, required=True,
                       help='Region of interest to analyze (e.g., "8_Korridor")')
    parser.add_argument('--threshold', type=int, default=15,
                       help='Transit threshold in seconds (default: 15)')
    parser.add_argument('--data', type=str, default='data/raw/processed_sensor_data.csv',
                       help='Path to the sensor data CSV file')
    parser.add_argument('--output', type=str, default='region_analysis_output',
                       help='Output directory for results and visualizations')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip creating visualizations')
    parser.add_argument('--temporal-detail', action='store_true',
                       help='Show detailed temporal analysis in console output')
    
    args = parser.parse_args()
    
    print("🏭 REGION TRANSIT ANALYSIS TOOL")
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
    
    # Check if region exists in data
    available_regions = data['region'].unique()
    if args.region not in available_regions:
        print(f"❌ Region '{args.region}' not found in data.")
        print(f"📋 Available regions: {', '.join(sorted(available_regions))}")
        return 1
    
    # Initialize analyzer
    analyzer = RegionTransitAnalyzer(data, args.region, args.threshold)
    
    # Run analysis
    print(f"\n🔍 Starting transit analysis...")
    results = analyzer.analyze_transitions()
    
    # Print summary
    analyzer.print_summary()
    
    # Print detailed temporal analysis if requested
    if args.temporal_detail:
        analyzer.print_temporal_summary()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export detailed results
    print(f"\n💾 Exporting detailed results...")
    analyzer.export_detailed_results(output_dir)
    
    # Create visualizations
    if not args.no_viz:
        print(f"\n📊 Creating visualizations...")
        analyzer.create_visualizations(output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)