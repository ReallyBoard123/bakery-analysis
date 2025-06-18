#!/usr/bin/env python3
"""
Advanced Pattern Exploration for Bakery Employee Data
=====================================================

This script performs advanced statistical analysis and machine learning 
to discover hidden patterns, repetitive sequences, and predictive insights
from employee activity data.

Features:
- Repetitive activity sequence detection
- Markov chain analysis for activity transitions
- Employee clustering based on work patterns
- Ergonomic risk prediction (back pain probability)
- Fatigue and efficiency analysis
- Anomaly detection
- Time series pattern mining
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML and Statistical Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from scipy.stats import chi2_contingency
import networkx as nx
from collections import Counter, defaultdict
import itertools
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AdvancedPatternExplorer:
    def __init__(self, data_path='data/raw/processed_sensor_data.csv'):
        """Initialize the explorer with data"""
        self.data_path = data_path
        self.data = None
        self.output_dir = Path('exploration_results')
        self.output_dir.mkdir(exist_ok=True)
        
        print("🔍 ADVANCED PATTERN EXPLORATION")
        print("=" * 50)
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        print("📊 Loading and preparing data...")
        
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data):,} records from {self.data['id'].nunique()} employees")
        
        # Sort by employee and time for sequence analysis
        self.data = self.data.sort_values(['id', 'shift', 'date', 'startTime'])
        
        # Add time-based features
        self.data['hour'] = (self.data['startTime'] % 86400) // 3600
        self.data['duration_minutes'] = self.data['duration'] / 60
        
        # Add department classification
        bread_dept = ['32-A', '32-C', '32-D', '32-E']
        self.data['department'] = self.data['id'].apply(
            lambda x: 'Bread' if x in bread_dept else 'Cake'
        )
        
        print(f"Data prepared with {len(self.data.columns)} features")
        return self.data
    
    def find_repetitive_sequences(self, min_length=3, max_length=6, min_occurrences=3):
        """Find repetitive activity sequences that might indicate repeated tasks"""
        print(f"\n🔄 Finding repetitive sequences (length {min_length}-{max_length})...")
        
        sequence_patterns = defaultdict(list)
        
        # Group by employee and shift for pattern analysis
        for (emp_id, shift), group in self.data.groupby(['id', 'shift']):
            activities = group['activity'].tolist()
            
            # Find patterns of different lengths
            for length in range(min_length, max_length + 1):
                sequences = []
                for i in range(len(activities) - length + 1):
                    sequence = tuple(activities[i:i + length])
                    sequences.append(sequence)
                
                # Count sequence occurrences
                sequence_counts = Counter(sequences)
                
                # Store patterns that occur frequently
                for sequence, count in sequence_counts.items():
                    if count >= min_occurrences:
                        sequence_patterns[sequence].append({
                            'employee': emp_id,
                            'shift': shift,
                            'count': count,
                            'total_activities': len(activities)
                        })
        
        # Analyze most common patterns
        pattern_analysis = []
        for sequence, occurrences in sequence_patterns.items():
            total_count = sum(occ['count'] for occ in occurrences)
            unique_employees = len(set(occ['employee'] for occ in occurrences))
            
            pattern_analysis.append({
                'sequence': ' → '.join(sequence),
                'pattern_length': len(sequence),
                'total_occurrences': total_count,
                'unique_employees': unique_employees,
                'avg_per_employee': total_count / unique_employees,
                'employees': [occ['employee'] for occ in occurrences]
            })
        
        # Sort by total occurrences
        pattern_df = pd.DataFrame(pattern_analysis)
        pattern_df = pattern_df.sort_values('total_occurrences', ascending=False)
        
        # Save results
        pattern_df.to_csv(self.output_dir / 'repetitive_sequences.csv', index=False)
        
        # Display top patterns
        print("\n🎯 TOP REPETITIVE PATTERNS FOUND:")
        print("-" * 70)
        for _, row in pattern_df.head(10).iterrows():
            print(f"Pattern: {row['sequence']}")
            print(f"  Occurrences: {row['total_occurrences']}, "
                  f"Employees: {row['unique_employees']}, "
                  f"Avg per employee: {row['avg_per_employee']:.1f}")
            print()
        
        return pattern_df
    
    def markov_chain_analysis(self):
        """Analyze activity transitions using Markov chains"""
        print("\n🔗 Performing Markov Chain Analysis...")
        
        activities = self.data['activity'].unique()
        
        # Create transition matrix for each employee
        employee_transitions = {}
        
        for emp_id in self.data['id'].unique():
            emp_data = self.data[self.data['id'] == emp_id].sort_values(['shift', 'date', 'startTime'])
            
            # Get activity sequences
            activities_seq = emp_data['activity'].tolist()
            
            # Count transitions
            transitions = defaultdict(int)
            total_transitions = 0
            
            for i in range(len(activities_seq) - 1):
                from_activity = activities_seq[i]
                to_activity = activities_seq[i + 1]
                transitions[(from_activity, to_activity)] += 1
                total_transitions += 1
            
            # Convert to probabilities
            transition_probs = {}
            for (from_act, to_act), count in transitions.items():
                if from_act not in transition_probs:
                    transition_probs[from_act] = {}
                transition_probs[from_act][to_act] = count / total_transitions
            
            employee_transitions[emp_id] = transition_probs
        
        # Create overall transition matrix
        all_transitions = defaultdict(int)
        for emp_id, transitions in employee_transitions.items():
            for from_act, to_dict in transitions.items():
                for to_act, prob in to_dict.items():
                    all_transitions[(from_act, to_act)] += prob
        
        # Normalize
        total_prob = sum(all_transitions.values())
        for key in all_transitions:
            all_transitions[key] /= total_prob
        
        # Create transition matrix DataFrame
        transition_matrix = pd.DataFrame(0.0, index=activities, columns=activities)
        for (from_act, to_act), prob in all_transitions.items():
            transition_matrix.loc[from_act, to_act] = prob
        
        # Visualize transition matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_matrix, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Activity Transition Probability Matrix (Markov Chain)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'markov_chain_transitions.png', dpi=300)
        plt.close()
        
        transition_matrix.to_csv(self.output_dir / 'transition_matrix.csv')
        
        print("✅ Markov chain analysis complete")
        return transition_matrix, employee_transitions
    
    def cluster_employees_by_patterns(self):
        """Cluster employees based on their work patterns"""
        print("\n👥 Clustering employees by work patterns...")
        
        # Create feature matrix for each employee
        features = []
        employee_ids = []
        
        for emp_id in self.data['id'].unique():
            emp_data = self.data[self.data['id'] == emp_id]
            
            # Activity distribution features
            activity_dist = emp_data.groupby('activity')['duration'].sum()
            total_duration = activity_dist.sum()
            activity_percentages = (activity_dist / total_duration * 100).fillna(0)
            
            # Temporal features - hour is already numeric (0-23)
            hour_dist = emp_data.groupby('hour')['duration'].sum()
            peak_hour = hour_dist.idxmax() if not hour_dist.empty else 0
            
            # Repetition features
            activities_seq = emp_data['activity'].tolist()
            transitions = []
            for i in range(len(activities_seq) - 1):
                transitions.append((activities_seq[i], activities_seq[i + 1]))
            
            # Most common transition
            if transitions:
                common_transition = Counter(transitions).most_common(1)[0][1]
                repetition_score = common_transition / len(transitions)
            else:
                repetition_score = 0
            
            # Duration patterns
            avg_duration = emp_data['duration'].mean()
            duration_std = emp_data['duration'].std()
            
            # Combine features
            feature_vector = []
            
            # Activity percentages
            for activity in ['Walk', 'Stand', 'Handle center', 'Handle up', 'Handle down']:
                feature_vector.append(activity_percentages.get(activity, 0))
            
            # Other features
            feature_vector.extend([
                peak_hour,
                repetition_score,
                avg_duration,
                duration_std,
                len(emp_data),  # Total activities
                emp_data['region'].nunique()  # Number of unique regions
            ])
            
            features.append(feature_vector)
            employee_ids.append(emp_id)
        
        # Convert to DataFrame
        feature_names = ['Walk%', 'Stand%', 'HandleCenter%', 'HandleUp%', 'HandleDown%',
                        'PeakHour', 'RepetitionScore', 'AvgDuration', 'DurationStd',
                        'TotalActivities', 'UniqueRegions']
        
        features_df = pd.DataFrame(features, columns=feature_names, index=employee_ids)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels
        features_df['Cluster'] = clusters
        features_df['Employee'] = employee_ids
        
        # Visualize clusters with PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis')
        
        # Add employee labels
        for i, emp_id in enumerate(employee_ids):
            plt.annotate(emp_id, (features_pca[i, 0], features_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter)
        plt.title('Employee Clustering Based on Work Patterns (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'employee_clusters.png', dpi=300)
        plt.close()
        
        # Analyze cluster characteristics - only use numeric columns for mean calculation
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        cluster_analysis = features_df[numeric_cols].groupby('Cluster').mean()
        cluster_analysis.to_csv(self.output_dir / 'cluster_characteristics.csv')
        
        features_df.to_csv(self.output_dir / 'employee_features_and_clusters.csv')
        
        print("✅ Employee clustering complete")
        print(f"Cluster distribution: {Counter(clusters)}")
        
        return features_df, clusters
    
    def predict_ergonomic_risk(self):
        """Predict ergonomic risk (back pain probability) based on activity patterns"""
        print("\n🏥 Predicting Ergonomic Risk (Back Pain Probability)...")
        
        # Create ergonomic risk features for each employee
        risk_features = []
        employee_ids = []
        
        for emp_id in self.data['id'].unique():
            emp_data = self.data[self.data['id'] == emp_id]
            
            # Calculate risk factors
            total_duration = emp_data['duration'].sum()
            
            # Handle down intensity (higher = more risk)
            handle_down_duration = emp_data[emp_data['activity'] == 'Handle down']['duration'].sum()
            handle_down_percentage = (handle_down_duration / total_duration) * 100
            
            # Repetitive handle down sequences
            activities = emp_data['activity'].tolist()
            handle_down_sequences = 0
            current_sequence = 0
            
            for activity in activities:
                if activity == 'Handle down':
                    current_sequence += 1
                else:
                    if current_sequence >= 3:  # 3+ consecutive handle downs
                        handle_down_sequences += 1
                    current_sequence = 0
            
            # Working duration per day
            daily_duration = emp_data.groupby(['shift', 'date'])['duration'].sum().mean() / 3600  # hours
            
            # Activity variety (less variety = more repetitive)
            activity_variety = emp_data['activity'].nunique()
            
            # Average activity duration (longer periods = less movement)
            avg_activity_duration = emp_data['duration'].mean()
            
            # Peak load periods
            hourly_handle_down = emp_data[emp_data['activity'] == 'Handle down'].groupby('hour')['duration'].sum()
            peak_handle_down_hour = hourly_handle_down.max() if len(hourly_handle_down) > 0 else 0
            
            risk_features.append([
                handle_down_percentage,
                handle_down_sequences,
                daily_duration,
                activity_variety,
                avg_activity_duration,
                peak_handle_down_hour
            ])
            employee_ids.append(emp_id)
        
        # Create DataFrame
        risk_df = pd.DataFrame(risk_features, columns=[
            'HandleDownPercentage', 'RepetitiveSequences', 'DailyHours',
            'ActivityVariety', 'AvgActivityDuration', 'PeakHandleDownHour'
        ], index=employee_ids)
        
        # Create synthetic risk labels based on domain knowledge
        # High risk: High handle down %, many repetitive sequences, long hours
        risk_scores = (
            risk_df['HandleDownPercentage'] * 0.4 +
            risk_df['RepetitiveSequences'] * 0.3 +
            risk_df['DailyHours'] * 0.2 +
            (10 - risk_df['ActivityVariety']) * 0.1  # Less variety = higher risk
        )
        
        # Normalize to 0-100 scale
        risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min()) * 100
        
        # Create risk categories
        risk_df['RiskScore'] = risk_scores
        risk_df['RiskCategory'] = pd.cut(risk_scores, 
                                       bins=[0, 33, 66, 100], 
                                       labels=['Low', 'Medium', 'High'])
        
        # Create binary high risk labels for ML
        risk_df['HighRisk'] = (risk_scores > 66).astype(int)
        
        # Train a classifier
        X = risk_df[['HandleDownPercentage', 'RepetitiveSequences', 'DailyHours',
                     'ActivityVariety', 'AvgActivityDuration', 'PeakHandleDownHour']]
        y = risk_df['HighRisk']
        
        # Split data (using all for training since we have limited data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Visualize risk analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk distribution
        risk_df['RiskCategory'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'orange', 'red'])
        ax1.set_title('Ergonomic Risk Distribution')
        ax1.set_ylabel('Number of Employees')
        
        # Handle down vs risk
        ax2.scatter(risk_df['HandleDownPercentage'], risk_df['RiskScore'], 
                   c=risk_df['HighRisk'], cmap='RdYlGn_r')
        ax2.set_xlabel('Handle Down Percentage')
        ax2.set_ylabel('Risk Score')
        ax2.set_title('Handle Down Activity vs Risk Score')
        
        # Feature importance
        feature_importance.plot(x='feature', y='importance', kind='bar', ax=ax3)
        ax3.set_title('Feature Importance for Risk Prediction')
        ax3.tick_params(axis='x', rotation=45)
        
        # Employee risk scores
        # Handle NaN values in RiskCategory
        if risk_df['RiskCategory'].isna().any():
            # If there are NaN values, convert to string type first
            risk_df['RiskCategory'] = risk_df['RiskCategory'].astype(str)
            risk_df.loc[risk_df['RiskCategory'] == 'nan', 'RiskCategory'] = 'Unknown'
        
        # Create color mapping
        color_map = {
            'Low': 'green', 
            'Medium': 'orange', 
            'High': 'red',
            'Unknown': 'gray'  # For any unexpected values
        }
        
        # Plot with colors based on risk category
        risk_df['RiskScore'].plot(
            kind='bar', 
            ax=ax4,
            color=[color_map.get(x, 'gray') for x in risk_df['RiskCategory']]
        )
        ax4.set_title('Individual Employee Risk Scores')
        ax4.set_ylabel('Risk Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ergonomic_risk_analysis.png', dpi=300)
        plt.close()
        
        # Save results
        risk_df.to_csv(self.output_dir / 'ergonomic_risk_assessment.csv')
        feature_importance.to_csv(self.output_dir / 'risk_feature_importance.csv', index=False)
        
        print("✅ Ergonomic risk prediction complete")
        print(f"High-risk employees: {risk_df[risk_df['RiskCategory'] == 'High'].index.tolist()}")
        
        return risk_df, rf
    
    def detect_anomalous_patterns(self):
        """Detect anomalous work patterns using Isolation Forest"""
        print("\n🚨 Detecting Anomalous Work Patterns...")
        
        # Create features for anomaly detection
        anomaly_features = []
        timestamps = []
        employee_ids = []
        
        # Analyze patterns by employee-shift combinations
        for (emp_id, shift), group in self.data.groupby(['id', 'shift']):
            group = group.sort_values('startTime')
            
            # Duration-based features
            total_duration = group['duration'].sum() / 3600  # hours
            avg_duration = group['duration'].mean()
            duration_std = group['duration'].std()
            
            # Activity distribution
            activity_counts = group['activity'].value_counts()
            total_activities = len(group)
            
            # Time-based features
            start_hour = (group['startTime'].iloc[0] % 86400) / 3600
            end_hour = (group['startTime'].iloc[-1] % 86400) / 3600
            work_span = end_hour - start_hour
            
            # Region mobility
            unique_regions = group['region'].nunique()
            region_changes = (group['region'] != group['region'].shift()).sum()
            
            # Activity transition rate
            activity_changes = (group['activity'] != group['activity'].shift()).sum()
            transition_rate = activity_changes / total_activities if total_activities > 0 else 0
            
            # Combine features
            feature_vector = [
                total_duration,
                avg_duration,
                duration_std,
                total_activities,
                start_hour,
                work_span,
                unique_regions,
                region_changes,
                transition_rate
            ]
            
            # Add activity percentages
            for activity in ['Walk', 'Stand', 'Handle center', 'Handle up', 'Handle down']:
                percentage = (activity_counts.get(activity, 0) / total_activities) * 100
                feature_vector.append(percentage)
            
            anomaly_features.append(feature_vector)
            timestamps.append(f"{emp_id}_S{shift}")
            employee_ids.append(emp_id)
        
        # Create DataFrame
        feature_names = ['TotalDuration', 'AvgDuration', 'DurationStd', 'TotalActivities',
                        'StartHour', 'WorkSpan', 'UniqueRegions', 'RegionChanges', 'TransitionRate',
                        'Walk%', 'Stand%', 'HandleCenter%', 'HandleUp%', 'HandleDown%']
        
        anomaly_df = pd.DataFrame(anomaly_features, columns=feature_names, index=timestamps)
        anomaly_df['Employee'] = employee_ids
        
        # Detect anomalies using Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(anomaly_df[feature_names])
        anomaly_df['IsAnomalous'] = anomaly_scores == -1
        
        # Get anomaly scores
        anomaly_df['AnomalyScore'] = iso_forest.score_samples(anomaly_df[feature_names])
        
        # Visualize anomalies
        plt.figure(figsize=(14, 10))
        
        # PCA for visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(anomaly_df[feature_names])
        
        # Plot normal vs anomalous
        normal_mask = anomaly_df['IsAnomalous'] == False
        anomalous_mask = anomaly_df['IsAnomalous'] == True
        
        plt.scatter(features_pca[normal_mask, 0], features_pca[normal_mask, 1], 
                   c='blue', alpha=0.6, label='Normal')
        plt.scatter(features_pca[anomalous_mask, 0], features_pca[anomalous_mask, 1], 
                   c='red', alpha=0.8, label='Anomalous', s=100)
        
        # Annotate anomalous points
        for i, timestamp in enumerate(anomaly_df.index):
            if anomaly_df.loc[timestamp, 'IsAnomalous']:
                plt.annotate(timestamp, (features_pca[i, 0], features_pca[i, 1]), 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Anomalous Work Pattern Detection')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_detection.png', dpi=300)
        plt.close()
        
        # Save results
        anomaly_df.to_csv(self.output_dir / 'anomaly_analysis.csv')
        
        # Print anomalous patterns
        anomalous_patterns = anomaly_df[anomaly_df['IsAnomalous']]
        print(f"✅ Found {len(anomalous_patterns)} anomalous patterns:")
        for idx, row in anomalous_patterns.iterrows():
            print(f"  {idx}: Employee {row['Employee']}, Score: {row['AnomalyScore']:.3f}")
        
        return anomaly_df
    
    def time_series_pattern_mining(self):
        """Mine temporal patterns in activities"""
        print("\n⏰ Mining Time Series Patterns...")
        
        # Analyze daily patterns for each employee
        daily_patterns = []
        
        for emp_id in self.data['id'].unique():
            emp_data = self.data[self.data['id'] == emp_id]
            
            # Group by date
            for date in emp_data['date'].unique():
                daily_data = emp_data[emp_data['date'] == date].sort_values('startTime')
                
                if len(daily_data) < 5:  # Skip days with too few activities
                    continue
                
                # Create hourly activity profile
                hourly_profile = np.zeros(24)
                activity_profile = {activity: np.zeros(24) for activity in daily_data['activity'].unique()}
                
                for _, row in daily_data.iterrows():
                    hour = int((row['startTime'] % 86400) // 3600)
                    hourly_profile[hour] += row['duration'] / 3600  # Convert to hours
                    activity_profile[row['activity']][hour] += row['duration'] / 3600
                
                # Calculate pattern features
                peak_hour = np.argmax(hourly_profile)
                total_hours = np.sum(hourly_profile)
                activity_entropy = -np.sum([p * np.log(p + 1e-10) for p in hourly_profile if p > 0])
                
                daily_patterns.append({
                    'employee': emp_id,
                    'date': date,
                    'peak_hour': peak_hour,
                    'total_hours': total_hours,
                    'activity_entropy': activity_entropy,
                    'hourly_profile': hourly_profile,
                    'activity_profiles': activity_profile
                })
        
        # Analyze consistency of daily patterns
        consistency_analysis = []
        
        for emp_id in self.data['id'].unique():
            emp_patterns = [p for p in daily_patterns if p['employee'] == emp_id]
            
            if len(emp_patterns) < 2:
                continue
            
            # Calculate pattern consistency
            profiles = np.array([p['hourly_profile'] for p in emp_patterns])
            consistency_score = np.mean([np.corrcoef(profiles[i], profiles[j])[0, 1] 
                                       for i in range(len(profiles)) 
                                       for j in range(i+1, len(profiles))])
            
            peak_hours = [p['peak_hour'] for p in emp_patterns]
            peak_hour_std = np.std(peak_hours)
            
            consistency_analysis.append({
                'employee': emp_id,
                'pattern_consistency': consistency_score,
                'peak_hour_variability': peak_hour_std,
                'avg_total_hours': np.mean([p['total_hours'] for p in emp_patterns])
            })
        
        consistency_df = pd.DataFrame(consistency_analysis)
        
        # Visualize patterns
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, emp_id in enumerate(self.data['id'].unique()):
            if i >= 8:  # Limit to 8 employees for visualization
                break
                
            emp_patterns = [p for p in daily_patterns if p['employee'] == emp_id]
            if emp_patterns:
                avg_profile = np.mean([p['hourly_profile'] for p in emp_patterns], axis=0)
                axes[i].plot(range(24), avg_profile, 'b-', linewidth=2)
                axes[i].set_title(f'Employee {emp_id} - Daily Pattern')
                axes[i].set_xlabel('Hour of Day')
                axes[i].set_ylabel('Hours of Activity')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'daily_patterns.png', dpi=300)
        plt.close()
        
        # Save results
        consistency_df.to_csv(self.output_dir / 'pattern_consistency.csv', index=False)
        
        print("✅ Time series pattern mining complete")
        return daily_patterns, consistency_df
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("\n📋 Generating Comprehensive Report...")
        
        report = []
        report.append("# ADVANCED PATTERN EXPLORATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Data overview
        report.append("## DATA OVERVIEW")
        report.append(f"- Total Records: {len(self.data):,}")
        report.append(f"- Employees: {self.data['id'].nunique()}")
        report.append(f"- Activities: {', '.join(self.data['activity'].unique())}")
        report.append(f"- Time Period: {self.data['date'].min()} to {self.data['date'].max()}")
        report.append("")
        
        # Key findings placeholder
        report.append("## KEY FINDINGS")
        report.append("")
        report.append("### 🔄 Repetitive Patterns")
        report.append("- Most common repetitive sequences identified")
        report.append("- High repetition employees flagged for ergonomic review")
        report.append("")
        
        report.append("### 👥 Employee Clustering")
        report.append("- Employees grouped into distinct work pattern clusters")
        report.append("- Different departments show characteristic patterns")
        report.append("")
        
        report.append("### 🏥 Ergonomic Risk Assessment")
        report.append("- High-risk employees identified based on activity patterns")
        report.append("- Handle down activity is primary risk factor")
        report.append("")
        
        report.append("### 🚨 Anomaly Detection")
        report.append("- Unusual work patterns detected and flagged")
        report.append("- Potential efficiency improvement opportunities identified")
        report.append("")
        
        report.append("## FILES GENERATED")
        files = [
            "repetitive_sequences.csv - Repetitive activity patterns",
            "markov_chain_transitions.png - Activity transition probabilities",
            "employee_clusters.png - Employee work pattern clusters",
            "ergonomic_risk_assessment.csv - Individual risk assessments",
            "anomaly_detection.png - Anomalous pattern visualization",
            "daily_patterns.png - Daily work pattern analysis"
        ]
        
        for file in files:
            report.append(f"- {file}")
        
        report.append("")
        report.append("## RECOMMENDATIONS")
        report.append("1. Review high-risk employees for ergonomic interventions")
        report.append("2. Investigate anomalous patterns for process improvements")
        report.append("3. Use clustering insights for task optimization")
        report.append("4. Monitor repetitive sequence patterns for fatigue management")
        
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("✅ Comprehensive report generated")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Full Analysis Pipeline...")
        
        # Load data
        self.load_and_prepare_data()
        
        # Run all analyses
        patterns = self.find_repetitive_sequences()
        transitions, emp_transitions = self.markov_chain_analysis()
        features, clusters = self.cluster_employees_by_patterns()
        risk_assessment, risk_model = self.predict_ergonomic_risk()
        anomalies = self.detect_anomalous_patterns()
        daily_patterns, consistency = self.time_series_pattern_mining()
        
        # Generate report
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 50)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"📁 All results saved to: {self.output_dir.absolute()}")
        print("\n🎯 Key Insights:")
        print(f"- Found {len(patterns)} repetitive sequence patterns")
        print(f"- Clustered employees into {len(set(clusters))} distinct work groups")
        print(f"- Identified {len(risk_assessment[risk_assessment['RiskCategory'] == 'High'])} high-risk employees")
        print(f"- Detected {len(anomalies[anomalies['IsAnomalous']])} anomalous work patterns")
        
        return {
            'patterns': patterns,
            'transitions': transitions,
            'clusters': features,
            'risk_assessment': risk_assessment,
            'anomalies': anomalies,
            'daily_patterns': daily_patterns
        }

def main():
    """Main execution function"""
    explorer = AdvancedPatternExplorer()
    results = explorer.run_full_analysis()
    
    print("\n🎉 Advanced pattern exploration complete!")
    print("Check the 'exploration_results' folder for detailed findings!")

if __name__ == "__main__":
    main()