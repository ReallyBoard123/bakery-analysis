import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta

def analyze_cart_idle_simple():
    """
    Analyze cart idle times from the cart_data folder.
    Uses environment variables for input/output paths.
    """
    # Get input and output paths from environment variables with fallbacks
    input_folder = os.environ.get('CART_DATA_DIR', 'output/cart_data')
    output_dir = os.environ.get('IDLE_ANALYSIS_DIR', 'output/cart_analysis')
    
    input_path = Path(input_folder)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Reading cart data from: {input_path.absolute()}")
    print(f"📊 Saving idle analysis to: {output_path.absolute()}")
    if not input_path.exists():
        print(f"Error: Directory {input_path} does not exist")
        return None
    
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return None
    
    all_idle_periods = []
    
    for file_path in csv_files:
        print(f"Processing {file_path.name}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Ensure column names are clean and consistent
            df.columns = df.columns.str.strip()
            
            # Convert numeric columns to float
            numeric_cols = ['startTime', 'endTime', 'duration']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Process each cart ID in the file
            for cart_id, cart_group in df.groupby('cartId'):
                # Sort by start time
                cart_group = cart_group.sort_values('startTime')
                
                # Process each shift and date combination
                for (shift, date), shift_group in cart_group.groupby(['shift', 'date']):
                    shift_group = shift_group.sort_values('startTime')
                    
                    # Find gaps between consecutive activities in the same region
                    for i in range(len(shift_group) - 1):
                        current = shift_group.iloc[i]
                        next_row = shift_group.iloc[i + 1]
                        
                        # Check if same region and there's a gap
                        if (current['region'] == next_row['region'] and 
                            next_row['startTime'] > current['endTime']):
                            
                            idle_duration = next_row['startTime'] - current['endTime']
                            
                            # Only consider gaps > 1 second
                            if idle_duration > 1:
                                idle_record = {
                                    'cartId': cart_id,
                                    'last_person_id': current['current_person_id'],
                                    'shift': shift,
                                    'date': date,
                                    'startTime': current['endTime'],
                                    'endTime': next_row['startTime'],
                                    'duration': round(idle_duration, 3),  # Round to 3 decimal places
                                    'region': current['region'],
                                    'activity_before': current['activity'],
                                    'activity_after': next_row['activity'],
                                    'source_file': file_path.name
                                }
                                all_idle_periods.append(idle_record)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create and save DataFrame
    if all_idle_periods:
        result_df = pd.DataFrame(all_idle_periods)
        
        # Calculate duration in minutes and hours for better readability
        result_df['duration_min'] = (result_df['duration'] / 60).round(2)
        result_df['duration_hrs'] = (result_df['duration'] / 3600).round(4)
        
        # Reorder columns
        cols = ['cartId', 'shift', 'date', 'startTime', 'endTime', 'duration', 
                'duration_min', 'duration_hrs', 'region', 'last_person_id',
                'activity_before', 'activity_after', 'source_file']
        result_df = result_df[cols]
        
        # Sort by cart ID and start time
        result_df = result_df.sort_values(['cartId', 'startTime'])
        
        # Save to CSV
        output_file = output_path / "cart_idle_periods.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved idle periods to {output_file.absolute()}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total idle periods found: {len(result_df)}")
        print(f"Total idle time: {result_df['duration_hrs'].sum():.2f} hours")
        print(f"Average idle time: {result_df['duration_min'].mean():.2f} minutes")
        
        # Print top regions by idle time
        print("\nTop regions by idle time:")
        region_stats = result_df.groupby('region').agg({
            'duration_hrs': 'sum',
            'duration': 'count'
        }).sort_values('duration_hrs', ascending=False).head(10)
        region_stats.columns = ['total_hours', 'count']
        region_stats['avg_minutes'] = (region_stats['total_hours'] * 60 / region_stats['count']).round(1)
        print(region_stats)
        
        return result_df
    else:
        print("No idle periods found.")
        return None

# Usage
if __name__ == "__main__":
    # Run with default paths from environment variables or fallbacks
    result = analyze_cart_idle_simple()
    if result is not None:
        print("\n✅ Idle periods analysis complete!")
    else:
        print("\n❌ Idle periods analysis failed.")