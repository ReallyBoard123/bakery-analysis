import pandas as pd

def process_bakery_data(input_file, output_file):
    # Load the data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Print initial shape
    initial_shape = df.shape
    print(f"Initial dataframe shape: {initial_shape}")
    
    # 1. Remove data from shift 6
    df = df[df['shift'] != 6]
    print(f"After removing shift 6: {df.shape}")
    
    # 2. Remove rows where isPauseData = True
    df = df[df['isPauseData'] == False]
    print(f"After removing pause data: {df.shape}")
    
    # 3. Remove specified columns
    columns_to_drop = ['startTimeClock', 'endTimeClock', 'source_file', 'isPauseData']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"After removing specified columns: {df.shape}")
    
    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    # Print summary of changes
    print(f"\nSummary:")
    print(f"Initial row count: {initial_shape[0]}")
    print(f"Final row count: {df.shape[0]}")
    print(f"Rows removed: {initial_shape[0] - df.shape[0]}")
    print(f"Initial column count: {initial_shape[1]}")
    print(f"Final column count: {df.shape[1]}")

if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "combined_sensor_data.csv"
    output_file = "processed_sensor_data.csv"
    
    process_bakery_data(input_file, output_file)