import pandas as pd
from pathlib import Path

def rename_regions(df):
    """
    Rename regions in the dataframe according to the specified mapping.
    Returns the modified dataframe and statistics about the changes.
    """
    # Region mapping dictionary
    region_mapping = {
        'Fettbacken': '1_Fettbacken',
        '1_Konditorei_station': '2a_Konditorei_station',
        '2_Konditorei_station': '2b_Konditorei_station',
        '3_Konditorei_station': '2c_Konditorei_station',
        'konditorei_deco': '3_Konditorei_deco_raum',
        '1_Brotstation': '4a_Brotstation',
        '2_Brotstation': '4b_Brotstation',
        '3_Brotstation': '4c_Brotstation',
        '4_Brotstation': '4d_Brotstation',
        'klein_kühlschrank_und_wasser': '4d_Brotstation',
        '1_Stickenofen': '5a_Stickenofen',
        '2_Stickenofen': '5b_Stickenofen',
        'Etagenofen': '5c_Etagenofen',
        'Gärraum': '5d_Etagenofen',
        'Mehlmaschine': '6a_Mehl_Brötchen_Maschine',
        'Rollenhandtuchspend': '6c_Mehl_Brötchen_Maschine',
        'brotkisten_regal': '7a_Brotwagon_und_kisten',
        'production_plan': '7a_Brotwagon_und_kisten',
        'leere_Brotwagen': '7b_Brotwagon_und_kisten',
        'leere_Kisten': '7c_Brotwagon_und_kisten',
        'Brotwagon_regal_haltebereich': '7d_Brotwagon_und_kisten',
        'bereitstellen_prepared_goods': '7e_Brotwagon_und_kisten',
        'loading_area': '7f_Brotwagon_und_kisten',
        '1_corridor': '8_Korridor',
        'cookies_packing_hand': '9_Kekse_Verpackung'
    }
    
    # Get original region counts
    original_counts = df['region'].value_counts().to_dict()
    
    # Apply the renaming
    df['region'] = df['region'].map(region_mapping).fillna(df['region'])
    
    # Get new region counts
    new_counts = df['region'].value_counts().to_dict()
    
    # Calculate statistics
    total_rows = len(df)
    changed_rows = sum(count for region, count in original_counts.items() 
                      if region in region_mapping)
    
    # Generate report
    report = []
    report.append("=== Region Renaming Summary ===")
    report.append(f"Total rows processed: {total_rows}")
    report.append(f"Rows with renamed regions: {changed_rows}")
    
    # Find unchanged regions
    unchanged = set(original_counts.keys()) - set(region_mapping.keys())
    if unchanged:
        report.append("\nRegions that were not renamed:")
        for region in sorted(unchanged):
            report.append(f"- {region} ({original_counts[region]} rows)")
    
    # Show mapping results
    report.append("\nRegion mapping results:")
    for old_name, new_name in region_mapping.items():
        if old_name in original_counts:
            report.append(f"- {old_name} -> {new_name} ({original_counts[old_name]} rows)")
    
    return df, "\n".join(report)

def main():
    # Define paths
    input_file = Path("data/raw/processed_sensor_data.csv")
    output_file = Path("data/processed/renamed_sensor_data.csv")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read the data
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Rename regions
        print("Renaming regions...")
        df_renamed, report = rename_regions(df)
        
        # Save the results
        print(f"Saving results to {output_file}...")
        df_renamed.to_csv(output_file, index=False)
        
        # Print the report
        print("\n" + "="*50)
        print(report)
        print("="*50)
        print("\nProcessing complete!")
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found!")
        return 1
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
