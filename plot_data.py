import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define colors for specific regions
REGION_COLORS = {
    'Fettbacken': '#6BAED6',            # Light blue
    'Konditorei_station': '#3182BD',    # Blue
    'konditorei_deco': '#08519C',       # Dark blue          # Green
    '1_Brotstation': 'lawngreen',         # Lighter green
    '2_Brotstation': 'forestgreen',         # Even darker green
    '3_Brotstation': 'darkgreen',         # Much darker green
    '4_Brotstation': 'mediumspringgreen',         # Very dark green
    'Mehlmaschine': '#4A8A50',          # Darker light green
    'brotkisten_regal': '#FFFF00',      # Yellow
    '1_Konditorei_station': 'powderblue',  # Light blue
    '2_Konditorei_station': 'deepskyblue',  # Medium blue
    '3_Konditorei_station': 'steelblue',  # Darker blue
    'bereitstellen_prepared_goods': '#FFD700',  # Gold
    '1_corridor': '#FFC107',                   # Amber
    'leere_Kisten': '#FFB300',                 # Darker yellow
}

# Define activity types to process
ACTIVITY_TYPES = ['Walk', 'Stand', 'Handle center', 'Handle up', 'Handle down']

def generate_activity_barplots(base_folder):
    for activity in ACTIVITY_TYPES:
        # Construct the path to the CSV file for the current activity
        csv_file = os.path.join(base_folder, f"{activity.lower().replace(' ', '_')}_region_stats.csv")

        # Check if the file exists
        if not os.path.exists(csv_file):
            print(f"Error: The file '{csv_file}' does not exist.")
            continue

        # Read the CSV file
        data = pd.read_csv(csv_file)

        # Get unique employee IDs
        employee_ids = data['id'].unique()

        for emp_id in employee_ids:
            # Filter data for the current employee ID
            emp_data = data[data['id'] == emp_id]

            # Plot the bar graph
            plt.figure(figsize=(20, 12), dpi=600)  # Double the size and DPI for higher resolution
            colors = [REGION_COLORS.get(region, '#CCCCCC') for region in emp_data['region']]
            plt.bar(emp_data['time_category'], emp_data['total_minutes'], color=colors)
            plt.xlabel('Time Category')
            plt.ylabel('Total Minutes')
            plt.title(f'{activity} Duration for Employee {emp_id}')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Explicitly set larger font sizes for all text elements
            plt.rcParams.update({
                'axes.titlesize': 24,  # Title font size
                'axes.labelsize': 22,  # X and Y label font size
                'xtick.labelsize': 20,  # X tick label font size
                'ytick.labelsize': 20,  # Y tick label font size
                'legend.fontsize': 18,  # Legend font size
                'font.size': 20  # General font size
            })

            # Add legend for regions
            unique_regions = emp_data['region'].unique()
            legend_patches = [mpatches.Patch(color=REGION_COLORS.get(region, '#CCCCCC'), label=region) for region in unique_regions]
            plt.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

            # Save the graph as a PNG file
            output_path = f'output/activity_analysis/{activity.lower().replace(" ", "_")}/{emp_id}_activity_duration.png'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

            # Print confirmation message
            print(f"Graph saved for {activity} - Employee {emp_id} at {output_path}")

if __name__ == "__main__":
    # Specify the base folder where the activity-specific CSV files are stored
    base_folder = 'output'
    generate_activity_barplots(base_folder)