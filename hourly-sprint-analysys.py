import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta

# --- Configuration ---
HOURLY_DUMP_FILE = 'hourly_data_dump_9_to_5.csv'
FILTERED_SPRINT_HOURLY_FILE = 'sprint_start_hourly_data_9_to_5.csv' # New file for filtered hourly data
SPRINT_START_DATE = '2022-04-27' # First Wednesday to consider (adjust if needed based on data range)
SPRINT_END_DATE = '2025-04-16'   # Last Wednesday reference date
NUM_EXTREME_DAYS = 5 # Number of driest/rainiest sprint start days to show

# --- WMO Weather Code Descriptions (Copied for reference) ---
WMO_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Depositing rime fog',
    51: 'Drizzle, light intensity', 53: 'Drizzle, moderate intensity', 55: 'Drizzle, dense intensity',
    56: 'Freezing Drizzle, light intensity', 57: 'Freezing Drizzle, dense intensity',
    61: 'Rain, slight intensity', 63: 'Rain, moderate intensity', 65: 'Rain, heavy intensity',
    66: 'Freezing Rain, light intensity', 67: 'Freezing Rain, heavy intensity',
    71: 'Snow fall, slight intensity', 73: 'Snow fall, moderate intensity', 75: 'Snow fall, heavy intensity',
    77: 'Snow grains',
    80: 'Rain showers, slight', 81: 'Rain showers, moderate', 82: 'Rain showers, violent',
    85: 'Snow showers, slight', 86: 'Snow showers, heavy',
    95: 'Thunderstorm, slight or moderate',
    96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
}

# --- Main Script Logic ---

# 1. Check if base hourly data dump exists
if not os.path.exists(HOURLY_DUMP_FILE):
    print(f"Error: Input file '{HOURLY_DUMP_FILE}' not found.")
    print("Please run the data dump script first to generate it.")
    exit()

# 2. Check if filtered sprint hourly file exists, generate if not
if not os.path.exists(FILTERED_SPRINT_HOURLY_FILE):
    print(f"Filtered file '{FILTERED_SPRINT_HOURLY_FILE}' not found. Generating...")
    try:
        # Load the full 9-5 hourly dump
        hourly_df_full = pd.read_csv(HOURLY_DUMP_FILE)
        # Parse the 'date_local' column
        try:
            hourly_df_full['date_local'] = pd.to_datetime(hourly_df_full['date_local'], utc=False)
            if hourly_df_full['date_local'].dt.tz is None:
                 print("Warning: Timezone info not found in 'date_local' during filtering. Proceeding...")
        except Exception as e:
             print(f"Error parsing 'date_local' during filtering: {e}. Attempting basic parsing.")
             hourly_df_full['date_local'] = pd.to_datetime(hourly_df_full['date_local'])

        # Extract date part for filtering
        hourly_df_full['date'] = hourly_df_full['date_local'].dt.date
        hourly_df_full['date'] = pd.to_datetime(hourly_df_full['date']) # Convert back for comparison

        # Generate target sprint start dates (every other Wednesday)
        # Ensure the range covers the data available in the dump
        min_data_date = hourly_df_full['date'].min()
        max_data_date = hourly_df_full['date'].max()
        print(f"Data range in dump: {min_data_date.strftime('%Y-%m-%d')} to {max_data_date.strftime('%Y-%m-%d')}")

        # Generate dates forward from SPRINT_START_DATE to ensure correct sequence
        target_sprint_dates = pd.date_range(
            start=SPRINT_START_DATE,
            end=SPRINT_END_DATE, # Use the defined end date
            freq='14D' # 14 days frequency
        )
        # Filter target dates to be within the actual data range
        target_sprint_dates = target_sprint_dates[(target_sprint_dates >= min_data_date) & (target_sprint_dates <= max_data_date)]

        print(f"Generated {len(target_sprint_dates)} target sprint dates within data range.")

        # Filter the hourly dataframe
        sprint_hourly_df = hourly_df_full[hourly_df_full['date'].isin(target_sprint_dates)].copy()

        if sprint_hourly_df.empty:
             print(f"Warning: No hourly data found for the specified sprint start dates in '{HOURLY_DUMP_FILE}'. Cannot generate '{FILTERED_SPRINT_HOURLY_FILE}'.")
             exit()

        # Save the filtered data (dropping the temporary date column)
        sprint_hourly_df.drop(columns=['date'], inplace=True) # Drop the date-only column before saving
        sprint_hourly_df.to_csv(FILTERED_SPRINT_HOURLY_FILE, index=False, float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S %Z')
        print(f"Successfully generated and saved '{FILTERED_SPRINT_HOURLY_FILE}'.")
        # Use the newly created dataframe for the rest of the script
        hourly_df = sprint_hourly_df

    except Exception as e:
        print(f"Error generating filtered file: {e}")
        exit()
else:
    # If file exists, load it directly
    print(f"Using existing filtered file: '{FILTERED_SPRINT_HOURLY_FILE}'")
    try:
        hourly_df = pd.read_csv(FILTERED_SPRINT_HOURLY_FILE)
        # Parse the 'date_local' column
        try:
            hourly_df['date_local'] = pd.to_datetime(hourly_df['date_local'], utc=False)
            if hourly_df['date_local'].dt.tz is None:
                 print("Warning: Timezone info not found in 'date_local'. Assuming local time.")
        except Exception as e:
             print(f"Error parsing 'date_local': {e}. Attempting basic parsing.")
             hourly_df['date_local'] = pd.to_datetime(hourly_df['date_local'])

    except Exception as e:
        print(f"Error reading filtered file '{FILTERED_SPRINT_HOURLY_FILE}': {e}")
        exit()

# 3. Prepare loaded/filtered data for analysis
try:
    print(f"\nLoaded {len(hourly_df)} hourly records for Sprint Start days (9am-5pm).")
    if hourly_df.empty:
        print("No data to analyze. Exiting.")
        exit()
    print("Available columns:", hourly_df.columns.tolist())

    # Extract hour and date for grouping
    hourly_df['hour'] = hourly_df['date_local'].dt.hour
    hourly_df['date'] = hourly_df['date_local'].dt.date
    hourly_df['date'] = pd.to_datetime(hourly_df['date']) # Convert date back to datetime

except Exception as e:
    print(f"Error preparing data for analysis: {e}")
    exit()


# --- 4. Hourly Pattern Analysis (Now specific to Sprint Starts) ---
print("\nAnalyzing hourly patterns on Sprint Start Days (9am-5pm)...")

# Define required columns for hourly analysis - Use correct column names
hourly_analysis_cols = ['hour', 'rain', 'weather_code', 'temperature_2m']

def check_columns(df, required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Skipping analysis/graph due to missing columns: {', '.join(missing_cols)}")
        return False
    return True

hourly_summary = None
if check_columns(hourly_df, hourly_analysis_cols):
    try:
        # Group by hour
        hourly_groups = hourly_df.groupby('hour')

        # Calculate metrics - Use correct column names
        hourly_summary = hourly_groups.agg(
            avg_rain=('rain', 'mean'),
            avg_temp=('temperature_2m', 'mean'),
            # Calculate average weather code (interpret with caution)
            avg_code=('weather_code', 'mean'),
            # Find the most frequent non-NaN weather code for each hour
            most_frequent_code=('weather_code', lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        )
        # Add description for the most frequent code
        hourly_summary['most_frequent_code_desc'] = hourly_summary['most_frequent_code'].map(WMO_CODES).fillna('Unknown/Missing')

        print("\n--- Hourly Summary (Sprint Start Day Averages, 9am-5pm) ---")
        print(hourly_summary.to_string(float_format='%.3f'))

    except Exception as e:
        print(f"Error during hourly pattern analysis: {e}")
        hourly_summary = None # Ensure summary is None if error occurs
else:
     print("Skipping hourly pattern analysis due to missing columns.")


# --- 5. Daily 9-5 Aggregation & Extreme Day Analysis (Now specific to Sprint Starts) ---
print("\nAnalyzing daily summaries for Sprint Start Days (9am-5pm)...")

# Define required columns for daily aggregation - Use correct column name
daily_agg_cols = ['date', 'rain']
daily_summary_df = None

if check_columns(hourly_df, daily_agg_cols):
    try:
        # Group by date and sum rain, count hours (should be 8 if no data missing)
        # Use correct column name
        daily_summary_df = hourly_df.groupby('date').agg(
            total_rain_9_to_5=('rain', 'sum'),
            hours_present=('hour', 'count') # Check how many hours recorded per day
        )

        # Filter out days with incomplete data (less than 8 hours between 9 and 5)
        complete_days_summary = daily_summary_df[daily_summary_df['hours_present'] == 8].copy()

        if not complete_days_summary.empty:
            # Identify Driest Days (within 9-5 window, considering only complete days)
            # Use correct column name
            driest_days = complete_days_summary.nsmallest(NUM_EXTREME_DAYS, 'total_rain_9_to_5')
            print(f"\n--- Top {NUM_EXTREME_DAYS} Driest Sprint Start Days (9am-5pm, based on total rain) ---")
            print(driest_days.to_string(float_format='%.3f'))

            # Identify Rainiest Days (within 9-5 window, considering only complete days)
            # Use correct column name
            rainiest_days = complete_days_summary.nlargest(NUM_EXTREME_DAYS, 'total_rain_9_to_5')
            print(f"\n--- Top {NUM_EXTREME_DAYS} Rainiest Sprint Start Days (9am-5pm, based on total rain) ---")
            print(rainiest_days.to_string(float_format='%.3f'))
        else:
            print("\nWarning: No sprint start days found with complete 8 hours of data between 9am-5pm. Cannot determine driest/rainiest days accurately.")
            rainiest_days = pd.DataFrame() # Ensure rainiest_days is an empty DF

    except Exception as e:
        print(f"Error during daily summary analysis: {e}")
        daily_summary_df = None
        rainiest_days = pd.DataFrame() # Ensure rainiest_days is an empty DF
else:
    print("Skipping daily summary analysis due to missing columns.")
    rainiest_days = pd.DataFrame() # Ensure rainiest_days is an empty DF

# --- 6. Generate Graphs (Now specific to Sprint Starts) ---
print("\nGenerating graphs for hourly patterns on Sprint Start Days...")

# Graph 1: Average Rain per Hour on Sprint Starts
if hourly_summary is not None and 'avg_rain' in hourly_summary.columns:
    try:
        plt.figure(figsize=(10, 5))
        hourly_summary['avg_rain'].plot(kind='bar', color='dodgerblue')
        plt.title('Average Rainfall per Hour on Sprint Starts (9am - 5pm)') # Updated Title
        plt.ylabel('Average Rain (inches)')
        plt.xlabel('Hour of Day (Local Time)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Average Rain per Hour graph generated.")
    except Exception as e:
        print(f"  Error generating Average Rain graph: {e}")
else:
     print("  Skipping Average Rain per Hour graph (data unavailable).")


# Graph 2: Average Temperature per Hour on Sprint Starts
if hourly_summary is not None and 'avg_temp' in hourly_summary.columns:
    try:
        plt.figure(figsize=(10, 5))
        hourly_summary['avg_temp'].plot(kind='line', marker='o', color='orangered')
        plt.title('Average Temperature per Hour on Sprint Starts (9am - 5pm)') # Updated Title
        plt.ylabel('Average Temperature (°F)')
        plt.xlabel('Hour of Day (Local Time)')
        plt.xticks(ticks=hourly_summary.index)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Average Temperature per Hour graph generated.")
    except Exception as e:
        print(f"  Error generating Average Temperature graph: {e}")
else:
     print("  Skipping Average Temperature per Hour graph (data unavailable).")


# Graph 3: Most Frequent Weather Code per Hour on Sprint Starts
if hourly_summary is not None and 'most_frequent_code' in hourly_summary.columns:
     try:
        plt.figure(figsize=(12, 6))
        x_positions = range(len(hourly_summary.index))
        codes = hourly_summary['most_frequent_code'].fillna(-1).astype(int)
        descriptions = hourly_summary['most_frequent_code_desc']
        bars = plt.bar(x_positions, [1] * len(hourly_summary.index), color='lightgrey', tick_label=hourly_summary.index)
        plt.title('Most Frequent Weather Code per Hour on Sprint Starts (9am - 5pm)') # Updated Title
        plt.ylabel('')
        plt.yticks([])
        plt.xlabel('Hour of Day (Local Time) and Most Frequent Code')
        tick_labels = [f"{hour}: {desc}" for hour, desc in zip(hourly_summary.index, descriptions)]
        plt.xticks(ticks=x_positions, labels=tick_labels, rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Most Frequent Weather Code per Hour graph generated.")
     except Exception as e:
        print(f"  Error generating Most Frequent Code graph: {e}")
else:
     print("  Skipping Most Frequent Weather Code per Hour graph (data unavailable).")

# Graph 4: Average Weather Code per Hour on Sprint Starts
if hourly_summary is not None and 'avg_code' in hourly_summary.columns:
    try:
        plt.figure(figsize=(10, 5))
        hourly_summary['avg_code'].plot(kind='line', marker='s', color='purple')
        plt.title('Average Weather Code per Hour on Sprint Starts (9am - 5pm)') # Updated Title
        plt.ylabel('Average WMO Weather Code (Interpret with caution)')
        plt.xlabel('Hour of Day (Local Time)')
        plt.xticks(ticks=hourly_summary.index)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Average Weather Code per Hour graph generated.")
    except Exception as e:
        print(f"  Error generating Average Weather Code graph: {e}")
else:
     print("  Skipping Average Weather Code per Hour graph (data unavailable).")


# Graph 5: Rain on Top Rainiest Sprint Start Days (9am-5pm)
print("\nGenerating graph for rainiest sprint start days...")
if 'rainiest_days' in locals() and not rainiest_days.empty and 'total_rain_9_to_5' in rainiest_days.columns:
    try:
        plt.figure(figsize=(10, 5))
        rainiest_days.index = rainiest_days.index.strftime('%Y-%m-%d')
        rainiest_days['total_rain_9_to_5'].plot(kind='bar', color='navy')
        plt.title(f'Total Rainfall (9am-5pm) on Top {NUM_EXTREME_DAYS} Rainiest Sprint Start Days') # Updated Title
        plt.ylabel('Total Rain (inches)')
        plt.xlabel('Sprint Start Date')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Rainiest Sprint Start Days graph generated.")
    except Exception as e:
        print(f"  Error generating Rainiest Days graph: {e}")
else:
    print(f"  Skipping Rainiest Days graph (data unavailable or required columns missing).")


print("\n--- Hourly Sprint Start Analysis Script Complete ---")
