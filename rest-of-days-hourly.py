import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta

# --- Configuration ---
HOURLY_DUMP_FILE = 'hourly_data_dump_9_to_5.csv'
# Define the sprint start dates range to EXCLUDE
SPRINT_START_DATE_REF = '2022-04-27' # First Wednesday to exclude
SPRINT_END_DATE_REF = '2025-04-16'   # Last Wednesday reference date to exclude
NUM_EXTREME_DAYS = 5 # Number of driest/rainiest days to show among the rest

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

# 2. Load the full hourly 9-5 data
try:
    hourly_df_full = pd.read_csv(HOURLY_DUMP_FILE)
    # Parse the 'date_local' column
    try:
        hourly_df_full['date_local'] = pd.to_datetime(hourly_df_full['date_local'], utc=False)
        if hourly_df_full['date_local'].dt.tz is None:
             print("Warning: Timezone info not found in 'date_local'. Assuming local time.")
    except Exception as e:
         print(f"Error parsing 'date_local': {e}. Attempting basic parsing.")
         hourly_df_full['date_local'] = pd.to_datetime(hourly_df_full['date_local'])

    print(f"Loaded {len(hourly_df_full)} total hourly records (9am-5pm).")
    print("Available columns:", hourly_df_full.columns.tolist())

    # Extract date part for filtering
    hourly_df_full['date'] = hourly_df_full['date_local'].dt.date
    hourly_df_full['date'] = pd.to_datetime(hourly_df_full['date']) # Convert back for comparison

except Exception as e:
    print(f"Error reading or processing file '{HOURLY_DUMP_FILE}': {e}")
    exit()

# 3. Identify and Filter out Sprint Start Dates
try:
    print("\nIdentifying sprint start dates to exclude...")
    # Generate target sprint start dates (every other Wednesday) to exclude
    min_data_date = hourly_df_full['date'].min()
    max_data_date = hourly_df_full['date'].max()

    target_sprint_dates_to_exclude = pd.date_range(
        start=SPRINT_START_DATE_REF,
        end=SPRINT_END_DATE_REF,
        freq='14D' # 14 days frequency
    )
    # Ensure we only exclude dates actually present in the data
    target_sprint_dates_to_exclude = target_sprint_dates_to_exclude[
        (target_sprint_dates_to_exclude >= min_data_date) &
        (target_sprint_dates_to_exclude <= max_data_date)
    ]
    print(f"Identified {len(target_sprint_dates_to_exclude)} sprint start dates to exclude within data range.")

    # Filter the dataframe to keep only the "rest of the days"
    hourly_df = hourly_df_full[~hourly_df_full['date'].isin(target_sprint_dates_to_exclude)].copy()

    if hourly_df.empty:
        print("Error: No data remaining after excluding sprint start dates. Cannot proceed.")
        exit()

    print(f"Filtered data to {len(hourly_df)} hourly records for the 'Rest of Days'.")

    # Extract hour (date column already exists)
    hourly_df['hour'] = hourly_df['date_local'].dt.hour

except Exception as e:
    print(f"Error filtering data: {e}")
    exit()


# --- 4. Hourly Pattern Analysis (Rest of Days) ---
print("\nAnalyzing hourly patterns on Rest of Days (9am-5pm)...")

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

        print("\n--- Hourly Summary (Rest of Days Averages, 9am-5pm) ---")
        print(hourly_summary.to_string(float_format='%.3f'))

    except Exception as e:
        print(f"Error during hourly pattern analysis: {e}")
        hourly_summary = None # Ensure summary is None if error occurs
else:
     print("Skipping hourly pattern analysis due to missing columns.")


# --- 5. Daily 9-5 Aggregation & Extreme Day Analysis (Rest of Days) ---
print("\nAnalyzing daily summaries for Rest of Days (9am-5pm)...")

# Define required columns for daily aggregation - Use correct column name
daily_agg_cols = ['date', 'rain']
daily_summary_df = None
# Initialize driest_days as empty df
driest_days = pd.DataFrame()
rainiest_days = pd.DataFrame()


if check_columns(hourly_df, daily_agg_cols):
    try:
        # Group by date and sum rain, count hours
        daily_summary_df = hourly_df.groupby('date').agg(
            total_rain_9_to_5=('rain', 'sum'),
            hours_present=('hour', 'count') # Check how many hours recorded per day
        )

        # Filter out days with incomplete data (less than 8 hours between 9 and 5)
        complete_days_summary = daily_summary_df[daily_summary_df['hours_present'] == 8].copy()

        if not complete_days_summary.empty:
            # Identify Driest Days (within 9-5 window, considering only complete days)
            driest_days = complete_days_summary.nsmallest(NUM_EXTREME_DAYS, 'total_rain_9_to_5')
            print(f"\n--- Top {NUM_EXTREME_DAYS} Driest Rest of Days (9am-5pm, based on total rain) ---")
            print(driest_days.to_string(float_format='%.3f'))

            # Identify Rainiest Days (within 9-5 window, considering only complete days)
            rainiest_days = complete_days_summary.nlargest(NUM_EXTREME_DAYS, 'total_rain_9_to_5')
            print(f"\n--- Top {NUM_EXTREME_DAYS} Rainiest Rest of Days (9am-5pm, based on total rain) ---")
            print(rainiest_days.to_string(float_format='%.3f'))
        else:
            print("\nWarning: No 'Rest of Days' found with complete 8 hours of data between 9am-5pm. Cannot determine driest/rainiest days accurately.")
            # Keep driest_days and rainiest_days as empty DFs

    except Exception as e:
        print(f"Error during daily summary analysis: {e}")
        daily_summary_df = None
        # Keep driest_days and rainiest_days as empty DFs
else:
    print("Skipping daily summary analysis due to missing columns.")
    # Keep driest_days and rainiest_days as empty DFs

# --- 6. Generate Graphs (Rest of Days) ---
print("\nGenerating graphs for hourly patterns on Rest of Days...")

# Graph 1: Average Rain per Hour on Rest of Days
if hourly_summary is not None and 'avg_rain' in hourly_summary.columns:
    try:
        plt.figure(figsize=(10, 5))
        hourly_summary['avg_rain'].plot(kind='bar', color='dodgerblue')
        plt.title('Average Rainfall per Hour on Rest of Days (9am - 5pm)') # Updated Title
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


# Graph 2: Average Temperature per Hour on Rest of Days
if hourly_summary is not None and 'avg_temp' in hourly_summary.columns:
    try:
        plt.figure(figsize=(10, 5))
        hourly_summary['avg_temp'].plot(kind='line', marker='o', color='orangered')
        plt.title('Average Temperature per Hour on Rest of Days (9am - 5pm)') # Updated Title
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


# Graph 3: Most Frequent Weather Code per Hour on Rest of Days
if hourly_summary is not None and 'most_frequent_code' in hourly_summary.columns:
     try:
        plt.figure(figsize=(12, 6))
        x_positions = range(len(hourly_summary.index))
        codes = hourly_summary['most_frequent_code'].fillna(-1).astype(int)
        descriptions = hourly_summary['most_frequent_code_desc']
        bars = plt.bar(x_positions, [1] * len(hourly_summary.index), color='lightgrey', tick_label=hourly_summary.index)
        plt.title('Most Frequent Weather Code per Hour on Rest of Days (9am - 5pm)') # Updated Title
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

# Graph 4: Average Weather Code per Hour on Rest of Days
if hourly_summary is not None and 'avg_code' in hourly_summary.columns:
    try:
        plt.figure(figsize=(10, 5))
        hourly_summary['avg_code'].plot(kind='line', marker='s', color='purple')
        plt.title('Average Weather Code per Hour on Rest of Days (9am - 5pm)') # Updated Title
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


# Graph 5: Rain on Top Rainiest Rest of Days (9am-5pm)
print("\nGenerating graph for rainiest rest of days...")
if 'rainiest_days' in locals() and not rainiest_days.empty and 'total_rain_9_to_5' in rainiest_days.columns:
    try:
        plt.figure(figsize=(10, 5))
        rainiest_days.index = rainiest_days.index.strftime('%Y-%m-%d')
        rainiest_days['total_rain_9_to_5'].plot(kind='bar', color='navy')
        plt.title(f'Total Rainfall (9am-5pm) on Top {NUM_EXTREME_DAYS} Rainiest Rest of Days') # Updated Title
        plt.ylabel('Total Rain (inches)')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Rainiest Rest of Days graph generated.")
    except Exception as e:
        print(f"  Error generating Rainiest Days graph: {e}")
else:
    print(f"  Skipping Rainiest Days graph (data unavailable or required columns missing).")


# Graph 6: Rain on Top Driest Rest of Days (9am-5pm)
print("\nGenerating graph for driest rest of days...")
if 'driest_days' in locals() and not driest_days.empty and 'total_rain_9_to_5' in driest_days.columns:
    try:
        plt.figure(figsize=(10, 5))
        driest_days.index = driest_days.index.strftime('%Y-%m-%d')
        driest_days['total_rain_9_to_5'].plot(kind='bar', color='sandybrown')
        plt.title(f'Total Rainfall (9am-5pm) on Top {NUM_EXTREME_DAYS} Driest Rest of Days') # Updated Title
        plt.ylabel('Total Rain (inches)')
        max_val = driest_days['total_rain_9_to_5'].max()
        plt.ylim(0, max(max_val * 1.5, 0.05)) # Ensure ylim starts at 0
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Driest Rest of Days graph generated.")
    except Exception as e:
        print(f"  Error generating Driest Days graph: {e}")
else:
    print(f"  Skipping Driest Days graph (data unavailable or required columns missing).")


print("\n--- Rest of Days Hourly Analysis Script Complete ---")
