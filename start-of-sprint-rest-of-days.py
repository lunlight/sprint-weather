import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta

# --- Configuration ---
FULL_DUMP_FILE = 'full_daily_data_dump.csv'
# Define the sprint start dates range to EXCLUDE
SPRINT_START_DATE_REF = '2022-04-27' # First Wednesday to exclude
SPRINT_END_DATE_REF = '2025-04-16'   # Last Wednesday reference date to exclude

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
# Define code categories for likelihood heuristic
PRECIP_CODES = {
    51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77,
    80, 81, 82, 85, 86, 95, 96, 99
}

# --- Helper Function for Likelihood ---
def calculate_daily_likelihood(row):
    """Calculates simple rain likelihood based on daily data."""
    # Use .get with default 0 or np.nan to handle potentially missing columns gracefully
    # Assuming column names might be 'rain_sum' or 'rain_sum_inch', etc. Check available columns.
    rain = row.get('rain_sum_inch', row.get('rain_sum', 0))
    precip = row.get('precipitation_sum_inch', row.get('precipitation_sum', 0))
    code = row.get('weather_code', np.nan)

    # Handle potential NaN values after .get() before comparison
    rain = 0 if pd.isna(rain) else rain
    precip = 0 if pd.isna(precip) else precip

    # Check if essential data was actually missing
    rain_missing = pd.isna(row.get('rain_sum_inch', row.get('rain_sum', np.nan)))
    precip_missing = pd.isna(row.get('precipitation_sum_inch', row.get('precipitation_sum', np.nan)))
    code_missing = pd.isna(code)

    if rain_missing or precip_missing or code_missing:
        return "Data Missing"
    if rain > 0:
        return "Rained (Measured)"
    if precip > 0:
        return "Precipitation (No Rain Measured)"
    if code in PRECIP_CODES:
        return "Precipitation Indicated (Code)"
    # If code is not NaN, not in PRECIP_CODES and precip/rain are 0
    return "Likely Dry"

# --- Main Script Logic ---

# 1. Check if full data dump exists
if not os.path.exists(FULL_DUMP_FILE):
    print(f"Error: Input file '{FULL_DUMP_FILE}' not found.")
    print("Please run the data dump script first.")
    exit()

# 2. Load the full daily data dump
try:
    full_daily_df = pd.read_csv(FULL_DUMP_FILE)
    # Ensure 'date' column is parsed correctly as dates (without time)
    try:
        full_daily_df['date'] = pd.to_datetime(full_daily_df['date']).dt.date
        full_daily_df['date'] = pd.to_datetime(full_daily_df['date']) # Convert back for comparison
    except Exception as date_err:
        print(f"Error parsing 'date' column in {FULL_DUMP_FILE}: {date_err}")
        print("Please check the format of the 'date' column in the dump file.")
        exit()

    print(f"Loaded {len(full_daily_df)} total daily records.")
    print("Available columns:", full_daily_df.columns.tolist()) # Print available columns for debugging

except Exception as e:
    print(f"Error reading file '{FULL_DUMP_FILE}': {e}")
    exit()

# 3. Identify and Filter out Sprint Start Dates
try:
    print("\nIdentifying sprint start dates to exclude...")
    # Generate target sprint start dates (every other Wednesday) to exclude
    min_data_date = full_daily_df['date'].min()
    max_data_date = full_daily_df['date'].max()

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
    rest_of_days_df = full_daily_df[~full_daily_df['date'].isin(target_sprint_dates_to_exclude)].copy()

    if rest_of_days_df.empty:
        print("Error: No data remaining after excluding sprint start dates. Cannot proceed.")
        exit()

    print(f"Filtered data to {len(rest_of_days_df)} daily records for the 'Rest of Days'.")

except Exception as e:
    print(f"Error filtering data: {e}")
    exit()


# 4. Calculate Daily Rain Likelihood for the filtered data
print("\nCalculating daily rain likelihood for Rest of Days...")
# Apply likelihood function - it's designed to handle missing columns
rest_of_days_df['rain_likelihood_daily'] = rest_of_days_df.apply(calculate_daily_likelihood, axis=1)

# --- 5. Generate Graphs (Rest of Days) ---
print("\nGenerating graphs for Rest of Days...")

# Define required columns for each graph - adjust based on available columns printed earlier
likelihood_cols = ['rain_likelihood_daily']
# Try to use columns with units first, fall back to base names if necessary
precip_col = 'precipitation_sum_inch' if 'precipitation_sum_inch' in rest_of_days_df.columns else 'precipitation_sum'
rain_col = 'rain_sum_inch' if 'rain_sum_inch' in rest_of_days_df.columns else 'rain_sum'
precip_cols_req = ['date', precip_col, rain_col]

code_cols_req = ['weather_code']

temp_max_col = 'temperature_2m_max_f' if 'temperature_2m_max_f' in rest_of_days_df.columns else 'temperature_2m_max'
temp_mean_col = 'temperature_2m_mean_f' if 'temperature_2m_mean_f' in rest_of_days_df.columns else 'temperature_2m_mean'
temp_min_col = 'temperature_2m_min_f' if 'temperature_2m_min_f' in rest_of_days_df.columns else 'temperature_2m_min'
temp_cols_req = ['date', temp_max_col, temp_mean_col, temp_min_col]


# Function to check if all required columns exist
def check_columns(df, required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Skipping graph due to missing columns: {', '.join(missing_cols)}")
        return False
    return True

# Graph 1: Rain Likelihood Counts
print("\nAttempting to generate Likelihood graph (Rest of Days)...")
if check_columns(rest_of_days_df, likelihood_cols):
    try:
        plt.figure(figsize=(8, 5))
        likelihood_counts = rest_of_days_df['rain_likelihood_daily'].value_counts()
        # Define a specific order and colors for consistency
        category_order = ["Likely Dry", "Precipitation Indicated (Code)", "Precipitation (No Rain Measured)", "Rained (Measured)", "Data Missing"]
        colors = ['#2ca02c', '#ff7f0e', '#aec7e8', '#1f77b4', '#d3d3d3']
        likelihood_counts = likelihood_counts.reindex(category_order, fill_value=0) # Ensure order and all categories exist

        likelihood_counts.plot(kind='bar', color=colors)
        plt.title(f'Daily Rain Likelihood on Rest of Days\n(Excluding bi-weekly Wednesdays)')
        plt.ylabel('Number of Days')
        plt.xlabel('Likelihood Category')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        print("  Likelihood graph generated.")
    except Exception as e:
        print(f"  Error generating Likelihood graph: {e}")

# Graph 2: Precipitation Level
print("\nAttempting to generate Precipitation graph (Rest of Days)...")
if check_columns(rest_of_days_df, precip_cols_req):
    try:
        plt.figure(figsize=(12, 5))
        # Use determined column names
        plt.plot(rest_of_days_df['date'], rest_of_days_df[precip_col], color='skyblue', label='Precipitation Sum (inch)', alpha=0.7, linewidth=0.8)
        plt.plot(rest_of_days_df['date'], rest_of_days_df[rain_col], color='blue', label='Rain Sum (inch)', alpha=0.7, linewidth=0.8)
        # Consider plotting as lines for daily data over long period
        # Or maybe resample/aggregate before plotting bars if too cluttered

        plt.title(f'Daily Precipitation on Rest of Days\n(Excluding bi-weekly Wednesdays)')
        plt.ylabel('Precipitation (inches)') # Assumes inch unit from dump script
        plt.xlabel('Date')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Adjust interval
        plt.gcf().autofmt_xdate() # Auto adjust date labels
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Precipitation graph generated.")
    except Exception as e:
        print(f"  Error generating Precipitation graph: {e}")

# Graph 3: Weather Code Frequency
print("\nAttempting to generate Weather Code graph (Rest of Days)...")
if check_columns(rest_of_days_df, code_cols_req):
    try:
        plt.figure(figsize=(10, 6))
        # Drop rows where weather_code might be NaN before counting
        valid_codes_df = rest_of_days_df.dropna(subset=['weather_code'])
        weather_code_counts = valid_codes_df['weather_code'].astype(int).value_counts().sort_index()
        # Create labels with descriptions
        labels = [f"{code}: {WMO_CODES.get(code, 'Unknown')}" for code in weather_code_counts.index]

        plt.bar(labels, weather_code_counts.values, color='lightblue')
        plt.title(f'Weather Code Frequency on Rest of Days\n(Excluding bi-weekly Wednesdays)')
        plt.ylabel('Number of Days')
        plt.xlabel('WMO Weather Code')
        plt.xticks(rotation=90) # Rotate labels for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Weather Code graph generated.")
    except Exception as e:
        print(f"  Error generating Weather Code graph: {e}")

# Graph 4: Temperature Trends
print("\nAttempting to generate Temperature graph (Rest of Days)...")
if check_columns(rest_of_days_df, temp_cols_req):
    try:
        plt.figure(figsize=(12, 5))
        # Use determined column names
        plt.plot(rest_of_days_df['date'], rest_of_days_df[temp_max_col], label='Max Temp (°F)', color='red', alpha=0.7, linewidth=0.8)
        plt.plot(rest_of_days_df['date'], rest_of_days_df[temp_mean_col], label='Mean Temp (°F)', color='orange', alpha=0.7, linewidth=0.8)
        plt.plot(rest_of_days_df['date'], rest_of_days_df[temp_min_col], label='Min Temp (°F)', color='blue', alpha=0.7, linewidth=0.8)

        plt.title(f'Daily Temperature Trends on Rest of Days\n(Excluding bi-weekly Wednesdays)')
        plt.ylabel('Temperature (°F)') # Assumes F unit from dump script
        plt.xlabel('Date')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Adjust interval
        plt.gcf().autofmt_xdate() # Auto adjust date labels
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("  Temperature graph generated.")
    except Exception as e:
        print(f"  Error generating Temperature graph: {e}")

print("\n--- Rest of Days Daily Analysis Script Complete ---")
