import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import matplotlib.pyplot as plt # Import matplotlib for the new graph

# --- WMO Weather Code Descriptions ---
# Source: https://open-meteo.com/en/docs/historical-weather-api (WMO Weather interpretation codes)
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
# Define code categories for significance ranking
DRIZZLE_CODES = {51, 53, 55, 56, 57}
RAIN_CODES = {61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99} # Including thunderstorms
SNOW_CODES = {71, 73, 75, 77, 85, 86} # Unlikely in PR, but included for completeness
FOG_CODES = {45, 48}
DRY_CODES = {0, 1, 2, 3}

# --- API Client Setup ---
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# --- Date Setup ---
end_date_str = "2025-04-18"
try:
    end_date_obj = datetime.strptime(end_date_str, "%Y-%m-%d")
except ValueError:
    print(f"Error: Invalid end_date format: {end_date_str}. Please use YYYY-MM-DD.")
    exit()

start_date_obj = end_date_obj - timedelta(days=3*365.25)
start_date_str = start_date_obj.strftime("%Y-%m-%d")

# --- API Parameters ---
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 18.4274,
    "longitude": -67.1541,
    "start_date": start_date_str,
    "end_date": end_date_str,
    "daily": ["weather_code", "precipitation_sum", "temperature_2m_max", "rain_sum", "temperature_2m_min", "temperature_2m_mean", "precipitation_probability_mean"],
    "hourly": ["weather_code", "temperature_2m", "rain"],
    "timezone": "auto",
    "temperature_unit": "fahrenheit",
    "precipitation_unit": "inch",
    "timeformat": "unixtime"
}

# --- Fetch Data ---
print(f"Fetching weather data for Lat {params['latitude']}, Lon {params['longitude']}")
print(f"Period: {start_date_str} to {end_date_str}")
try:
    responses = openmeteo.weather_api(url, params=params)
except Exception as e:
    print(f"Error fetching data from API: {e}")
    exit()

# --- Process Response ---
response = responses[0]
print(f"\nProcessing data for:")
print(f"Coordinates: {response.Latitude():.4f}°N {response.Longitude():.4f}°E")
print(f"Elevation: {response.Elevation()} m asl")
try:
    local_timezone_str = response.Timezone().decode('utf-8')
    print(f"Timezone: {local_timezone_str} ({response.TimezoneAbbreviation().decode('utf-8')})")
except:
    print("Warning: Could not decode timezone information. Using UTC for hourly analysis.")
    local_timezone_str = "UTC"
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()} s")


# --- Process & Export Hourly Data ---
hourly = response.Hourly()
hourly_dataframe = pd.DataFrame()
if hourly is None or hourly.VariablesLength() < 3:
     print("Warning: Hourly data might be incomplete or missing. Skipping hourly CSV export and likelihood analysis.")
else:
    hourly_data = {"date_utc": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["weather_code"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["temperature_2m_f"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["rain_inch"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_dataframe = pd.DataFrame(data = hourly_data)

    print("\nExporting detailed hourly data to CSV...")
    try:
        hourly_dataframe_local = hourly_dataframe.copy()
        hourly_dataframe_local['date_local'] = hourly_dataframe_local['date_utc'].dt.tz_convert(local_timezone_str)
        hourly_dataframe_local.to_csv('hourly_weather_data.csv', index=False, float_format='%.2f',
                                      columns=['date_local', 'weather_code', 'temperature_2m_f', 'rain_inch'])
        print("Successfully saved hourly_weather_data.csv (with local time)")
    except Exception as e:
        print(f"Error saving hourly CSV: {e}")


# --- Process & Export Daily Data ---
daily = response.Daily()
daily_dataframe_full = pd.DataFrame()
if daily is None or daily.VariablesLength() < 7:
    print("Warning: Daily data might be incomplete or missing. Skipping daily CSV export and 2-week analysis.")
else:
    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True).date(),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True).date() - timedelta(days=1),
        freq='D'
    )}
    num_days = len(daily_data["date"])
    daily_data["weather_code"] = daily.Variables(0).ValuesAsNumpy()[:num_days]
    daily_data["precipitation_sum_inch"] = daily.Variables(1).ValuesAsNumpy()[:num_days]
    daily_data["temperature_2m_max_f"] = daily.Variables(2).ValuesAsNumpy()[:num_days]
    daily_data["rain_sum_inch"] = daily.Variables(3).ValuesAsNumpy()[:num_days]
    daily_data["temperature_2m_min_f"] = daily.Variables(4).ValuesAsNumpy()[:num_days]
    daily_data["temperature_2m_mean_f"] = daily.Variables(5).ValuesAsNumpy()[:num_days]
    daily_data["precipitation_probability_mean"] = daily.Variables(6).ValuesAsNumpy()[:num_days]
    daily_dataframe_full = pd.DataFrame(data = daily_data)

    print("\nExporting detailed daily data to CSV...")
    try:
        daily_dataframe_full.to_csv('daily_weather_data.csv', index=False, float_format='%.2f')
        print("Successfully saved daily_weather_data.csv")
    except Exception as e:
        print(f"Error saving daily CSV: {e}")


# --- Analyze Rain Likelihood (9am - 5pm) ---
if not hourly_dataframe.empty:
    print("\nAnalyzing rain likelihood during work hours (9am-5pm)...")
    hourly_df_analysis = hourly_dataframe.copy()
    try:
        hourly_df_analysis['local_time'] = hourly_df_analysis['date_utc'].dt.tz_convert(local_timezone_str)
    except Exception as e:
         print(f"Error converting timezone: {e}. Likelihood analysis might use UTC time.")
         hourly_df_analysis['local_time'] = hourly_df_analysis['date_utc']

    working_hours_mask = (hourly_df_analysis['local_time'].dt.hour >= 9) & (hourly_df_analysis['local_time'].dt.hour < 17)
    work_hours_df = hourly_df_analysis[working_hours_mask]

    daily_summary = work_hours_df.groupby(work_hours_df['local_time'].dt.date).agg(
        max_rain_in_period=('rain_inch', 'max'),
        unique_codes_in_period=('weather_code', lambda x: sorted(list(x.unique()))) # Store sorted unique codes
    )

    # Function to determine likelihood and most significant weather
    def determine_likelihood_and_weather(row):
        codes = set(row['unique_codes_in_period']) # Convert list back to set for easier checking
        likelihood = "No Data for Period"
        most_significant_code = None
        most_significant_desc = "N/A"

        if not codes:
            return likelihood, row['unique_codes_in_period'], most_significant_desc # Return original list for codes

        # Determine most significant code based on hierarchy
        if any(c in RAIN_CODES for c in codes):
            most_significant_code = max(c for c in codes if c in RAIN_CODES)
            likelihood = "Likely Rained (Code Indicated)" # Default if no measured rain
        elif any(c in DRIZZLE_CODES for c in codes):
            most_significant_code = max(c for c in codes if c in DRIZZLE_CODES)
            likelihood = "Possible Light Rain/Drizzle"
        elif any(c in SNOW_CODES for c in codes): # Unlikely for PR
             most_significant_code = max(c for c in codes if c in SNOW_CODES)
             likelihood = "Snow/Freezing Precip."
        elif any(c in FOG_CODES for c in codes):
             most_significant_code = max(c for c in codes if c in FOG_CODES)
             likelihood = "Unlikely Rain (Other Phenomena)"
        elif codes.issubset(DRY_CODES):
             most_significant_code = max(codes) # Highest dry code (e.g., 3 for Overcast)
             likelihood = "Most Likely Dry"
        else: # Handle codes not explicitly categorized (e.g., > 3 but not Fog/Drizzle/Rain/Snow)
             most_significant_code = max(codes)
             likelihood = "Unlikely Rain (Other Phenomena)"

        # Override likelihood if measured rain > 0
        if row['max_rain_in_period'] > 0:
            likelihood = "Likely Rained"
            # If measured rain, ensure significant code reflects rain/drizzle if present
            rain_codes_present = codes.intersection(RAIN_CODES)
            drizzle_codes_present = codes.intersection(DRIZZLE_CODES)
            if rain_codes_present:
                 most_significant_code = max(rain_codes_present)
            elif drizzle_codes_present:
                 most_significant_code = max(drizzle_codes_present)
            # Keep original most_significant_code if no rain/drizzle codes found despite measured rain (unusual)

        # Get description for the determined most significant code
        if most_significant_code is not None:
            most_significant_desc = WMO_CODES.get(most_significant_code, f"Unknown code ({most_significant_code})")

        return likelihood, row['unique_codes_in_period'], most_significant_desc

    # Apply the function to get likelihood, codes, and description
    results = daily_summary.apply(determine_likelihood_and_weather, axis=1, result_type='expand')
    daily_summary[['rain_likelihood_9_to_5', 'unique_codes_9_to_5', 'most_significant_weather_9_to_5']] = results

    # Create the final likelihood dataframe
    likelihood_df = daily_summary[['rain_likelihood_9_to_5', 'unique_codes_9_to_5', 'most_significant_weather_9_to_5']].reset_index()
    likelihood_df.rename(columns={'local_time': 'date'}, inplace=True)
    # Convert date column to datetime for resampling later
    likelihood_df['date'] = pd.to_datetime(likelihood_df['date'])


    # --- Export Enhanced Likelihood CSV ---
    print("\nExporting enhanced daily rain likelihood data to CSV...")
    try:
        # Convert list of codes to string for CSV compatibility
        likelihood_df_csv = likelihood_df.copy()
        likelihood_df_csv['unique_codes_9_to_5'] = likelihood_df_csv['unique_codes_9_to_5'].astype(str)
        likelihood_df_csv.to_csv('daily_rain_likelihood_9_to_5.csv', index=False)
        print("Successfully saved daily_rain_likelihood_9_to_5.csv")
    except Exception as e:
        print(f"Error saving likelihood CSV: {e}")

    # --- Generate Overall Likelihood Graph (Keep as is) ---
    print("\nGenerating overall rain likelihood graph...")
    try:
        likelihood_counts = likelihood_df['rain_likelihood_9_to_5'].value_counts()
        category_order = [
            "Likely Rained", "Likely Rained (Code Indicated)", "Possible Light Rain/Drizzle",
            "Unlikely Rain (Other Phenomena)", "Most Likely Dry", "Snow/Freezing Precip.", "No Data for Period"
        ]
        likelihood_counts = likelihood_counts.reindex(category_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        likelihood_counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#aec7e8', '#ffbb78', '#98df8a', '#2ca02c', '#ff9896', '#d3d3d3'])
        ax.set_title(f'Overall Rain Likelihood During Work Hours (9am-5pm)\n{start_date_str} to {end_date_str}')
        ax.set_ylabel('Number of Days')
        ax.set_xlabel('Likelihood Category')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
        print("Graph generated.")
    except Exception as e:
        print(f"Error generating likelihood graph: {e}")

    # --- NEW: Create Two-Week Likelihood Summary Table ---
    print("\nCalculating two-week summary of rain likelihoods...")
    try:
        likelihood_df.set_index('date', inplace=True)
        # Resample by 2 weeks and count likelihood occurrences
        two_week_likelihood_summary = likelihood_df.resample('2W', label='right', closed='right')['rain_likelihood_9_to_5'] \
                                                .value_counts() \
                                                .unstack(fill_value=0) # Pivot categories into columns

        # Ensure all categories from category_order are present as columns
        for cat in category_order:
            if cat not in two_week_likelihood_summary.columns:
                two_week_likelihood_summary[cat] = 0
        # Reorder columns consistently
        two_week_likelihood_summary = two_week_likelihood_summary[category_order]

        # Add period start date
        two_week_likelihood_summary['Period Start Date'] = two_week_likelihood_summary.index - pd.Timedelta(days=13)

        # Format for printing
        two_week_likelihood_summary.reset_index(inplace=True) # Make Period End Date a column
        two_week_likelihood_summary.rename(columns={'date': 'Period End Date'}, inplace=True)
        two_week_likelihood_summary['Period Start Date'] = two_week_likelihood_summary['Period Start Date'].dt.strftime('%Y-%m-%d')
        two_week_likelihood_summary['Period End Date'] = two_week_likelihood_summary['Period End Date'].dt.strftime('%Y-%m-%d')

        # Reorder columns for printing
        print_cols = ['Period Start Date', 'Period End Date'] + category_order
        two_week_likelihood_summary = two_week_likelihood_summary[print_cols]

        print("\n--- Rain Likelihood Summary (Every Two Weeks, 9am-5pm) ---")
        print("(Counts represent number of days in each category per period)")
        print(two_week_likelihood_summary.to_string(index=False))

    except Exception as e:
        print(f"Error generating two-week likelihood summary: {e}")

else:
    print("\nSkipping rain likelihood analysis because hourly data is missing.")


# --- Analyze 2-Week Rainfall Amount (Original Analysis) ---
if not daily_dataframe_full.empty and "rain_sum_inch" in daily_dataframe_full.columns:
    print("\nAnalyzing rainfall amount in 2-week intervals...")
    daily_dataframe_analysis = daily_dataframe_full.copy()
    daily_dataframe_analysis['date'] = pd.to_datetime(daily_dataframe_analysis['date'])
    daily_dataframe_analysis.set_index('date', inplace=True)

    two_weekly_rain = daily_dataframe_analysis['rain_sum_inch'].resample('2W', label='right', closed='right').sum()

    results_table = pd.DataFrame({
        'Period End Date': two_weekly_rain.index,
        'Total Rain (inch)': two_weekly_rain.values
    })
    results_table['Rained'] = results_table['Total Rain (inch)'] > 0.0
    results_table['Period Start Date'] = results_table['Period End Date'] - pd.Timedelta(days=13)
    results_table = results_table[['Period Start Date', 'Period End Date', 'Total Rain (inch)', 'Rained']]

    print("\n--- Rainfall Amount Summary (Every Two Weeks) ---")
    results_table_display = results_table.copy()
    results_table_display['Period Start Date'] = results_table_display['Period Start Date'].dt.strftime('%Y-%m-%d')
    results_table_display['Period End Date'] = results_table_display['Period End Date'].dt.strftime('%Y-%m-%d')
    results_table_display['Total Rain (inch)'] = results_table_display['Total Rain (inch)'].round(2)
    print(results_table_display.to_string(index=False))
else:
    print("\nSkipping 2-week rainfall amount analysis because daily data or 'rain_sum_inch' column is missing.")


print("\n--- Analysis Complete ---")
