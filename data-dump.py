import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np # Import numpy for nan and array checks
from retry_requests import retry
from datetime import datetime, timedelta
import time # To add slight delay

# --- WMO Weather Code Descriptions (Optional, but good for reference) ---
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

# --- API Client Setup ---
# Use a cache to store downloaded data and avoid re-fetching
cache_session = requests_cache.CachedSession('.cache', expire_after = -1) # Cache indefinitely
# Retry failed requests
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# --- Date Setup ---
# Use the same date range as the analysis script
end_date_str = "2025-04-18"
try:
    end_date_obj = datetime.strptime(end_date_str, "%Y-%m-%d")
except ValueError:
    print(f"Error: Invalid end_date format: {end_date_str}. Please use YYYY-MM-DD.")
    exit()

# Calculate start date (3 years prior)
start_date_obj = end_date_obj - timedelta(days=3*365.25)
start_date_str = start_date_obj.strftime("%Y-%m-%d")

# --- API Parameters ---
# Define location
latitude = 18.4274
longitude = -67.1541

# Define ALL available parameters according to Open-Meteo Historical API docs (as of late 2023/early 2024)
# Check https://open-meteo.com/en/docs/historical-weather-api for the most current list
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": latitude,
	"longitude": longitude,
	"start_date": start_date_str,
	"end_date": end_date_str,
	"hourly": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
        "precipitation", "rain", "snowfall", "snow_depth", "weather_code",
        "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
        "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
        "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
        "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
        "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
        "is_day", "sunshine_duration", "shortwave_radiation",
        "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
        "global_tilted_irradiance", "terrestrial_radiation"
        # Add more if available/needed, e.g., ERA5 specific variables if using that model explicitly
    ],
	"daily": [
        "weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
        "sunrise", "sunset", "daylight_duration", "sunshine_duration",
        "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
        "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration"
    ],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch",
	"timeformat": "unixtime", # Use unixtime for easy conversion
	"timezone": "auto" # Important for local time conversion later
}

# --- Fetch Data ---
print(f"Fetching full weather data dump for Lat {params['latitude']}, Lon {params['longitude']}")
print(f"Period: {start_date_str} to {end_date_str}")
print("Requesting all available daily and hourly parameters...")

try:
    responses = openmeteo.weather_api(url, params=params)
    # Optional: Add a small delay after the request
    time.sleep(1)
except Exception as e:
    print(f"Fatal Error: Could not fetch data from API: {e}")
    exit()

# --- Process Response ---
response = responses[0]
print(f"\nData received for:")
print(f"Coordinates: {response.Latitude():.4f}°N {response.Longitude():.4f}°E")
print(f"Elevation: {response.Elevation()} m asl")
try:
    local_timezone_str = response.Timezone().decode('utf-8')
    print(f"Timezone: {local_timezone_str} ({response.TimezoneAbbreviation().decode('utf-8')})")
except:
    print("Warning: Could not decode timezone information. Using UTC.")
    local_timezone_str = "UTC" # Fallback
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()} s")

# --- Process & Save FULL Daily Data ---
daily = response.Daily()
if daily is not None and daily.VariablesLength() > 0:
    print("\nProcessing full daily data...")
    # Create datetime index for daily data
    daily_dates = pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True).date(),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True).date() - timedelta(days=1),
        freq='D'
    )
    num_days_expected = len(daily_dates)

    daily_data_dict = {"date": daily_dates}

    # Dynamically process all returned daily variables
    for i in range(daily.VariablesLength()):
        variable = daily.Variables(i)
        # FIX: Check if variable object exists
        if variable is None:
            print(f"    Warning: Daily variable at index {i} is None. Skipping.")
            continue # Skip to the next variable

        # Ensure index i is valid for params['daily'] list
        if i >= len(params['daily']):
             print(f"    Warning: More variables returned ({daily.VariablesLength()}) than requested in daily params ({len(params['daily'])}). Skipping extra variable at index {i}.")
             continue

        var_name = params['daily'][i] # Get name from our request list
        print(f"  - Processing daily variable: {var_name}")

        # Get values, handle potential None return
        values = variable.ValuesAsNumpy()
        if values is None:
             print(f"    Warning: ValuesAsNumpy() returned None for daily '{var_name}'. Skipping.")
             continue # Skip to the next variable

        # FIX: Check if 'values' is actually array-like (has length) before using len()
        if not hasattr(values, '__len__'):
             print(f"    Error: Values for daily '{var_name}' is not array-like (type: {type(values)}). Skipping.")
             # Optionally log the value: print(f"Value was: {values}")
             continue # Skip to the next variable

        # Now it should be safe to check length
        if len(values) == num_days_expected:
            daily_data_dict[var_name] = values
        else:
            print(f"    Warning: Length mismatch for daily '{var_name}'. Expected {num_days_expected}, got {len(values)}. Padding with NaN.")
            # Pad with NaN if lengths don't match (might happen with API issues)
            padded_values = np.full(num_days_expected, np.nan)
            len_to_copy = min(len(values), num_days_expected)
            padded_values[:len_to_copy] = values[:len_to_copy]
            daily_data_dict[var_name] = padded_values


    daily_dataframe_full = pd.DataFrame(data=daily_data_dict)

    # Handle potential sunrise/sunset unix timestamps if requested
    if 'sunrise' in daily_dataframe_full.columns and pd.api.types.is_numeric_dtype(daily_dataframe_full['sunrise']):
        # Check if column exists and contains numeric data before conversion
        daily_dataframe_full['sunrise'] = pd.to_datetime(daily_dataframe_full['sunrise'], unit='s', utc=True).dt.tz_convert(local_timezone_str)
    if 'sunset' in daily_dataframe_full.columns and pd.api.types.is_numeric_dtype(daily_dataframe_full['sunset']):
         # Check if column exists and contains numeric data before conversion
        daily_dataframe_full['sunset'] = pd.to_datetime(daily_dataframe_full['sunset'], unit='s', utc=True).dt.tz_convert(local_timezone_str)


    print("\nSaving full daily data dump to CSV...")
    try:
        daily_dataframe_full.to_csv('full_daily_data_dump.csv', index=False, float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S %Z')
        print("Successfully saved full_daily_data_dump.csv")
    except Exception as e:
        print(f"Error saving full daily CSV: {e}")
else:
    print("\nNo daily data returned by API.")


# --- Process & Save Filtered (9am-5pm) Hourly Data ---
hourly = response.Hourly()
if hourly is not None and hourly.VariablesLength() > 0:
    print("\nProcessing full hourly data...")
    # Create datetime index for hourly data (UTC first)
    hourly_dates_utc = pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )
    num_hours_expected = len(hourly_dates_utc)

    hourly_data_dict = {"date_utc": hourly_dates_utc}

    # Dynamically process all returned hourly variables
    for i in range(hourly.VariablesLength()):
        variable = hourly.Variables(i)
         # FIX: Check if variable object exists
        if variable is None:
            print(f"    Warning: Hourly variable at index {i} is None. Skipping.")
            continue

         # Ensure index i is valid for params['hourly'] list
        if i >= len(params['hourly']):
             print(f"    Warning: More variables returned ({hourly.VariablesLength()}) than requested in hourly params ({len(params['hourly'])}). Skipping extra variable at index {i}.")
             continue

        var_name = params['hourly'][i] # Get name from our request list
        print(f"  - Processing hourly variable: {var_name}")

        # Get values, handle potential None return
        values = variable.ValuesAsNumpy()
        if values is None:
             print(f"    Warning: ValuesAsNumpy() returned None for hourly '{var_name}'. Skipping.")
             continue

        # FIX: Check if 'values' is actually array-like (has length) before using len()
        if not hasattr(values, '__len__'):
             print(f"    Error: Hourly values for '{var_name}' is not array-like (type: {type(values)}). Skipping.")
             # Optionally log the value: print(f"Value was: {values}")
             continue

        # Now it should be safe to check length
        if len(values) == num_hours_expected:
            hourly_data_dict[var_name] = values
        else:
            print(f"    Warning: Length mismatch for hourly '{var_name}'. Expected {num_hours_expected}, got {len(values)}. Padding with NaN.")
            # Pad with NaN if lengths don't match
            padded_values = np.full(num_hours_expected, np.nan)
            len_to_copy = min(len(values), num_hours_expected)
            padded_values[:len_to_copy] = values[:len_to_copy]
            hourly_data_dict[var_name] = padded_values

    hourly_dataframe_full = pd.DataFrame(data=hourly_data_dict)

    # Convert to local time
    print("\nConverting hourly data to local timezone...")
    try:
        hourly_dataframe_full['date_local'] = hourly_dataframe_full['date_utc'].dt.tz_convert(local_timezone_str)
    except Exception as e:
        print(f"Error converting timezone: {e}. Using UTC time.")
        hourly_dataframe_full['date_local'] = hourly_dataframe_full['date_utc'] # Fallback

    # Filter for working hours (9:00 AM to 4:59 PM)
    print("Filtering hourly data for 9am to 5pm...")
    working_hours_mask = (hourly_dataframe_full['date_local'].dt.hour >= 9) & (hourly_dataframe_full['date_local'].dt.hour < 17)
    hourly_dataframe_9_to_5 = hourly_dataframe_full[working_hours_mask].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Optionally drop the UTC column before saving
    if 'date_utc' in hourly_dataframe_9_to_5.columns:
        hourly_dataframe_9_to_5.drop(columns=['date_utc'], inplace=True)

    print("\nSaving filtered (9am-5pm) hourly data dump to CSV...")
    if not hourly_dataframe_9_to_5.empty:
        try:
            hourly_dataframe_9_to_5.to_csv('hourly_data_dump_9_to_5.csv', index=False, float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S %Z')
            print("Successfully saved hourly_data_dump_9_to_5.csv")
        except Exception as e:
            print(f"Error saving filtered hourly CSV: {e}")
    else:
        print("Warning: No data found within the 9am-5pm time window after filtering.")
else:
    print("\nNo hourly data returned by API.")


print("\n--- Data Dump Script Complete ---")

