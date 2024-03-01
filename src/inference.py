# -------------------------------
# https://open-meteo.com/ less than 10000 API calls per day
# https://open-meteo.com/en/docs#hourly=apparent_temperature,precipitation_probability&wind_speed_unit=ms&timezone=Europe%2FBerlin&forecast_days=3&models=best_match
# https://api.open-meteo.com/v1/forecast?latitude=50.927518&longitude=5.384517&minutely_15=apparent_temperature,precipitation_probability&wind_speed_unit=ms&timezone=Europe%2FBerlin&forecast_days=2&models=best_match
# 
# -------------------------------
# Latitude: 50.927518
# Longitude: 5.384517

import openmeteo_requests # pip install openmeteo-requests

import requests_cache # pip install requests-cache
import pandas as pd
from retry_requests import retry # pip install retry-requests

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 50.927518,
	"longitude": 5.384517,
	"minutely_15": ["apparent_temperature", "precipitation"],
	"wind_speed_unit": "ms",
	"timezone": "Europe/Berlin",
	"forecast_days": 2,
	"models": "best_match"
}
responses = openmeteo.weather_api(url, params=params) # can raise an exception, TODO: catch it

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process minutely_15 data. The order of variables needs to be the same as requested.
minutely_15 = response.Minutely15()
minutely_15_apparent_temperature = minutely_15.Variables(0).ValuesAsNumpy()
minutely_15_precipitation = minutely_15.Variables(1).ValuesAsNumpy()

minutely_15_data = {"date": pd.date_range(
	start = pd.to_datetime(minutely_15.Time(), unit = "s", utc = True),
	end = pd.to_datetime(minutely_15.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = minutely_15.Interval()),
	inclusive = "left"
)}
minutely_15_data["apparent_temperature"] = minutely_15_apparent_temperature
minutely_15_data["precipitation"] = minutely_15_precipitation

minutely_15_dataframe = pd.DataFrame(data = minutely_15_data)
print(minutely_15_dataframe)

# TODO: resample to every 10 minutes after an hour using mean()