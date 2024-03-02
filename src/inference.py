# -------------------------------
# https://open-meteo.com/ less than 10000 API calls per day
# https://open-meteo.com/en/docs
# https://api.open-meteo.com/v1/forecast?latitude=50.927518&longitude=5.384517&minutely_15=apparent_temperature,precipitation_probability&wind_speed_unit=ms&timezone=Europe%2FBerlin&forecast_days=2&models=best_match
# 
# -------------------------------
# Latitude: 50.927518
# Longitude: 5.384517

import numpy as np # pip install numpy
import pandas as pd # pip install pandas
import openmeteo_requests # pip install openmeteo-requests
import requests_cache # pip install requests-cache
from retry_requests import retry # pip install retry-requests


def weather_forecast_data(start_datetime_incl: np.datetime64, forecast_horizon: np.timedelta64, freq: str = "10T",
		latitude: float = 50.927518, longitude: float = 5.384517) -> pd.DataFrame:
	
	end_datetime_incl = start_datetime_incl + forecast_horizon

	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in minutely_15 is important to assign them correctly below
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"minutely_15": ["apparent_temperature", "precipitation"],
		"wind_speed_unit": "ms",
		"timezone": "auto", # determine timezone automatically from latitude and longitude
		"start_minutely_15": str(start_datetime_incl),
		"end_minutely_15": str(end_datetime_incl),
		#"forecast_days": 1,
		"models": "best_match"
	}
	responses = openmeteo.weather_api(url, params=params) # can raise an exception, TODO: catch it

	# Process first location. (Add a for-loop for multiple locations or weather models; here not necessary.)
	response = responses[0]
	timezone = response.Timezone().decode("utf-8")
	#print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	#print(f"Elevation {response.Elevation()} m asl")
	#print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	#print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process minutely_15 data. The order of variables needs to be the same as requested.
	minutely_15 = response.Minutely15()
	minutely_15_apparent_temperature = minutely_15.Variables(0).ValuesAsNumpy()
	minutely_15_precipitation = minutely_15.Variables(1).ValuesAsNumpy()

	minutely_15_data = {"DateTime": pd.date_range(
		start = pd.to_datetime(minutely_15.Time(), unit = "s", utc=True).tz_convert(timezone),
		end = pd.to_datetime(minutely_15.TimeEnd(), unit = "s", utc=True).tz_convert(timezone),
		freq = pd.Timedelta(seconds = minutely_15.Interval()),
		inclusive = "left"
	)}
	minutely_15_data["apparent_temperature"] = minutely_15_apparent_temperature
	minutely_15_data["precipitation"] = minutely_15_precipitation

	df = pd.DataFrame(data = minutely_15_data)
	df.set_index("DateTime", inplace=True)

	# Resample 15 minute data to chosen frequency
	df = df.resample("1T").ffill() # forward fill values so they are repeated every minute in a 15 minute interval
	df = df.resample(freq).mean() # resample to chosen frequency via mean substitution 

	return df

start_datetime_incl = np.datetime64("2024-03-02T16:20:00")
forecast_horizon = np.timedelta64(2, 'h')
df = weather_forecast_data(start_datetime_incl, forecast_horizon, freq="10T")

print(df)