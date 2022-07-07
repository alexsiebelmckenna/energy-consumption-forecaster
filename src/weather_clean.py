import pandas as pd
import numpy as np
import pdb
from utils import *
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from utils import *

weather = pd.read_csv(
    "../data/intermediate/weather_00.csv"
).drop(
    ["Unnamed: 0"], 
    axis=1
)

weather["Datetime"] = weather["date"].map(str) + " " + weather["Time"]

weather["Datetime"] = weather["Datetime"].apply(
    lambda x: datetime.strptime(
        x, 
        "%Y%m%d %I:%M %p"
    )
)

# Get the hour and minute values of each timestamp to check which values aren't at the hour mark
weather["hour"] = weather["Datetime"].apply(lambda x: x.hour)
weather["minute"] = weather["Datetime"].apply(lambda x: x.minute)

# Check which days have exactly 24 readings
weather_24 = weather[
    weather[
        "date"
    ].isin(
        weather.groupby(
            "date"
        ).size()[
            weather.groupby(
                "date"
            ).size().values==24
        ].reset_index()[
            "date"
        ].values
    )
]

# Get the readings of the days with greater than 24 readings
weather_gt24 = weather[
    weather[
        "date"
    ].isin(
        weather.groupby(
            "date"
        ).size()[
            weather.groupby(
                "date"
            ).size().values>24
        ].reset_index()[
            "date"
        ].values
    )
]

# See if there's a reading for each hour on the hour for every date
weather_gt24.groupby(["date", "hour"]).size().reset_index().rename({0:"count"}, axis=1)

dates_weather_gt24 = list(weather_gt24["date"].unique())

for date in dates_weather_gt24:
    if dates_weather_gt24.index(date) == 0:
        min_datetime = weather_gt24[weather_gt24["date"] == date]["Datetime"].min()
        date_range_weather_gt24_df = pd.date_range(min_datetime, periods = 24, freq='1h').to_series()
    else:
        min_datetime = weather_gt24[weather_gt24["date"] == date]["Datetime"].min()
        date_range_weather_gt24_df = date_range_weather_gt24_df.append(pd.date_range(min_datetime, periods = 24, freq='1h').to_series())


date_range_weather_gt24_df = pd.DataFrame(date_range_weather_gt24_df.reset_index(drop=True).rename({"index":"Datetime"}, axis=1)).rename({0:"Datetime"}, axis=1)

date_range_weather_gt24_df["date"] = date_range_weather_gt24_df["Datetime"].dt.date.apply(lambda x: int(str(x).replace('-', '')))
date_range_weather_gt24_df["hour"] = date_range_weather_gt24_df["Datetime"].apply(lambda x: x.hour)
date_range_weather_gt24_df["minute"] = date_range_weather_gt24_df["Datetime"].apply(lambda x: x.minute)


date_range_weather_gt24_df = pd.merge(date_range_weather_gt24_df, weather_gt24, how="outer", on=["date", "hour"], indicator=True)

# Hourly readings missing from weather_gt24
# In the case of House 00, only one: 2016-11-06 17:00:00
readings_to_scrape_gt24 = date_range_weather_gt24_df[
    date_range_weather_gt24_df["_merge"] == "left_only"
].rename(
    {
        "Datetime_x":"Datetime"
    }, 
    axis=1
)["Datetime"]

subset_weather_gt24 = weather_gt24.drop_duplicates(subset=["date", "hour"], keep="first")

# Get the readings of the days with less than 24 readings
weather_lt24 = weather[
    weather[
        "date"
    ].isin(
        weather.groupby(
            "date"
        ).size()[
            weather.groupby(
                "date"
            ).size().values<24
        ].reset_index()[
            "date"
        ].values
    )
]

dates_lt24 = weather_lt24["date"].unique()

if len(dates_lt24) == 1:
    date_lt24 = dates_lt24[0]
    min_datetime = weather_lt24[weather_lt24["date"]==date_lt24]["Datetime"].min()
    date_range_weather_lt24_df = pd.DataFrame(pd.date_range(min_datetime, periods = 24, freq='1h')).rename({0:"Datetime"}, axis=1)
    date_range_weather_lt24_df = pd.merge(date_range_weather_lt24_df, weather_lt24, on=["Datetime"], how="outer", indicator=True)
    readings_to_scrape_lt24 = date_range_weather_lt24_df[date_range_weather_lt24_df["_merge"]=="left_only"]["Datetime"]
else:
    for date in dates_lt24:
        if dates_lt24.index(date) == 0:
            min_datetime = weather_lt24[weather_lt24["date"]==date_lt24]["Datetime"].min()
            date_range_weather_lt24 = pd.date_range(min_datetime, periods = 24, freq='1h').to_series
        else:
            min_datetime = weather_lt24[weather_lt24["date"] == date]["Datetime"].min()
            date_range_weather_lt24 = date_range_weather_lt24.append(pd.date_range(min_datetime, periods = 24, freq='1h').to_series())
    date_range_weather_lt24_df = pd.DataFrame(date_range_weather_lt24).rename({0:"Datetime"}, axis=1)
    date_range_weather_lt24_df = pd.merge(date_range_weather_lt24_df, weather_lt24, on=["Datetime"], how="outer", indicator=True)
    readings_to_scrape_lt24 = date_range_weather_lt24_df[date_range_weather_lt24_df["_merge"]=="left_only"]["Datetime"]

readings_to_scrape = pd.concat([readings_to_scrape_lt24, readings_to_scrape_gt24]).reset_index(drop=True)

weather = weather_24.append(weather_lt24).append(subset_weather_gt24)

# Start cleaning the weather data

# Subset data to get around missing data issue
weather = weather[weather["Datetime"]>"2016-11-11 23:00:00"]

weather["wind_speed_int"] = weather["Wind Speed"].astype("str").str.extractall("(\d+)").unstack().fillna('').sum(axis=1).astype(int)
weather["pressure_float"] = weather["Pressure"].astype("str").str.extract(r"[+-]?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?")[0]
weather["temp_int"] = weather["Temperature"].astype("str").str.extractall("(\d+)").unstack().fillna('').sum(axis=1).astype(int)
weather["dew_point_int"] = weather["Temperature"].astype("str").str.extractall("(\d+)").unstack().fillna('').sum(axis=1).astype(int)
weather["humidity_int"] = weather["Humidity"].astype("str").str.extractall("(\d+)").unstack().fillna('').sum(axis=1).astype(int)
weather["wind_gust_int"] = weather["Wind Gust"].astype("str").str.extractall("(\d+)").unstack().fillna('').sum(axis=1).astype(int)
weather["wind_direction_cat"] = weather["Wind"]
weather["condition_cat"] = weather["Condition"]

subset_weather = weather[
    [
        "Datetime", "hour", "temp_int", "dew_point_int", "humidity_int", "condition_cat", "wind_direction_cat", "wind_speed_int", 
        "wind_gust_int", "pressure_float"
    ]
]

subset_weather.to_csv("../data/intermediate/weather_clean.csv")





