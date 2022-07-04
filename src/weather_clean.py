import pandas as pd
import pdb
from utils import *
from datetime import datetime

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

# Check which days have greater than 24 readings
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
grouped_weather_gt24 = weather_gt24.groupby(
    ["date", "hour"]
).apply(
    lambda x: (
        x["minute"]==0
    ).sum()
).reset_index().rename(
    {0:"on_the_hour"}, 
    axis=1
)

pdb.set_trace()


# Check which days have fewer than 24 readings
weather_ft24 = weather[
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


