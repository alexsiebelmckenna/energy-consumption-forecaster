import datetime
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import pdb
import seaborn as sns
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

files = glob.glob('../data/intermediate/00_*.pkl')

start = time.time()
df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
end = time.time()
print(f"Total time elapsed (to load and concatenate pickled electricity consumption data files): {str(datetime.timedelta(seconds = end - start))}")

start = time.time()
df_total = df[df['appliance'] == "total"]
df_total = df_total.rename({"timestamp":"Datetime"}, axis=1)
df_resampled = df_total.set_index("Datetime").resample('1h').mean().fillna(method="ffill")
end = time.time()
print(f"Total time elapsed (to subset and resample electricity consumption data): {str(datetime.timedelta(seconds = end - start))}")

start = time.time()
weather = pd.read_csv("../data/intermediate/weather_clean.csv").drop(
    ["Unnamed: 0"], 
    axis=1
)
weather["Datetime"] = pd.to_datetime(weather["Datetime"])
weather = weather.set_index("Datetime")
end = time.time()
print(f"Total time elapsed (to load weather data): {str(datetime.timedelta(seconds = end - start))}")

df_merged = pd.merge(df_resampled, weather, on="Datetime", how="left")

#df_merged = df_merged[df_merged.index>"2016-11-11 23:00:00"]

df_merged["active_power"] = df_merged["active_power"].astype("float")

df_merged["date"] = df_merged.index.date
df_merged["month"] = df_merged.index.month
df_merged["day"] = df_merged.index.day
df_merged["hour"] = df_merged.index.day



df_merged["temp_int"] = df_merged.groupby("date")["temp_int"].apply(lambda x: x.fillna(method="ffill"))

pdb.set_trace()





