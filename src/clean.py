import datetime
import functools as ft
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import pyarrow
import re
import seaborn
import sys
import time

from os import listdir

# Changes directory to src
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


from utils import *

os.chdir("../")

start = time.time()

print("Gathering and dumping data...")
gather_and_dump_data("data/enertalk-dataset/")
print("Loading pickled data...")
raw = load_pickles()
print("***Resampling data to 1 hour intervals...***")
print("Resampling \"total\" data to 1 hour intervals...")
df_resampled_total = resample_df(df=raw, appliance="total")
print("Resampling \"TV\" data to 1 hour intervals...")
df_resampled_TV = resample_df(df=raw, appliance="TV")
print("Resampling \"washing machine\" data to 1 hour intervals...")
df_resampled_washing_machine = resample_df(df=raw, appliance="washing-machine")
print("Resampling \"rice cooker\" data to 1 hour intervals...")
df_resampled_rice_cooker = resample_df(df=raw, appliance="rice-cooker")
print("Resampling \"water purifier\" data to 1 hour intervals...")
df_resampled_water_purifier = resample_df(df=raw, appliance="water-purifier")
print("Resampling \"microwave\" data to 1 hour intervals...")
df_resampled_microwave = resample_df(df=raw, appliance="microwave")
print("Resampling \"kimchi fridge\" data to 1 hour intervals...")
df_resampled_kimchi_fridge = resample_df(df=raw, appliance="kimchi-fridge")
dfs = [
    df_resampled_total, df_resampled_TV, df_resampled_washing_machine,
    df_resampled_rice_cooker, df_resampled_water_purifier, df_resampled_microwave,
    df_resampled_kimchi_fridge
]
print("Merging \"total\" data with \"appliance\" data...")
df_resampled = ft.reduce(lambda left, right: pd.merge(left, right, on="Datetime", how="left"), dfs)
print("Loading weather data...")
df_weather = load_weather_data()
print("Merging household energy consumption data and weather data...")
df_merged = merge_and_clean(df=df_resampled, df_weather=df_weather)
print("Adding additional features...")
df = add_date_features(df_merged)

numeric_columns = df.select_dtypes(include=[np.float, np.int]).columns

df = apply_fill_missing_vals(df, colnames=numeric_columns)

df = on_off_indicator(df=df, appliance="TV")
df = on_off_indicator(df=df, appliance="washing-machine")
df = on_off_indicator(df=df, appliance="rice-cooker")
df = on_off_indicator(df=df, appliance="water-purifier")
df = on_off_indicator(df=df, appliance="microwave")
df = on_off_indicator(df=df, appliance="kimchi-fridge")

print("Saving file...")
df.to_csv("data/intermediate/df_00.csv")

end = time.time()

time_elapsed = end - start

print(f"Total time elapsed to clean input data: {str(datetime.timedelta(seconds = time_elapsed))}")