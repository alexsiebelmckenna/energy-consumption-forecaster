import datetime
import functools as ft
import glob
import matplotlib.pyplot as plt
import matplotlib
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
from utils import *

active_power_aliases = {
    "total":"ap_total",
    "TV":"ap_TV",
    "washing-machine":"ap_washing-machine",
    "rice-cooker":"ap_rice-cooker",
    "water-purifier":"ap_water-purifier",
    "microwave": "ap_microwave",
    "kimchi-fridge":"ap_kimchi-fridge"
}

reactive_power_aliases = {
    "total":"rp_total",
    "TV":"rp_TV",
    "washing-machine":"rp_washing-machine",
    "rice-cooker":"rp_rice-cooker",
    "water-purifier":"rp_water-purifier",
    "microwave": "rp_microwave",
    "kimchi-fridge":"rp_kimchi-fridge"
}

def data_process(path, house, date, file):
    file_num = re.findall(r'(.*?)\_', file)[0]
    filetype = re.findall(r'\_(.*?)\.', file)[0]
    data = pd.read_parquet(
        path + house + '/' + date + '/' + file
    )
    # Format timestamp column
    data['Datetime'] = data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.))
    # Add house column
    data['house'] = house
    # Add appliance column
    if file_num not in ['03', '05']:
        data['appliance'] = filetype
    if file_num == '03':
        data['appliance'] = 'fridge_1'
    if file_num == '05':
        data['appliance'] = 'fridge_2'
    return data

def gather_and_dump_data(path):
    # Start time
    start = time.time()
    
    # List all houses available in path ('00', '01', ...)
    houses = listdir_remove(path)

    for house in houses:
        # List all dates available in path ('20170113', '20170114', ...)
        dates = listdir_remove(path=path+house)

        for date in dates:
            print(f"Trying to read file {date} | File {dates.index(date)+1}/{len(dates)}")
            # List all files available in path ('00_total.parquet.gzip','02_washing-machine.parquet.gzip', ...)
            files = listdir_remove(path=path+house+'/'+date)
            
            for file in files:
            
                # Initialize empty DataFrame if first iteration
                if (files.index(file) == 0):
                    raw = data_process(path=path, house=house, date=date, file=file)
                else:
                    data = data_process(path=path, house=house, date=date, file=file)
                    raw = pd.concat([raw, data])
                
                pickle_out = open("../data/intermediate/"+house+"_"+date+".pkl", "wb")
                pickle.dump(raw, pickle_out)
                pickle_out.close()
                
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed: {str(datetime.timedelta(seconds=time_elapsed))}")

def load_pickles():
    # Changes directory to src
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    files = glob.glob("../data/intermediate/00_*.pkl")

    start = time.time()
    df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
    end = time.time()
    print(f"Total time elapsed (to load and concatenate pickled electricity consumption data files): {str(datetime.timedelta(seconds = end - start))}")
    return df

def load_weather_data(weather_path="../data/intermediate/weather_clean.csv"):
    start = time.time()
    weather = pd.read_csv(weather_path).drop(
        "Unnamed: 0", 
        axis=1
    )
    weather["Datetime"] = pd.to_datetime(weather["Datetime"])
    weather = weather.set_index("Datetime")
    end = time.time()
    print(f"Total time elapsed (to load weather data): {str(datetime.timedelta(seconds = end - start))}")
    return weather

def merge_and_clean(df, df_weather):
    df_merged = pd.merge(df, df_weather, on="Datetime", how="left")
    return df_merged

def resample_df(df, appliance, ap_dict=active_power_aliases, rp_dict=reactive_power_aliases):
    ap_column_name_str = ap_dict[appliance]
    rp_column_name_str = rp_dict[appliance]
    df = df[
        df[
            "appliance"
        ] == appliance
    ].set_index(
        "Datetime"
    ).resample(
        "1h"
    ).mean(
    ).drop(
        "timestamp", 
        axis=1
    ).rename(
        {
            "active_power":ap_column_name_str,
            "reactive_power":rp_column_name_str
        },
        axis=1
    )
    return df

def add_date_features(df):
    df["date"] = df.index.date
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    return df

print("Gathering and dumping data...")
gather_and_dump_data("../data/enertalk-dataset/")
print("Loading pickled data...")
df = load_pickles()
print("***Resampling data to 1 hour intervals...***")
print("Resampling \"total\" data to 1 hour intervals...")
df_resampled_total = resample_df(df=df, appliance="total")
print("Resampling \"TV\" data to 1 hour intervals...")
df_resampled_TV = resample_df(df=df, appliance="TV")
print("Resampling \"washing machine\" data to 1 hour intervals...")
df_resampled_washing_machine = resample_df(df=df, appliance="washing-machine")
print("Resampling \"rice cooker\" data to 1 hour intervals...")
df_resampled_rice_cooker = resample_df(df=df, appliance="rice-cooker")
print("Resampling \"water purifier\" data to 1 hour intervals...")
df_resampled_water_purifier = resample_df(df=df, appliance="water-purifier")
print("Resampling \"microwave\" data to 1 hour intervals...")
df_resampled_microwave = resample_df(df=df, appliance="microwave")
print("Resampling \"kimchi fridge\" data to 1 hour intervals...")
df_resampled_kimchi_fridge = resample_df(df=df, appliance="kimchi-fridge")
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
df_merged = add_date_features(df_merged)
print("Saving file...")
df_merged.to_csv("../data/intermediate/df_00.csv")