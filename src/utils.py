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

onoff_indicator_aliases = {
    "TV":"i_TV",
    "washing-machine":"i_washing-machine",
    "rice-cooker":"i_rice-cooker",
    "water-purifier":"i_water-purifier",
    "microwave": "i_microwave",
    "kimchi-fridge":"i_kimchi-fridge"
}

def listdir_remove(path, string='.DS_Store', sort=True):
    # List directories in path and remove string(s) if string(s) exist in directory list
    # Intended for OS-specific files (e.g. .DS_Store on Macs, which is the default)
    listed = listdir(path)
    if string in listed:
        listed.remove(string)
    if sort:
        listed = sorted(listed)
    return listed

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
                
                pickle_out = open("data/intermediate/"+house+"_"+date+".pkl", "wb")
                pickle.dump(raw, pickle_out)
                pickle_out.close()
                
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed: {str(datetime.timedelta(seconds=time_elapsed))}")

def load_pickles():
    files = glob.glob("data/intermediate/00_*.pkl")

    start = time.time()
    df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
    end = time.time()
    print(f"Total time elapsed (to load and concatenate pickled electricity consumption data files): {str(datetime.timedelta(seconds = end - start))}")
    return df

def load_weather_data(weather_path="data/intermediate/weather_clean.csv"):
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

def resample_df(
    df, 
    appliance, 
    ap_dict=active_power_aliases, 
    rp_dict=reactive_power_aliases
):
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

def on_off_indicator(
    df, 
    appliance, 
    threshold=15, 
    onoff_dict=onoff_indicator_aliases, 
    ap_dict=active_power_aliases
):
    if appliance=="total":
        print("Can only provide on/off indicator for an appliance.")
    else:
        i_column_name_str = onoff_dict[appliance]
        ap_column_name_str = ap_dict[appliance]
        df[i_column_name_str] = 1 * (df[ap_column_name_str] > 15)
    return df

def onehotencode(df):
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(df)
    subset = encoder.transform(df)
    return subset

def fill_missing_vals(df, colname):
    """
    First, backward fills missing values with the value exactly one day before. Then, if there are any remaining missing values, forward fills with the value exactly one day after. Function keeps increasing numbers until all missing values are either backward filled or forward filled.

    Rules:
    1. Exclude rows earlier than the earliest df Datetime value plus one day and keep for later
    2. Out of the remaining rows, for days with zero non-NaN values, forward fill from next day
    3. Out of the remaining rows, for days with some non-NaN values, backfill earliest reading or forward fill latest reading
    4. For rows earlier than the earliest df Datetime value plus one day, if 
    """
    # Get earliest Datetime value then add one
    # df_min_datetime_plus_one_day = df.index.min()+datetime.timedelta(days=1)
    # Get latest Datetime value then minus one
    # df_max_datetime_minus_one_day = df.index.max()+datetime.timedelta(days=1)

    df_col_copy = df[colname].copy().reset_index().set_index("Datetime")
    if df_col_copy.isnull().sum()[0] > 0:

        # Get number of missing readings by date
        num_missing = df[colname].isnull().groupby(df["date"]).sum()
        

        # Get dates with at least one missing reading
        dates_lt24 = num_missing[
            (
                num_missing > 0
            ) & (
                num_missing < 24
            )
        ]

        df_lt24 = df.loc[
            df["date"].isin(dates_lt24.index), 
            ["date", colname]
        ]

        df_lt24 = df_lt24.groupby(
            "date"
        ).fillna(
            method="ffill"
        ).fillna(
            method="bfill"
        )

        # Get dates with 24 missing readings
        dates_eq24 = num_missing[
            num_missing == 24
        ]

        df_eq24 = df.loc[
            df["date"].isin(dates_eq24.index), 
            colname
        ]
        
        df_eq24 = df_eq24.reset_index().set_index("Datetime")

        dates = np.unique(df_eq24.index.date)

        for date in dates:
            idx_date_before = df_eq24[
                df_eq24.index.date == date
            ].index - datetime.timedelta(days=1) 

            new_colvals = df.loc[idx_date_before, colname]
            
            new_colvals.index = new_colvals.index + datetime.timedelta(days=1)
            
            if new_colvals.isnull().sum() > 0:
                idx_still_missing_plus_one = new_colvals[
                    new_colvals.isnull()
                ].index + datetime.timedelta(days=1)
                colvals = df.loc[
                    idx_still_missing_plus_one, 
                    colname
                ]

                colvals.index = colvals.index - datetime.timedelta(days=1)
                new_colvals = new_colvals[(new_colvals.index < colvals.index.min())]
                new_colvals = pd.concat([new_colvals, colvals])
                new_colvals = new_colvals.reset_index().set_index("Datetime")
                idx_to_fill = new_colvals.index

                df_eq24.loc[idx_to_fill] = new_colvals
        
        df_filled = pd.concat(
            [
                df_lt24, 
                df_eq24
            ]
        ).sort_index()

        idx_to_fill = df_filled.index

        df_col_copy.loc[idx_to_fill] = df_filled

        if df_col_copy[colname].isnull().sum() > 0:
            df_col_copy = df_col_copy.groupby(
                df_col_copy.index.date
            ).fillna(
                method="ffill"
            ).fillna(
                method="bfill"
            )

    return df_col_copy

def apply_fill_missing_vals(df, colnames):
    for colname in colnames:
        df[colname] = fill_missing_vals(df, colname)
    return df

def CreateLags(df, colname, num_lags):
    for lag in range(1, num_lags+1):
        new_colname = colname + "_(t-" + str(lag) + ")"
        df[new_colname] = df[colname].shift(lag).fillna(0)
    return df

def EncodeCyclical(df, colname):
    max_val = df[colname].max()
    df["sin_"+colname] = np.sin(2 * np.pi * df[colname]/max_val)
    df["cos_"+colname] = np.sin(2 * np.pi * df[colname]/max_val)
    df.drop(colname, axis=1)
    return df


def RangeList(n):
    return list(range(n))

def RemoveItem(item_list, full_list):
    for item in item_list:
        full_list.remove(item)
    return full_list

def RegWrapper(reg_type, reg_value):
    if reg_type == "l1":
        return L1(reg_value)
    if reg_type == "l2":
        return L2(reg_value)
    if reg_type == "l1l2":
        return L1L2(reg_value)