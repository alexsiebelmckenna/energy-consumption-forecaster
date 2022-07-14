import datetime
import functools as ft
import numpy as np
import pandas as pd
import pdb
import seaborn as sns

from matplotlib import pyplot as plt

appliances = {
    "TV",
    "washing-machine",
    "rice-cooker",
    "water-purifier",
    "microwave",
    "kimchi-fridge"
}

onoff_indicator_aliases = {
    "TV":"i_TV",
    "washing-machine":"i_washing-machine",
    "rice-cooker":"i_rice-cooker",
    "water-purifier":"i_water-purifier",
    "microwave": "i_microwave",
    "kimchi-fridge":"i_kimchi-fridge"
}

active_power_aliases = {
    "total":"ap_total",
    "TV":"ap_TV",
    "washing-machine":"ap_washing-machine",
    "rice-cooker":"ap_rice-cooker",
    "water-purifier":"ap_water-purifier",
    "microwave": "ap_microwave",
    "kimchi-fridge":"ap_kimchi-fridge"
}


df = pd.read_csv("../data/intermediate/df_00.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")

# One hot encode categorical variables

# Create on/off variables; one hot encode 

def on_off_indicator(df, appliance, threshold=15, onoff_dict=onoff_indicator_aliases, ap_dict=active_power_aliases):

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

numeric_columns = df.select_dtypes(include=[np.float, np.int]).columns

df = apply_fill_missing_vals(df, colnames=numeric_columns)


df = on_off_indicator(df=df, appliance="TV")
df = on_off_indicator(df=df, appliance="washing-machine")
df = on_off_indicator(df=df, appliance="rice-cooker")
df = on_off_indicator(df=df, appliance="water-purifier")
df = on_off_indicator(df=df, appliance="microwave")
df = on_off_indicator(df=df, appliance="kimchi-fridge")

pdb.set_trace()
