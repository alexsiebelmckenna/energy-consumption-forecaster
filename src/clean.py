from os import listdir

import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pickle
import pyarrow
import re
import seaborn
import sys
import time

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
    data['timestamp'] = data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.))
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

def gather_data(path):
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
    return raw

raw = gather_data('../data/enertalk-dataset/')
