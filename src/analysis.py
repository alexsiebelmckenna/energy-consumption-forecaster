import datetime
import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

files = glob.glob('../data/intermediate/00_*.pkl')

start = time.time()
df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
end = time.time()
print(f"Total time elapsed: {str(datetime.timedelta(seconds = end - start))}")

df_subset = df[df['appliance'] == "total"]
