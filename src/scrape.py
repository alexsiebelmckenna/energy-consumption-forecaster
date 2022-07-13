# Scrapes weather website for hourly Seoul weather data

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from utils import *

import datetime
import pandas as pd
import time

start = time.time()

driver = webdriver.Chrome(ChromeDriverManager().install())

dates = listdir_remove("../data/enertalk-dataset/00/")

for date in dates:
    date_str = '-'.join([date[:4], date[4:6], date[6:]])

    print(f"Scraping weather data for {date_str} | File {dates.index(date)+1}/{len(dates)}")

    driver.get("https://www.wunderground.com/history/daily/RKSS/date/"+date_str+"?type=submit%20value%3DView")

    time.sleep(15)

    # Initialize empty DataFrame if first iteration
    if (dates.index(date) == 0):
        daily_obs_table = pd.read_html(
            driver.find_element_by_xpath(
                "//table[@class='mat-table cdk-table mat-sort ng-star-inserted']"
            ).get_attribute(
                "outerHTML"
            )
        )[0].dropna(
            axis=0
        )
        daily_obs_table["date"] = str(date)
    else:
        table = pd.read_html(
            driver.find_element_by_xpath(
                "//table[@class='mat-table cdk-table mat-sort ng-star-inserted']"
            ).get_attribute(
                "outerHTML"
            )
        )[0].dropna(
            axis=0
        )
        table["date"] = str(date)
        daily_obs_table = pd.concat([daily_obs_table, table])

daily_obs_table.to_csv("../data/intermediate/weather_00.csv")

end = time.time()

time_elapsed = end - start

print(f"Time elapsed: {str(datetime.timedelta(seconds=time_elapsed))}")