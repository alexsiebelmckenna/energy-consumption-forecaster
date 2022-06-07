from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import pdb
import time
import urllib3

driver = webdriver.Chrome(ChromeDriverManager().install())

time_list = [] 
temp_list = [] 
dew_point_list = [] 
humidity_list = []
wind_list = []
wind_speed_list = []
wind_gust_list = []
pressure_list = []
precip_list = []
condition_list = []


driver.get("https://www.wunderground.com/history/daily/RKSS/date/2016-11-1"+"?type=submit%20value%3DView")

time.sleep(15)

print(driver.find_element_by_xpath("//table[@class='mat-table cdk-table mat-sort ng-star-inserted']").text)

pdb.set_trace()

#print(driver.find_element_by_xpath("//table[@class='mat-table cdk-table mat-sort ng-star-inserted']").text)

#print(len(cells))

#content = driver.page_source



#time.sleep(10)

#soup = BeautifulSoup(content, 'html.parser')

# class list set
#class_list = set()

# get all tags
##tags = {tag.name for tag in soup.find_all()}
  
# iterate all tags
#for tag in tags:
  
    # find all element of tag
 #   for i in soup.find_all( tag ):
  
        # if tag has attribute of class
#        if i.has_attr( "class" ):
  
#            if len( i['class'] ) != 0:
 #               class_list.add(" ".join( i['class']))
  
#print(class_list)


#for a in soup.findAll('a', href=True, attrs={'class':'_31qSD5'}):
#    name=a.find('div', attrs={'class':'_3wU53n'})
