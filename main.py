import teleconnect_functions as myfunc

#---------------------Libraries-----------------------

import numpy as np
import pandas as pd
import datetime
import sys, os
import matplotlib.pyplot as plt
import sklearn as sk
import plotly.express as px
import seaborn as sns
import ipywidgets as widgets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy import signal
import statsmodels.api as sm
from matplotlib import figure
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping, ModelCheckpoint

#---------------------SNOTEL dict---------------------

SNOTEL_dict = {
  'adin': '301',
  'blue_lakes': '356',
  'burnside_lake': '1051',
  'carson_pass': '1067',
  'cedar_pass': '391',
  'css_lab': '428',
  'dismal_swamp': '446',
  'ebbetts_pass': '462',
  'echo_peak': '463',
  'fallen_leaf': '473', 
  'hagans_meadow': '508',
  'heavenly_valley': '518',
  'independence_camp': '539',
  'independence_creek': '540',
  'independence_lake': '541',
  'leavitt_lake': '574',
  'leavitt_meadows': '575',
  'lobdell_lake': '587',
  'monitor_pass': '633', 
  'poison_flat': '697',
  'rubicon_2': '724',
  'sonora_pass': '771',
  'spratt_creek': '778',
  'palisades_tahoe': '784',
  'tahoe_city_cross': '809',
  'truckee_2': '834', 
  'virginia_lakes_ridge': '846',
  'ward_creek_3': '848',
  # 'crowder_flat': '977', 
  # 'forestdale_creek': '1049',
  # 'horse_meadow': '1050',
  # 'summit_meadow': '1052',
  # 'state_line': '1258', 
  # 'fredonyer_peak': '1277'
}

#-----------------------Variables---------------------

#SOAP request
url="https://wcc.sc.egov.usda.gov/awdbWebService/services?WSDL"
headers = {'content-type': 'application/soap+xml'}
curr_date = f'{datetime.today().year}-{datetime.today().month}-{datetime.today().day}'

#ENSO webscraper
enso_webpage = requests.get('https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices')
features = ['month','NINO12','NINO3','NINO3.4','NINO4', 'anomaly'] #Maybe unused, check after everything is implemented

#PDO webscraper
pdo_webpage = requests.get('https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat')

#AO webscraper
ao_webpage = requests.get('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table')

#Variables for feature selection/manipulation
features_iloc = [0,3,4,5,6,7,8,9,10,11,15,17] #maybe get rid of this
seasonal_columns = ['ANOM', 'ANOM.1', 'ANOM.2', 'ANOM.3', 'pdo_data', 'ao_data'] #Columns to seasonally decompose
final_feats = ['SWE(in)', 'Accumulated_Precipitation','month_shifted', 'NINO12_shifted','NINO3_shifted','NINO4_shifted',\
               'NINO3.4_shifted','ANOM_seasonal_shifted','ANOM.1_seasonal_shifted','ANOM.2_seasonal_shifted',\
               'ANOM.3_seasonal_shifted','pdo_data_seasonal_shifted','ao_data_seasonal_shifted'] #Final table columns

#Model
n_future = 12

#----------------------Run It--------------------------

myfunc.create_fit_save_model(SNOTEL_dict['adin'])
