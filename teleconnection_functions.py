#----------------------LIBRARIES-------------------
import math
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import requests
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
#----------------------VARIABLES-------------------
curr_date = f'{datetime.today().year}-{datetime.today().month}-{datetime.today().day}'
enso_webpage = requests.get('https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices')
pdo_webpage = requests.get('https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat')
ao_webpage = requests.get('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table')
#----------------------FUNCTIONS-------------------


def soap_request_precip(snotel_id):
  body = f'''<?xml version="1.0" encoding="UTF-8"?>
            <SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:q0="http://www.wcc.nrcs.usda.gov/ns/awdbWebService" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              <SOAP-ENV:Body>
                <q0:getData>
                  <stationTriplets>{snotel_id}:CA:SNTL</stationTriplets>
                  <elementCd>PREC</elementCd>
                  <ordinal>1</ordinal>
                  <duration>DAILY</duration>
                  <getFlags>false</getFlags>
                  <beginDate>1985-10-01</beginDate>
                  <endDate>{curr_date}</endDate>
                  <alwaysReturnDailyFeb29>false</alwaysReturnDailyFeb29>
                </q0:getData>
              </SOAP-ENV:Body>
            </SOAP-ENV:Envelope>'''



def data_cleaning(sheet_name):
  sheet = pd.read_excel(data, sheet_name)
  sheet.to_csv(f'{sheet_name}.csv')
  cleaned = pd.read_csv(sheet_name + ".csv", sep = ";", header = None)
  df = cleaned[0].str.split(',', expand=True)
  df = df.iloc[:, 1:]
  df.iloc[1:, 0] = df.iloc[1:, 0].str.replace('-', '').str.replace('"', '')
  new_header = df.iloc[0].str.replace('"', '')
  df = df[1:].applymap(lambda x: x.replace('"', ''))
  df.columns = new_header
  return df, sheet_name

def scrape_enso():
  pd.DataFrame(str(enso_webpage.content).replace("   ", ' ').replace("  ", " ").replace(' ', ';').replace('b', ';').replace('n', ';')\
              .replace("'", "").replace('+', "").replace("MON", 'month').split('\\')).to_csv('enso.csv')
  df = pd.read_csv('enso.csv', delimiter = ";", header = 1)
  df = df.drop(columns = {'0,'})
  df['DATE'] = pd.date_range('1982-01-01', periods = df.shape[0], freq = "1M")
  arr = []
  for i in range(len(df['DATE'])):
    df['DATE'][i] = df['DATE'][i].replace(day = 1)
  
  return df

def scrape_pdo():
  arr = []
  pdo_temp = pd.DataFrame()

  f = str(pdo_webpage.content[18:])
  pd.DataFrame(str(pdo_webpage.content[18:]).replace("   ", ';').replace("  ", ";").replace(' ', ';').replace(',', ';')\
              .replace("'", "").split('\\n')).to_csv("pdo.csv")
  df = pd.read_csv('pdo.csv', delimiter = ";", header = 1)
  df["0,bYear"] = pd.date_range('1854-10-01', periods = df.shape[0], freq = '1Y')
  for i in range(len(df['0,bYear'])):
    df['0,bYear'][i] = df['0,bYear'][i].replace(day = 1).replace(month = 1)
  df = df[df['0,bYear'] >= '1984-10-01']
  for i in range(len(df.iloc[1:, 0])):
    for j in range(1, len(df.columns)):
      arr.append(df.iloc[i, j])

  pdo_temp['pdo_data'] = arr
  pdo_temp['Date'] = pd.date_range('1985-01-01', periods = len(arr), freq = '1M')
  for i in range(len(pdo_temp['Date'])):
    pdo_temp['Date'][i] = pdo_temp['Date'][i].replace(day = 1)

  pdo_temp = pdo_temp[pdo_temp['pdo_data'] != 99.99]

  return pdo_temp

def scrape_ao():
  arr = []
  return_table = pd.DataFrame()


  pd.DataFrame(str(ao_webpage.content).replace("      ", ';').replace("    ", ';').replace("   ", ';').replace("  ", ";").replace(' ', ';').replace(',', ';')\
                .replace("'", "").split('\\n')).to_csv("ao.csv")
  df = pd.read_csv('ao.csv', delimiter = ";", header = 1)
  df["0,b"] = pd.date_range('1950-10-01', periods = df.shape[0], freq = '1Y')
  for i in range(len(df['0,b'])):
    df['0,b'][i] = df['0,b'][i].replace(day = 1).replace(month = 1)
  df = df[df['0,b'].dt.year > 1984]
  df = df.drop(columns = {'Unnamed: 13'})
  for i in range(len(df.iloc[1:, 0])):
    for j in range(1, len(df.columns)):
      arr.append(df.iloc[i, j])

  new_list = [item for item in arr if not(math.isnan(item)) == True]

  return_table['ao_data'] = new_list
  return_table['Date'] = pd.date_range("1985-01-01", periods = len(new_list), freq = "1M")

  for i in range(len(return_table['Date'])):
    return_table['Date'][i] = return_table['Date'][i].replace(day = 1)

  return return_table

def sum_monthly(df, column):
  df.loc[:, [column]] = df.loc[:, [column]].astype(float)
  df = df.groupby(pd.Grouper(key="date", freq="1M")).sum()
  df['d'] = df.index
  df['d'] = df['d'].astype('datetime64[M]')
  return df



def merge_tables(SNOTEL_id):
  z = sum_monthly(soap_request_precip(SNOTEL_id), 'Accumulated_Precipitation')
  x1 = z.merge(temp, left_on='d', right_on = 'd', how = 'left' )
  x1['month'] = x1['d'].dt.month

  pdo = scrape_pdo()
  ao = scrape_ao()

  x1 = x1.merge(pdo, left_on='d', right_on = 'Date', how = 'left')
  x1 = x1.merge(ao, left_on='d', right_on = 'Date', how = 'left')

  return x1



def seasonally_decompose(ff, column):
    sd = seasonal_decompose(ff[column])
    ff[f'{column}_seasonal'] = sd.seasonal

    return ff


def lagged(ff, column):

  df = pd.DataFrame({"Predicted": ff['Accumulated_Precipitation'], "Data": ff[column]})


  for i in np.arange(-12,36):
    df[f"lag_{i}"] = df['Data'].shift(periods=i)

  a = np.array(abs(df.corr().iloc[0, 1:].values))

  ind = 0
  m = 0

  for i in range(len(a)):
    if a[i] > m:
      m = a[i]
      ind = i - 12 - 1

  ff[f'{column}_shifted'] = ff[column].shift(periods = ind)

  return ff



def table_creation(SNOTEL_id):

  df = merge_tables(SNOTEL_id)
  df.index = df['d']
  df = df[:-1]

  for i in seasonal_columns:
    seasonally_decompose(df, i)

  for c in df.columns[1:]:
    lagged(df, c)

  df = df[final_feats]

  df = df.iloc[36:-12]

  return df



def create_fit_save_model(SNOTEL_id):
  df = table_creation(SNOTEL_id)

  callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min'),
    ModelCheckpoint(f'best_{SNOTEL_id}_model.h5', monitor='val_loss', save_best_only=True, verbose=0),
  ]

  scaler = MinMaxScaler()
  scaler = scaler.fit(df)

  scaled_df = scaler.transform(df)

  train_dates = df.index

  trainX = []
  trainY = []

  n_future = 36 #Number of months we want to see into the future
  n_past = 240 #Number of months we want to use to see into the future

  for i in range(n_past, len(scaled_df) - n_future + 1):
    trainX.append(scaled_df[i - n_past:i, 0:scaled_df.shape[1]])
    trainY.append(scaled_df[i + n_future - 1:i + n_future, 1])

  trainX, trainY = np.array(trainX), np.array(trainY)
  
  model = Sequential()
  model.add(LSTM(64, activation = 'relu', input_shape = (trainX.shape[1], trainX.shape[2]), return_sequences = True))
  model.add(LSTM(32, activation = 'relu', return_sequences = False))
  model.add(Dropout(0.2))
  model.add(Dense(trainY.shape[1]))

  model.compile(optimizer = 'adam', loss = 'mse')

  #Lots of epochs, small batch size, function in callbacks stops training when val_loss plateaus preventing overfitting
  history = model.fit(trainX, trainY, epochs = 250, batch_size = 5, validation_split = .2, verbose = 1, callbacks=callbacks)

  forecast_dates = pd.date_range(list(train_dates)[-1], periods = n_future, freq = '1M').tolist()

  forecast = model.predict(trainX[-n_future:])

  forecast_copied = np.repeat(forecast, scaled_df.shape[1], axis = -1)
  y_future = scaler.inverse_transform(forecast_copied)[:, 0]

  for i in range(len(y_future)):
    if y_future[i] < 0:
      y_future[i] = 0

  return y_future
