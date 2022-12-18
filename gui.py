import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import leafmap.foliumap as leafmap


#----------------Data------------------#

cwd = os.getcwd()

df = pd.read_csv(cwd + '/historic_and_predicted.csv')
sites = pd.read_csv(cwd + '/snotel_sites.csv')

#sites = pd.read_csv(cwd)

x = sites['Latitude']
y = sites['Longitude']
name = sites['Name']
df['Dates'] = pd.date_range(start = '1985-10-01', freq = '1M', periods = len(df.iloc[:, 0]))

SNOTEL_dict = SNOTEL_dict = {
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
  'ward_creek_3': '848'}

#----------------Data------------------#


with st.sidebar:
    st.image("https://svs.gsfc.nasa.gov/vis/a000000/a004100/a004171/jetstream_uk.0030.jpg")
    st.title("Predicting Precipitation in the Sierra Nevadas")
    choice = st.radio("Navigation", ["Site List", "About", "Download"])

if choice == "Site List":
    st.title("Sites and predicted values")
    option = st.selectbox("Select SNOTEL site:",(SNOTEL_dict))

    # create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Dates'], y=df[f'{option}_predicted'],
                        line=dict(color='royalblue', width=2, dash='dashdot')))

    fig.add_trace(go.Scatter(x=df['Dates'], y=df[option],
                         line=dict(color='firebrick', width=2)))

    st.plotly_chart(fig)

if choice == "About":
    pass

if choice == "Download":
    pass
