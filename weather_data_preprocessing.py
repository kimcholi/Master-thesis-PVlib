"""
Adventdalen weather data from UNIS weather station
- one file for temperature, wind speed (hourly)
    LT1_gr_C_Avg = temperature, VH2_10_min = wind speed
- one file for radiation (5 mins)
    CM3_opp_Wpm2_Avg = ghi
- both files are in .dat format, therefore convert to .csv format
- as also DNI and DHI is needed for pvlib, these will be modeled using a pvlib function
"""

import os  # for getting environment variables
import pathlib  # for finding the example dataset
import pvlib
import pandas as pd  # for data wrangling
import matplotlib.pyplot as plt  # for visualization
import numpy as np
from datetime import date,timedelta
from pvlib import solarposition, location
from pvlib import irradiance, tools

#hourly weather data (read csv file) 
df1 = pd.read_csv('Adventdalen_Hour.csv', usecols = ['TIMESTAMP', 'LT1_gr_C_Avg', 'VH2_10_min'], low_memory = False)
mask = (df1['TIMESTAMP'] >= '2022-01-01') & (df1['TIMESTAMP'] <= '2023-01-01') #adjust for timeframe
df1 = df1.loc[mask]
df1 = df1.reset_index(drop=True) #set timestamp as the index
df1['TIMESTAMP'] = pd.to_datetime(df1['TIMESTAMP'])
df1 = df1.set_index('TIMESTAMP')
df1 = df1.astype(np.float16) #change type of measurements to float

#5 mins radiation data
df2 = pd.read_csv('Adventdalen_New_Fem_minutt.csv', usecols = ['TIMESTAMP', 'CM3_opp_Wpm2_Avg'], low_memory = False)
mask = (df2['TIMESTAMP'] >= '2022-01-01')  & (df2['TIMESTAMP'] <= '2023-01-01') 
df2 = df2.loc[mask]
df2 = df2.reset_index(drop=True)
df2['TIMESTAMP'] = pd.to_datetime(df2['TIMESTAMP'])
df2 = df2.set_index('TIMESTAMP')
df2 = df2.astype(np.float16)
df2 = df2.CM3_opp_Wpm2_Avg.resample('H').mean() #resample 5 min frequency to hourly frequency
df2 = df2.to_frame()

#combine both datasets
df3 = df1.join(df2)
df3 = df3.rename(columns={'LT1_gr_C_Avg': 'temp_air', 'VH2_10_min': 'wind_speed', 'CM3_opp_Wpm2_Avg' : 'ghi'}) #similar labels to TMY

#solar position values
times = pd.date_range('2022-01-01', '2022-12-31 23:00:00', freq='H', tz='Europe/Oslo') #1T is 1 min frequency, H is hourly; use regarded timeframe
loc = location.Location(latitude = 78.22, longitude = 15.55, tz = times.tz) #use exact location
sp = loc.get_solarposition(times)

#model DNI and DHI from GHI, here: DIRINT model 
zen = sp.zenith
ghi = df3.ghi.tz_localize('Etc/GMT-1') #so it's the same timezone
dni = irradiance.dirint(ghi, zen, times, min_cos_zenith=tools.cosd(zen.max()), max_zenith=zen.max())
irrad = pd.DataFrame()
irrad['ghi'] = ghi
irrad.loc[irrad.ghi <50, 'ghi'] = 0 # noise - error on measuring device
irrad['dni'] = dni
dhi = ghi-(irrad.dni*tools.cosd(zen))
irrad['dhi'] = dhi

#put together all datasets and save as .csv file
df3 = df3.drop(columns='ghi')
df3 = df3.tz_localize('Etc/GMT+0') #only for first run
df_tmy = df3.join(irrad)
df_tmy['dni'] = df_tmy['dni'].astype('float32') #to be compatible with get_total_irradiance() 
df_tmy['dhi'] = df_tmy['dhi'].astype('float32') #to be compatible with get_total_irradiance() 
df_tmy.to_csv('weather_data_adv.csv')
