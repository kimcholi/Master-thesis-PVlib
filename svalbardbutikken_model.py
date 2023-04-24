##### SETUP #####

#import libraries
import pandas as pd #for data handling
import numpy as np #for data handling
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns #for visualization
import os  # for getting environment variables
import pathlib  # for finding datasets
import csv # for csv file handling
from sklearn.metrics import mean_absolute_error, mean_squared_error #for statistics


#pvlib imports
import pvlib
from pvlib.pvsystem import PVSystem, FixedMount, Array
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.pvsystem import dc_ohms_from_percent, dc_ohmic_losses, pvwatts_losses
from pvlib import shading
from pvlib import irradiance, tools
from pvlib.pvsystem import dc_ohms_from_percent, dc_ohmic_losses, pvwatts_losses

#temperature model
temperature_model_parameters_Coop = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
temperature_model_parameters_Nordpolet = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['insulated_back_glass_polymer']

#module and inverter specifications
cec_modules = pvlib.pvsystem.retrieve_sam(path='https://raw.githubusercontent.com/NREL/SAM/patch/deploy/libraries/CEC%20Modules.csv')
cec_inverters1 = pvlib.pvsystem.retrieve_sam('cecinverter')
cec_inverters2 = pvlib.pvsystem.retrieve_sam(path = 'https://raw.githubusercontent.com/NREL/SAM/patch/deploy/libraries/CEC%20Inverters.csv')

#Svalbardbutikken
module_coop_400 = cec_modules['Trina_Solar_TSM_400DE09_08'] #Trina Solar TSM 400 DE09.08
module_coop_395 = cec_modules['Trina_Solar_TSM_395DE09_08'] #Trina Solar TSM 395 DE09.08

#PV module specific
coop_gamma_pdc = -0.0034
coop_400_pdc0 = 400
coop_395_pdc0 = 395

inverter_coop_1 = cec_inverters1['SMA_America__STP50_US_40__480V_'] # SMA Tripower Core 1 (2x)
inverter_coop_2 = cec_inverters2['Ginlong_Technologies_Co___Ltd___Solis_100K_5G_US__480V_'] # SMA Tripower Core 2 (1x), replaced by Ginglong Solis 100K 5G

#input weather data (temp, wind speed, ghi, dhi, dni,albedo)
#04-07-22 until 31-10-22
#weather station
df_tmy = pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/weather data/weather data_270122/weather_data_adv.csv')
df_tmy = df_tmy.reset_index(drop=True) #set timestamp as the index
mask = (df_tmy['TIMESTAMP'] >= '2022-07-05')  & (df_tmy['TIMESTAMP'] <= '2022-10-31')
df_tmy = df_tmy.loc[mask]
df_tmy['TIMESTAMP'] = pd.to_datetime(df_tmy['TIMESTAMP'])
df_tmy = df_tmy.set_index('TIMESTAMP')

#satellite data
df_tmy_sat = pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/satellite data/weather_data_coop.csv')
df_tmy_sat = df_tmy_sat.reset_index(drop=True) #set timestamp as the index
mask = (df_tmy_sat['time'] >= '2022-07-05')  & (df_tmy_sat['time'] <= '2022-10-31')
df_tmy_sat = df_tmy_sat.loc[mask]
df_tmy_sat['time'] = pd.to_datetime(df_tmy_sat['time'])
df_tmy_sat = df_tmy_sat.set_index('time')

#time zone
tz = 'Europe/Oslo'

#time and location
times_coop = pd.date_range('2022-07-05 00:00:00', '2022-10-30 23:00:00', closed='left', freq='H', tz=tz) #adapt to measured ac generation time range
loc_coop = Location(latitude = 78.21816, longitude = 15.64022, altitude = 36) #29 m a.s.l., roughly 7 m high (Mons Ole Sellevold)
solar_position_coop = loc_coop.get_solarposition(times_coop, method = 'ephemeris')
solar_position_coop.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/weather data/albedo/albedo_check/solpos.csv')

#Shading map/losses
shading_coop = np.load('svm_Svalbardbutikken.npy')
shading_coop = pd.DataFrame(shading_coop)
shading_coop.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/map_shading_Svalbardbutikken.csv')

#weather station
solar_position_coop["shade"] = shading_coop.stack()[pd.MultiIndex.from_arrays([np.clip(solar_position_coop["zenith"].astype(int), None, 90), solar_position_coop["azimuth"].astype(int)])].values #implement a new column to solar position with the boolean values True and False depending on the sun's position and in relation to shading map
solar_position_coop.loc[solar_position_coop["zenith"] > 90, "shade"] = True  #if zenith is over 90 degrees set the shade to True
df_tmy['shade'] = solar_position_coop['shade'] #apply shade to weather data
df_tmy['shade'] = df_tmy['shade'].fillna(False).astype(bool) #replace NaN values with False as no information is known
df_tmy['dni'] = df_tmy['dni'].where(~df_tmy['shade'], 0) #set dni to 0 where shade is True

#applied Varga & Mayer (2021)
psi = shading.masking_angle_passias(13, gcr=0.484784831817805) #calculated GCR and tilt angle to calculate the masking angle by Passias 1984
alpha_s = tools.tand(tools.tand(solar_position_coop.elevation)/tools.cosd(solar_position_coop.azimuth)) #calculate the the southern projection of the elevation angle of the Sun
alpha_s = 1/alpha_s #southern projection of the elevation angle of the Sun
solar_position_coop.loc[alpha_s > psi, "direct_shade"] = True  #if the southern projection of the elevation angle of the Sun < masking angle, the surface is shaded
solar_position_coop.loc[alpha_s <= psi, "direct_shade"] = False #if the southern projection of the elevation angle of the Sun > masking angle, the surface is not shaded
df_tmy['direct_shade'] = solar_position_coop['direct_shade']
df_tmy['direct_shade'] = df_tmy['direct_shade'].fillna(False).astype(bool) #replace NaN values with False as no information is known
df_tmy['dni'] = df_tmy['dni'].where(~df_tmy['direct_shade'], 0) #set dni to 0 where shade is True

solar_position_coop.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/solar_position_coop_ws.csv')

#satellite
solar_position_coop["shade"] = shading_coop.stack()[pd.MultiIndex.from_arrays([np.clip(solar_position_coop["zenith"].astype(int), None, 90), solar_position_coop["azimuth"].astype(int)])].values
solar_position_coop.loc[solar_position_coop["zenith"] > 90, "shade"] = True
df_tmy_sat['shade'] = solar_position_coop['shade'] #apply shade to weather data
df_tmy_sat['shade'] = df_tmy_sat['shade'].fillna(False).astype(bool)
df_tmy_sat['dni'] = df_tmy_sat['dni'].where(~df_tmy_sat['shade'], 0)
solar_position_coop.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/solar_position_coop_sat.csv')

#measured power outputs
df_coop= pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/Svalbardbutikken_measurements_hourly.csv') #hourly measurements
df_coop = df_coop.reset_index(drop=True) #set timestamp as the index
df_coop['Time'] = pd.to_datetime(df_coop['Time'])
df_coop['Power'] = df_coop['Power']
df_coop = df_coop.set_index('Time')

#satellite power outputs
df_sat = pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/ac_svalbardbutikken_sat.csv')
df_sat = df_sat.reset_index(drop=True) #set timestamp as the index
mask = (df_sat['time'] >= '2022-07-05 00:00')  & (df_sat['time'] <= '2022-10-31')
df_sat = df_sat.loc[mask]
df_sat['time'] = pd.to_datetime(df_sat ['time'])
df_sat['ac_sum'] = df_sat['ac_sum']
df_sat = df_sat.set_index('time')
df_sat = df_sat.dropna()

##### PV SYSTEM SETUP #####

loc_coop = Location(latitude = 78.21147, longitude = 15.65409)

#azimuths and tilts
azimuth_coop_nw = 308 #for building 1,2 & roof
azimuth_coop_se = 128  #for building 1,2 & roof
tilt_coop_b = 13  #tilt for arbitrary building 1,2
tilt_coop_r = 27  #tilt for roof area of building 2


#mount buildings
mount_b_nw = FixedMount(tilt_coop_b, azimuth_coop_nw)
mount_b_se = FixedMount(tilt_coop_b, azimuth_coop_se)

#mount roof area
mount_r_nw = FixedMount(tilt_coop_r, azimuth_coop_nw)
mount_r_se = FixedMount(tilt_coop_r, azimuth_coop_se)

dc_ohmic_percent = 1.5 #dc ohmic losses set to default of 1.5%
array_kwargs_400 = dict(module_parameters = module_coop_400,
                    temperature_model_parameters = temperature_model_parameters_Coop, array_losses_parameters=dict(dc_ohmic_percent = dc_ohmic_percent))

array_kwargs_395 = dict(module_parameters = module_coop_395,
                    temperature_model_parameters = temperature_model_parameters_Coop, array_losses_parameters=dict(dc_ohmic_percent = dc_ohmic_percent))

#arrays (must have same orientation and tilt)
array_r1_se = Array(mount_r_se, name='Roof1_SE', modules_per_string = 20, strings = 3, **array_kwargs_395)
array_r2_se = Array(mount_b_se, name='Roof2_SE', modules_per_string = 20, strings = 2, **array_kwargs_400)
array_r22_se = Array(mount_b_se, name='Roof2_SE', modules_per_string = 21, strings = 1, **array_kwargs_400)
array_r2_nw = Array(mount_b_nw, name='Roof2_NW', modules_per_string = 20, strings = 2, **array_kwargs_400)
array_r22_nw = Array(mount_b_nw, name='Roof2_NW', modules_per_string = 21, strings = 1, **array_kwargs_400)

array_core1_1 = [array_r1_se, array_r2_se, array_r22_se, array_r2_nw, array_r22_nw]

array_r11_se = Array(mount_r_se, name='Roof1_SE', modules_per_string = 20, strings = 2, **array_kwargs_395)
array_r1_nw = Array(mount_r_nw, name='Roof1_NW', modules_per_string = 18, strings = 2, **array_kwargs_395)
array_r11_nw = Array(mount_r_nw, name='Roof1_NW', modules_per_string = 20, strings = 4, **array_kwargs_395)

array_core1_2 = [array_r11_se, array_r1_nw, array_r11_nw]

array_r23_nw = Array(mount_b_nw, name='Roof2_NW', modules_per_string = 20, strings = 2, **array_kwargs_400)
array_r24_nw = Array(mount_b_nw, name='Roof2_NW', modules_per_string = 21, strings = 1, **array_kwargs_400)
array_r22_se = Array(mount_b_se, name='Roof2_SE', modules_per_string = 20, strings = 2, **array_kwargs_400)
array_r23_se = Array(mount_b_se, name='Roof2_SE', modules_per_string = 21, strings = 1, **array_kwargs_400)
array_r3_nw = Array(mount_b_nw, name='Roof3_NW', modules_per_string = 22, strings = 5, **array_kwargs_400)
array_r33_nw = Array(mount_b_nw, name='Roof3_NW', modules_per_string = 16, strings = 2, **array_kwargs_400)
array_r3_se = Array(mount_b_se, name='Roof3_SE', modules_per_string = 22, strings = 5, **array_kwargs_400)
array_r33_se = Array(mount_b_se, name='Roof3_SE', modules_per_string = 16, strings = 2, **array_kwargs_400)

array_core2 = [array_r23_nw, array_r24_nw, array_r22_se, array_r23_se, array_r3_nw, array_r33_nw, array_r3_se, array_r33_se]

#define the three systems (one for each inverter)
"""
mismatch losses:
core1_1: 0.04808554958649436 = 4.808554958649436 %
core1_2: 0.041619228168883304 = 4.1619228168883304 %
core2: 0.06112678097561153 = 6.112678097561153 %
"""

system_core1_1 = PVSystem(arrays = array_core1_1, strings_per_inverter = 9, inverter_parameters = inverter_coop_1, losses_parameters =  dict(mismatch=4.808554958649436))
system_core1_2 = PVSystem(arrays = array_core1_2, strings_per_inverter = 8, inverter_parameters = inverter_coop_1, losses_parameters =  dict(mismatch=4.1619228168883304))
system_core2 = PVSystem(arrays = array_core2, strings_per_inverter = 20, inverter_parameters = inverter_coop_2, losses_parameters =  dict(mismatch=6.112678097561153))

#Interrow-shading losses
psi = shading.masking_angle_passias(tilt_coop_b, gcr=0.484784831817805)
shading_loss = shading.sky_diffuse_passias(psi)
df_tmy['dhi'] = df_tmy['dhi']*(1-shading_loss) #weather station
df_tmy_sat['dhi'] = df_tmy_sat['dhi']*(1-shading_loss) #satellite data
df_tmy.loc[(df_tmy['dhi'] == 0) & (df_tmy['dni'] == 0), 'ghi'] = 0 #if both dni and dhi are 0, set ghi 0 - reflection is neglected

#define the three ModelChains (one for each inverter)
mc_core1_1 = ModelChain(system_core1_1, loc_coop, clearsky_model = None, transposition_model = 'haydavies', solar_position_method = 'pyephem', airmass_model = 'kastenyoung1989', aoi_model='physical', spectral_model='no_loss',  dc_ohmic_model = 'dc_ohms_from_percent', losses_model = 'pvwatts', name = 'core1_1')
mc_core1_2 = ModelChain(system_core1_2, loc_coop, clearsky_model = None, transposition_model = 'haydavies', solar_position_method = 'nrel_c', airmass_model = 'kastenyoung1989', aoi_model='physical', spectral_model='no_loss', dc_ohmic_model = 'dc_ohms_from_percent', losses_model = 'pvwatts', name = 'core1_2')
mc_core2 = ModelChain(system_core2, loc_coop, clearsky_model = None, transposition_model = 'haydavies', solar_position_method = 'ephemeris', airmass_model = 'kastenyoung1989', aoi_model='physical', spectral_model='no_loss', dc_ohmic_model = 'dc_ohms_from_percent', losses_model = 'pvwatts', name = 'core2')

#run the ModelChains
mc_core1_1.run_model(df_tmy)
mc_core1_2.run_model(df_tmy)
mc_core2.run_model(df_tmy)

##### AC OUTPUT MODELING #####
#AC is in W, hourly values

#system ac outputs
ac_system = pd.DataFrame()
ac_system['ac_core1_1'] = mc_core1_1.results.ac #AC is in W, hourly values
ac_system.loc[ac_system.ac_core1_1 == -15.0216, 'ac_core1_1'] = 0 #set neg. power output to 0
ac_system['ac_core1_2'] = mc_core1_2.results.ac
ac_system.loc[ac_system.ac_core1_2 == -15.0216, 'ac_core1_2'] = 0
ac_system['ac_core2'] = mc_core2.results.ac
ac_system.loc[ac_system.ac_core2 == -1, 'ac_core2'] = 0
ac_system['ac_sum'] = ac_system.sum(axis=1)
ac_system['ac_measured'] = df_coop['Power'] #import measured power output for ac measured
ac_system['ac_modelerror'] = ac_system['ac_sum'] - ac_system['ac_measured']
ac_system.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/ac_svalbardbutikken_wsadv.csv')

#plot of model outputs
#weekly model output
fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
(ac_system.ac_sum.resample('W').sum()/1000000).plot(label='Inverter (Adventdalen Weather Station)', linestyle ='dotted', linewidth= 0.5, color = 'crimson', ax=ax) #divide by 1000000 to get MWh
(df_coop.Power.resample('W').sum()/1000000).plot(label='Measurements', linewidth= 0.5, color = 'black', ax=ax)
(df_sat.ac_sum.resample('W').sum()/1000000).plot(label='Inverter (ERA5 Satellite Data)', linestyle = 'dashed',  linewidth= 0.5, color = 'teal', ax=ax)
ax.set_xlim('2022-07-04', '2022-11-01')
ax.set_ylabel('Energy Output [MWh]', fontsize=8)
ax.set_xlabel('Months', fontsize=8)
ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)
ax.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize =8)
plt.savefig('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/adjusted/svalbardbutikken_ws_sat_week.png', dpi=300, bbox_inches='tight')
#daily model output
fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
(ac_system.ac_sum.resample('D').sum()/1000000).plot(label='Inverter (Adventdalen Weather Station)', linestyle ='dotted', color = 'crimson', linewidth=0.5, ax=ax)
(df_coop.Power.resample('D').sum()/1000000).plot(label='Measurements', color = 'black', linewidth=0.5, ax=ax)
(df_sat.ac_sum.resample('D').sum()/1000000).plot(label='Inverter (ERA5 Satellite Data)', linestyle = 'dashed', color = 'teal', linewidth=0.5, ax=ax)
ax.set_xlim('2022-07-04', '2022-11-01')
ax.set_ylabel('Energy Output [MWh]', fontsize=8)
ax.set_xlabel('Months', fontsize=8)
ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)
ax.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize =8)
plt.savefig('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/adjusted/svalbardbutikken_ws_sat_day.png', dpi=300, bbox_inches='tight')
#hourly model output
fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
(ac_system.ac_sum/1000).plot(label='Inverter (Adventdalen Weather Station)', linestyle ='dotted', linewidth=0.5, color = 'teal', ax=ax) #in kW
(df_coop.Power/1000).plot(label='Measurements', color = 'black', linewidth=0.5, ax=ax)
ax.set_xlim('2022-07-04', '2022-11-01')
ax.set_ylabel('Energy Output [MW]', fontsize=8)
ax.set_xlabel('Months', fontsize=8)
ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)
ax.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize =8)
plt.savefig('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/adjusted/svalbardbutikken_ws_hourly.png', dpi=300, bbox_inches='tight')

##### STATISTICAL MEASURES #####
ac_system = ac_system.dropna()
RMAE_sat_year = mean_absolute_error(df_sat['ac_measured'], df_sat['ac_sum'])/np.mean(df_sat['ac_measured'])
RMAE_ws_year = mean_absolute_error(ac_system['ac_measured'], ac_system['ac_sum'])/np.mean(ac_system['ac_measured'])
RRMSE_sat_year = mean_squared_error(df_sat['ac_measured'], df_sat['ac_sum'], squared=False)/np.mean(df_sat['ac_measured'])
RRMSE_ws_year = mean_squared_error(df_sat['ac_measured'], ac_system['ac_sum'], squared=False)/np.mean(ac_system['ac_measured'])

#write statistics to csv
file = open('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/RMAE_RRMSE_Svalbardbutikken_year.csv', 'w', newline = '')
with file:
    header = ['Weather Data', 'RMAE', 'RRMSE']
    writer = csv.DictWriter(file, fieldnames = header)

    writer.writeheader()
    writer.writerow({'Weather Data': 'Sat','RMAE': RMAE_sat_year, 'RRMSE': RRMSE_sat_year})
    writer.writerow({'Weather Data': 'WS','RMAE': RMAE_ws_year, 'RRMSE': RRMSE_ws_year})

RMAE_ws_july = mean_absolute_error(ac_system['ac_measured'][:648], ac_system['ac_sum'][:648])/np.mean(ac_system['ac_measured'][:648])
RMAE_ws_august = mean_absolute_error(ac_system['ac_measured'][648:1392], ac_system['ac_sum'][648:1392])/np.mean(ac_system['ac_measured'][648:1392])
RMAE_ws_september = mean_absolute_error(ac_system['ac_measured'][1392:2112], ac_system['ac_sum'][1392:2112])/np.mean(ac_system['ac_measured'][1392:2112])
RMAE_ws_october = mean_absolute_error(ac_system['ac_measured'][2112:], ac_system['ac_sum'][2112:])/np.mean(ac_system['ac_measured'][2112:])

RRMSE_ws_july = mean_squared_error(ac_system['ac_measured'][:648], ac_system['ac_sum'][:648], squared=False)/np.mean(ac_system['ac_measured'][:648])
RRMSE_ws_august = mean_squared_error(ac_system['ac_measured'][648:1392], ac_system['ac_sum'][648:1392], squared=False)/np.mean(ac_system['ac_measured'][648:1392])
RRMSE_ws_september = mean_squared_error(ac_system['ac_measured'][1392:2112], ac_system['ac_sum'][1392:2112], squared=False)/np.mean(ac_system['ac_measured'][1392:2112])
RRMSE_ws_october = mean_squared_error(ac_system['ac_measured'][2112:], ac_system['ac_sum'][2112:], squared=False)/np.mean(ac_system['ac_measured'][2112:])

#write statistics to csv
file = open('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/RMAE_RRMSE_Svalbardbutikken_month.csv', 'w', newline = '')
with file:
    header = ['Month', 'RMAE', 'RRMSE']
    writer = csv.DictWriter(file, fieldnames = header)

    writer.writeheader()
    writer.writerow({'Month': 'July', 'RMAE': RMAE_ws_july, 'RRMSE': RRMSE_ws_july})
    writer.writerow({'Month': 'August', 'RMAE': RMAE_ws_august, 'RRMSE': RRMSE_ws_august})
    writer.writerow({'Month': 'September', 'RMAE': RMAE_ws_september, 'RRMSE': RRMSE_ws_september})
    writer.writerow({'Month': 'October', 'RMAE': RMAE_ws_october, 'RRMSE': RRMSE_ws_october})
