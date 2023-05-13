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
from pvlib.pvsystem import calcparams_cec, singlediode, sapm_effective_irradiance
from pvlib.singlediode import bishop88, bishop88_v_from_i, bishop88_i_from_v, bishop88_mpp
from pvlib import irradiance
from pvlib.pvsystem import dc_ohms_from_percent, dc_ohmic_losses

#module and inverter specifications
sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
cec_modules = pvlib.pvsystem.retrieve_sam(path='https://raw.githubusercontent.com/NREL/SAM/patch/deploy/libraries/CEC%20Modules.csv')
cec_inverters1 = pvlib.pvsystem.retrieve_sam('cecinverter')
cec_inverters2 = pvlib.pvsystem.retrieve_sam(path = 'https://raw.githubusercontent.com/NREL/SAM/patch/deploy/libraries/CEC%20Inverters.csv')

#KSAT
module_ksat = cec_modules['Hanwha_Q_CELLS_Q_PEAK_DUO_G5_320'] #Hanwha Q CELLS QPEAK DUO G5
inverter_ksat_1 = cec_inverters1['SMA_America__SB5_0_1SP_US_40__240V_'] #Sunny Boy SB5.0-1SP-40
inverter_ksat_2 = cec_inverters2['SMA_America__SB5_0_1SP_US_41__240V_'] #Sunny Boy SB5.0-1AV-41 replaced with Sunny Boy SB5.0-1SP-41


#input weather data (temp, wind speed, ghi, dhi, dni)
df_tmy = pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/weather data/weather data_270122/weather_data_plataberget.csv')
df_tmy = df_tmy.reset_index(drop=True) #set timestamp as the index
mask = (df_tmy['TIMESTAMP'] >= '2022-03-23')  & (df_tmy['TIMESTAMP'] <= '2022-11-01')
df_tmy = df_tmy.loc[mask]
df_tmy['TIMESTAMP'] = pd.to_datetime(df_tmy['TIMESTAMP'])
df_tmy = df_tmy.set_index('TIMESTAMP')
#add albedo data to weather date dataframe
df_albedo = pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/plataberget_fieldwork_meteodata_280223/TOA5_6222.Data_meteo.csv', usecols = ['TIMESTAMP', 'Kglob', 'Kref'], low_memory = False)
mask = (df_albedo['TIMESTAMP'] >= '2022-03-23')  & (df_albedo['TIMESTAMP'] <= '2022-11-01')
df_albedo = df_albedo.loc[mask]
df_albedo = df_albedo.reset_index(drop=True)
df_albedo['TIMESTAMP'] = pd.to_datetime(df_albedo['TIMESTAMP'])
df_albedo = df_albedo.set_index('TIMESTAMP')
df_albedo = df_albedo.astype(np.float16)
df_albedo = df_albedo.resample('H').mean() #resample 5 min frequency to hourly frequency
df_albedo = df_albedo.rename(columns={'Kref':'SW_up', 'Kglob':'SW_down'})
df_albedo['albedo'] = df_albedo.SW_up/df_albedo.SW_down
df_albedo.loc[df_albedo.SW_down < 50, 'albedo'] = np.nan #SW down under 50 Wm-2 is noise, therefore set it to NaN
df_albedo.loc[df_albedo.albedo > 1, 'albedo'] = np.nan # albedo over 1 is not possible - error on measuring device
df_albedo.albedo = df_albedo.albedo.interpolate(method='linear') #interpolate NaN values linearly
df_albedo.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/weather data/albedo/albedo_check/rad_abeldo_pb.csv')
df_tmy['albedo'] = df_albedo.albedo.values


#satellite
df_tmy_sat = pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/satellite data/weather_data_ksat.csv')
df_tmy_sat = df_tmy_sat.reset_index(drop=True) #set timestamp as the index
mask = (df_tmy_sat['time'] >= '2022-03-23')  & (df_tmy_sat['time'] <= '2022-11-01')
df_tmy_sat = df_tmy_sat.loc[mask]
df_tmy_sat['time'] = pd.to_datetime(df_tmy_sat['time'])
df_tmy_sat = df_tmy_sat.set_index('time')

#time zone
tz = 'Europe/Oslo'

#time and location
times_ksat = pd.date_range('2022-03-23 00:00:00', '2022-11-01 01:00:00', closed='left', freq='H', tz=tz) #adapt to measured ac generation time range
loc_sg26 = Location(latitude = 78.23137, longitude = 15.40709, tz = times_ksat.tz, altitude = 466)
loc_sg64 = Location(latitude = 78.22638, longitude = 15.42917, tz = times_ksat.tz, altitude = 466)
solar_position_sg26 = loc_sg26.get_solarposition(times_ksat, method='ephemeris')
solar_position_sg64 = loc_sg64.get_solarposition(times_ksat, method='ephemeris')

#shading map
shading_ksat = np.load('svm_KSAT_SG26.npy')
shading_ksat = pd.DataFrame(shading_ksat)
shading_ksat.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/map_shading_KSAT.csv')

#shade SG26
solar_position_sg26["shade"] = shading_ksat.stack()[pd.MultiIndex.from_arrays([np.clip(solar_position_sg26["zenith"].astype(int), None, 90), solar_position_sg26["azimuth"].astype(int)])].values
solar_position_sg26.loc[solar_position_sg26["zenith"] >= 90, "shade"] = True
df_tmy['shade_sg26'] = solar_position_sg26['shade'] #apply shade to weather data
df_tmy['shade_sg26'] = df_tmy['shade_sg26'].fillna(False).astype(bool)
df_tmy['dni_shade_sg26'] = df_tmy['dni'].where(~df_tmy['shade_sg26'], 0)
solar_position_sg26.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/solar_position_sg26.csv')

#shade SG64
solar_position_sg64["shade"] = shading_ksat.stack()[pd.MultiIndex.from_arrays([np.clip(solar_position_sg64["zenith"].astype(int), None, 90), solar_position_sg64["azimuth"].astype(int)])].values  #implement a new column to solar position with the boolean values True and False depending on the sun's position and in relation to shading map
solar_position_sg64.loc[solar_position_sg64["zenith"] >= 90, "shade"] = True #if zenith is over 90 degrees set the shade to True
df_tmy['shade_sg64'] = solar_position_sg64['shade'] #apply shade to weather data
df_tmy['shade_sg64'] = df_tmy['shade_sg64'].fillna(False).astype(bool)
df_tmy['dni_shade_sg64'] = df_tmy['dni'].where(~df_tmy['shade_sg64'], 0)
solar_position_sg64.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/solar_position_sg64.csv')

##### PV SYSTEM SETUP #####

tilt = 90 #as they are all wall-mounted
temperature_model_parameters_KSAT = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['insulated_back_glass_polymer'] #depending on material and type of mounting

#SG26 facing North
#wired together ne1,ne2,n || nw1,nw2
azimuth_ksat_ne1 = 48
azimuth_ksat_ne2 = 24
azimuth_ksat_n = 360
azimuth_ksat_nw1 = 336
azimuth_ksat_nw2 = 312

#mount SG26
mount_ne1 = FixedMount(tilt, azimuth_ksat_ne1)
mount_ne2 = FixedMount(tilt, azimuth_ksat_ne2)
mount_n = FixedMount(tilt, azimuth_ksat_n)
mount_nw1 = FixedMount(tilt, azimuth_ksat_nw1)
mount_nw2 = FixedMount(tilt, azimuth_ksat_nw2)

#SG64 facing South
#wired together sw1,sw2,s1 || s2,se1,se2
azimuth_ksat_sw1 = 240
azimuth_ksat_sw2 = 216
azimuth_ksat_s1 = 192
azimuth_ksat_s2 = 168
azimuth_ksat_se1 = 144
azimuth_ksat_se2 = 120

#mount SG64
mount_sw1 = FixedMount(tilt, azimuth_ksat_sw1)
mount_sw2 = FixedMount(tilt, azimuth_ksat_sw2)
mount_s1 = FixedMount(tilt, azimuth_ksat_s1)
mount_s2 = FixedMount(tilt, azimuth_ksat_s2)
mount_se1 = FixedMount(tilt, azimuth_ksat_se1)
mount_se2 = FixedMount(tilt, azimuth_ksat_se2)

array_kwargs_sg26 = dict(module_parameters = module_ksat,
                       temperature_model_parameters = temperature_model_parameters_KSAT,
                       modules_per_string = 3)
array_kwargs_sg64 = dict(module_parameters = module_ksat,
                       temperature_model_parameters = temperature_model_parameters_KSAT,
                       modules_per_string = 2)

#fed into inverter 1
sg26_ne1 = Array(mount_ne1, name ='NE1', **array_kwargs_sg26)
sg26_ne2 = Array(mount_ne2, name ='NE2', **array_kwargs_sg26)
sg26_n = Array(mount_n, name ='N', **array_kwargs_sg26)
sg26_nw1 = Array(mount_nw1, name ='NW1', **array_kwargs_sg26)
sg26_nw2 = Array(mount_nw2, name ='NW2', **array_kwargs_sg26)

#fed into inverter 2
sg64_sw1 = Array(mount_sw1, name ='SW1', **array_kwargs_sg64)
sg64_sw2 = Array(mount_sw2, name ='SW2', **array_kwargs_sg64)
sg64_s1 = Array(mount_s1, name ='S1', **array_kwargs_sg64)
sg64_s2 = Array(mount_s2, name ='S2', **array_kwargs_sg64)
sg64_se1 = Array(mount_se1, name ='SE1', **array_kwargs_sg64)
sg64_se2 = Array(mount_se2, name ='SE2', **array_kwargs_sg64)


#PV module specific
ksat_gamma_pdc = -0.0036 #The temperature coefficient of power. Typically -0.002 to -0.005 per degree C. [1/C]
ksat_pdc0 =320 #Power of the modules at 1000 W/m^2 and cell reference temperature. [W]

##### AC OUTPUT MODELING #####

"""all parameters defined which are needed for modeling p_mp"""

azimuth = [azimuth_ksat_ne1, azimuth_ksat_ne2, azimuth_ksat_n, azimuth_ksat_nw1, azimuth_ksat_nw2,
           azimuth_ksat_sw1, azimuth_ksat_sw2, azimuth_ksat_s1, azimuth_ksat_s2, azimuth_ksat_se1,
           azimuth_ksat_se2]

arrays = [sg26_ne1, sg26_ne2, sg26_n, sg26_nw1, sg26_nw2, sg64_sw1, sg64_sw2, sg64_s1, sg64_s2, sg64_se1, sg64_se2]

arrays_names = ['NE1', 'NE2', 'N', 'NW1', 'NW2', 'SW1', 'SW2', 'S1', 'S2', 'SE1', 'SE2']

poa_global = pd.DataFrame()
poa_direct = pd.DataFrame()
poa_diffuse = pd.DataFrame()
poa_ground_diffuse = pd.DataFrame()
cell_temp = pd.DataFrame()
eff_irrad = pd.DataFrame()
aoi_all = pd.DataFrame()

#POA irradiance and Cell temperature
for i in range(0,5): #Model poa irradiance and cell temp. depending on the array (North facing panels)
    sg26_poa = arrays[i].get_irradiance(solar_zenith=solar_position_sg26['apparent_zenith'], solar_azimuth=solar_position_sg26['azimuth'], ghi=df_tmy['ghi'], dhi=df_tmy['dhi'], dni=df_tmy['dni_shade_sg26'], albedo = df_tmy.albedo, model='perez')
    sg26_cell_temp = arrays[i].get_cell_temperature(sg26_poa['poa_global'], df_tmy['temp_air'], df_tmy['wind_speed'], 'sapm')

    poa_global[arrays_names[i]] = sg26_poa['poa_global']
    poa_direct[arrays_names[i]] = sg26_poa['poa_direct']
    poa_diffuse[arrays_names[i]] = sg26_poa['poa_diffuse']
    poa_ground_diffuse[arrays_names[i]] = sg26_poa['poa_ground_diffuse']

    cell_temp[arrays_names[i]] = sg26_cell_temp

for i in range(5,11): #(South facing panels)
    sg64_poa = arrays[i].get_irradiance(solar_zenith=solar_position_sg64['apparent_zenith'], solar_azimuth=solar_position_sg64['azimuth'], ghi=df_tmy['ghi'], dhi=df_tmy['dhi'], dni=df_tmy['dni_shade_sg64'], albedo = df_tmy.albedo,  model='perez')
    sg64_cell_temp = arrays[i].get_cell_temperature(sg64_poa['poa_global'], df_tmy['temp_air'], df_tmy['wind_speed'], 'sapm')

    poa_global[arrays_names[i]] = sg64_poa['poa_global']
    poa_direct[arrays_names[i]] = sg64_poa['poa_direct']
    poa_diffuse[arrays_names[i]] = sg64_poa['poa_diffuse']
    poa_ground_diffuse[arrays_names[i]] = sg64_poa['poa_ground_diffuse']

    cell_temp[arrays_names[i]] = sg64_cell_temp

#Approximation: picked most common value from Sandia Modules Database - MODALWERT in Excel
module_ksat['A0'] = 0.921941
module_ksat['A1'] = 0.0708917
module_ksat['A2'] = -0.0142724
module_ksat['A3'] = 0.0011709
module_ksat['A4'] = -0.000033705
module_ksat['B0'] = 1
module_ksat['B1'] = -0.002438
module_ksat['B2'] = 0.0003103
module_ksat['B3'] = -0.00001246
module_ksat['B4'] = 0.000000211
module_ksat['B5'] = -1.36E-09
module_ksat['FD'] = 1

#Air Mass for both locations
am_sg26 = loc_sg26.get_airmass(times_ksat, solar_position_sg26, model = 'simple')
am_sg64 = loc_sg64.get_airmass(times_ksat, solar_position_sg64, model ='kastenyoung1989')

for i in range(0,5): #Model AOI and effective irradiance depending on the array (North facing panels)
    aoi_sg26 = irradiance.aoi(tilt, azimuth[i], solar_position_sg26['apparent_zenith'],solar_position_sg26['azimuth'])
    aoi_all[arrays_names[i]]=aoi_sg26
    eff_irrad_sg26 = sapm_effective_irradiance(poa_direct[arrays_names[i]], poa_diffuse[arrays_names[i]], am_sg26.airmass_absolute, aoi_sg26, module_ksat)
    eff_irrad[arrays_names[i]] = eff_irrad_sg26

for i in range(5,11): #(South facing panels)
    aoi_sg64 = irradiance.aoi(tilt, azimuth[i], solar_position_sg64['apparent_zenith'],solar_position_sg64['azimuth'])
    aoi_all[arrays_names[i]]=aoi_sg64
    eff_irrad_sg64 = sapm_effective_irradiance(poa_direct[arrays_names[i]], poa_diffuse[arrays_names[i]], am_sg64.airmass_absolute, aoi_sg64, module_ksat)
    eff_irrad[arrays_names[i]] = eff_irrad_sg64

#apply IAM losses to DNI
aoi_sg26 = irradiance.aoi(tilt, azimuth[i], solar_position_sg26['apparent_zenith'],solar_position_sg26['azimuth'])
aoi_all[arrays_names[i]]=aoi_sg26
iam_sg26 = pvlib.iam.ashrae(aoi_sg26)
df_tmy['dni'] = df_tmy['dni'] * iam_sg26
aoi_sg64 = irradiance.aoi(tilt, azimuth[i], solar_position_sg64['apparent_zenith'],solar_position_sg64['azimuth'])
aoi_all[arrays_names[i]]=aoi_sg64
iam_sg64 = pvlib.iam.physical(aoi_sg64)
df_tmy['dni'] = df_tmy['dni'] * iam_sg64

"""functions to determine the I-V curve parameters"""

#string1: NE1, NE2, N
#string2: NW1, NW2
#string3: SW1, SW2, S1
#string4: S2, SE1, SE2

i_mp_str1 = pd.DataFrame()
v_mp_str1 = pd.DataFrame()
i_mp_str2 = pd.DataFrame()
v_mp_str2 = pd.DataFrame()
i_mp_str3 = pd.DataFrame()
v_mp_str3 = pd.DataFrame()
i_mp_str4 = pd.DataFrame()
v_mp_str4 = pd.DataFrame()

for i in range(0,3): #looking at the different strings
    #calculate the five parameters for the single diode equation
    #photocurrent IL, saturation current I0, Series resistance Rs, Shunt resistance Rsh, nNsVth
    IL, I0, Rs, Rsh, nNsVth = calcparams_cec(eff_irrad[arrays_names[i]], cell_temp[arrays_names[i]],
                                             module_ksat.alpha_sc, module_ksat.a_ref, module_ksat.I_L_ref,
                                             module_ksat.I_o_ref, module_ksat.R_sh_ref, module_ksat.R_s,
                                             module_ksat.Adjust) #for every hour depending on the irradiance and cell temp
    #solve single diode equation to obtain IV curve for each array
    #i_sc short circuit current, v_oc open circuit voltage, i_mp maximum power point current, v_mp maximum power point voltage,p_mp maximum power point power, i_x current at 0.5*v_oc, i_xx current at 0.5*(v_oc+v_mp)
    sd = singlediode(IL, I0, Rs, Rsh, nNsVth)
    i_mp_str1[arrays_names[i]] = sd.i_mp
    v_mp_str1[arrays_names[i]] = sd.v_mp

i_mp_str1['I_min'] = i_mp_str1.min(axis = 1) #minimum current per string determined
v_mp_str1['V_total'] = v_mp_str1.sum(axis=1)*3 #since it's connected in series and each array has 3 modules, it has to be multiplied by 3

for i in range(3,5):
    IL, I0, Rs, Rsh, nNsVth = calcparams_cec(eff_irrad[arrays_names[i]], cell_temp[arrays_names[i]],
                                             module_ksat.alpha_sc, module_ksat.a_ref, module_ksat.I_L_ref,
                                             module_ksat.I_o_ref, module_ksat.R_sh_ref, module_ksat.R_s,
                                             module_ksat.Adjust) #for every hour depending on the irradiance and cell temp
    sd = singlediode(IL, I0, Rs, Rsh, nNsVth)
    i_mp_str2[arrays_names[i]] = sd.i_mp
    v_mp_str2[arrays_names[i]] = sd.v_mp

i_mp_str2['I_min'] = i_mp_str2.min(axis = 1)
v_mp_str2['V_total'] = v_mp_str2.sum(axis=1)*3

for i in range(5,8):
    IL, I0, Rs, Rsh, nNsVth = calcparams_cec(eff_irrad[arrays_names[i]], cell_temp[arrays_names[i]],
                                             module_ksat.alpha_sc, module_ksat.a_ref, module_ksat.I_L_ref,
                                             module_ksat.I_o_ref, module_ksat.R_sh_ref, module_ksat.R_s,
                                             module_ksat.Adjust) #for every hour depending on the irradiance and cell temp
    sd = singlediode(IL, I0, Rs, Rsh, nNsVth)
    i_mp_str3[arrays_names[i]] = sd.i_mp
    v_mp_str3[arrays_names[i]] = sd.v_mp

i_mp_str3['I_min'] = i_mp_str3.min(axis = 1)
v_mp_str3['V_total'] = v_mp_str3.sum(axis=1)*2 #since it's connected in series and each array has 2 modules, it has to be multiplied by 2

for i in range(8,11):
    IL, I0, Rs, Rsh, nNsVth = calcparams_cec(eff_irrad[arrays_names[i]], cell_temp[arrays_names[i]],
                                             module_ksat.alpha_sc, module_ksat.a_ref, module_ksat.I_L_ref,
                                             module_ksat.I_o_ref, module_ksat.R_sh_ref, module_ksat.R_s,
                                             module_ksat.Adjust) #for every hour depending on the irradiance and cell temp
    sd = singlediode(IL, I0, Rs, Rsh, nNsVth)
    i_mp_str4[arrays_names[i]] = sd.i_mp
    v_mp_str4[arrays_names[i]] = sd.v_mp

i_mp_str4['I_min'] = i_mp_str4.min(axis = 1)
v_mp_str4['V_total'] = v_mp_str4.sum(axis=1)*2

#modeled DC
p_mp_dc = pd.DataFrame()
#generate the maximum power point power in DC for each timestep for each string
p_mp_dc['sg26_str1'] = i_mp_str1['I_min'] * v_mp_str1['V_total']
p_mp_dc['sg26_str2'] = i_mp_str2['I_min'] * v_mp_str2['V_total']
p_mp_dc['sg64_str1'] = i_mp_str3['I_min'] * v_mp_str3['V_total']
p_mp_dc['sg64_str2'] = i_mp_str4['I_min'] * v_mp_str4['V_total']
p_mp_dc.to_csv('dc_ksat.csv')

#DC ohmic wiring losses
res_ksat11 = dc_ohms_from_percent(module_ksat.V_mp_ref, module_ksat.I_mp_ref, 1.5, 9, 1) #vmp_ref, imp_ref, dc_ohmic_percent, modules_per_string, strings
res_ksat12 = dc_ohms_from_percent(module_ksat.V_mp_ref, module_ksat.I_mp_ref, 1.5, 6, 1)
res_ksat2 = dc_ohms_from_percent(module_ksat.V_mp_ref, module_ksat.I_mp_ref, 1.5, 6, 1)

ohmic_loss_sg26_1 = dc_ohmic_losses(res_ksat11, i_mp_str1['I_min'])
ohmic_loss_sg26_2 = dc_ohmic_losses(res_ksat12, i_mp_str2['I_min'])
ohmic_loss_sg64_1 = dc_ohmic_losses(res_ksat2, i_mp_str3['I_min'])
ohmic_loss_sg64_2 = dc_ohmic_losses(res_ksat2, i_mp_str4['I_min'])

p_mp_dc['sg26_str1']= ((p_mp_dc['sg26_str1'])-ohmic_loss_sg26_1)
p_mp_dc['sg26_str2']= ((p_mp_dc['sg26_str2'])-ohmic_loss_sg26_2)
p_mp_dc['sg64_str1']= ((p_mp_dc['sg64_str1'])-ohmic_loss_sg64_1)
p_mp_dc['sg64_str2']= ((p_mp_dc['sg64_str2'])-ohmic_loss_sg64_2)

#mismatching effect
#https://pvpmc.sandia.gov/modeling-steps/3-dc-array-iv/mismatch-losses/

sg26_1_p_mod = [9*320]
sg26_2_p_mod = [6*320]
sg64_p_mod = [6*320]

mm_sg26_1 = 1-((p_mp_dc['sg26_str1'])/(sg26_1_p_mod-ohmic_loss_sg26_1))
mm_sg26_2 = 1-((p_mp_dc['sg26_str2'])/(sg26_2_p_mod-ohmic_loss_sg26_2))
mm_sg64_1 = 1-((p_mp_dc['sg64_str1'])/(sg64_p_mod-ohmic_loss_sg64_1))
mm_sg64_2 = 1-((p_mp_dc['sg64_str2'])/(sg64_p_mod-ohmic_loss_sg64_2))

loss_mm_sg26_1 = mm_sg26_1.mean()
loss_mm_sg26_2 = mm_sg26_2.mean()
loss_mm_sg64_1 = mm_sg64_1.mean()
loss_mm_sg64_2 = mm_sg64_2.mean()

p_mp_dc['sg26_str1']= p_mp_dc['sg26_str1']*loss_mm_sg26_1
p_mp_dc['sg26_str2']= p_mp_dc['sg26_str2']*loss_mm_sg26_2
p_mp_dc['sg64_str1']= p_mp_dc['sg64_str1']*loss_mm_sg64_1
p_mp_dc['sg64_str2']= p_mp_dc['sg64_str2']*loss_mm_sg64_2


#modeled AC incl. losses
p_mp_ac = pd.DataFrame()
#convert p_mp DC to AC using a multi input inverter as each inverter has two strings as input
p_mp_ac['sg26'] = pvlib.inverter.sandia_multi([v_mp_str1['V_total'], v_mp_str2['V_total']], [p_mp_dc['sg26_str1'], p_mp_dc['sg26_str2']], inverter_ksat_1)
p_mp_ac['sg64'] = pvlib.inverter.sandia_multi([v_mp_str3['V_total'], v_mp_str4['V_total']], [p_mp_dc['sg64_str1'], p_mp_dc['sg64_str2']], inverter_ksat_2)

#import of measured data
df_sg26= pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/SG26_measurements.csv') #daily measurements
df_sg64= pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/pv system data/SG64_measurements.csv') #daily measurements
df_sg26 = df_sg26.reset_index(drop=True) #set timestamp as the index
df_sg26['Date'] = pd.to_datetime(df_sg26['Date'])
df_sg26['kWh'] = df_sg26['kWh']
df_sg26 = df_sg26.set_index('Date')
df_sg64 = df_sg64.reset_index(drop=True) #set timestamp as the index
df_sg64['Date'] = pd.to_datetime(df_sg64['Date'])
df_sg64['kWh'] = df_sg64['kWh']
df_sg64 = df_sg64.set_index('Date')

#import of modeled AC with satellite data
df_sat = pd.read_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/results 100323/ac_ksat_satellite.csv')
df_sat = df_sat.reset_index(drop=True) #set timestamp as the index
df_sat['TIMESTAMP'] = pd.to_datetime(df_sat['TIMESTAMP'])
df_sat['sg26_ac'] = df_sat['sg26_ac']
df_sat['sg64_ac'] = df_sat['sg64_ac']
df_sat = df_sat.set_index('TIMESTAMP')

#measurements and model output together in one dataframe
ksat_ac = pd.DataFrame() #fiure this out
ksat_ac['sg26_ac'] = p_mp_ac['sg26'].resample('D').sum()/1000
ksat_ac['sg64_ac'] = p_mp_ac['sg64'].resample('D').sum()/1000
ksat_ac['ac_measured_sg26'] = df_sg26.kWh[21:].values
ksat_ac['ac_measured_sg64'] = df_sg64.kWh[21:].values
ksat_ac['ac_sg26_modelerror'] = ksat_ac.sg26_ac - ksat_ac.ac_measured_sg26
ksat_ac['ac_sg64_modelerror'] = ksat_ac.sg64_ac - ksat_ac.ac_measured_sg64
ksat_ac = ksat_ac.dropna()
ksat_ac.to_csv('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/ac_ksat_sat.csv')

#model AC output
#SG26 weekly output
fig1, ax1 = plt.subplots(figsize=(6, 2), dpi=300)
(p_mp_ac['sg26'].resample('W').sum()/1000).plot(label='Inverter (Platåberget Weather station)', linestyle ='dotted', linewidth= 0.5, color = 'crimson', ax=ax1) #as the sum over the week is resampled, the unit is kWh, divide by 1000 as model gives W
df_sg26['kWh'].resample('W').sum().plot(label='Measurements', color = 'black', linewidth= 0.5, ax=ax1)
df_sat['sg26_ac'].resample('W').sum().plot(label='Inverter (ERA5 Satellite data)', linestyle = 'dashed', linewidth= 0.5, color = 'teal', ax=ax1)
ax1.set_xlim('2022-03-23', '2022-11-01')
ax1.set_ylabel('Energy Output [kWh]', fontsize=8)
ax1.set_xlabel('Months', fontsize=8)
ax1.xaxis.set_tick_params(labelsize=8)
ax1.yaxis.set_tick_params(labelsize=8)
ax1.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize=8)
plt.savefig('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/adjusted/KSAT_sg26_ws_sat_week.png', dpi=300, bbox_inches='tight')
#SG64 weekly AC output
fig2, ax2 = plt.subplots(figsize=(6, 2), dpi=300)
(p_mp_ac['sg64'].resample('W').sum()/1000).plot(label='Inverter (Platåberget Weather station)', linestyle ='dotted', linewidth= 0.5, color = 'crimson', ax=ax2)
df_sg64['kWh'].resample('W').sum().plot(label='Measurements', color = 'black', linewidth= 0.5, ax=ax2)
df_sat['sg64_ac'].resample('W').sum().plot(label='Inverter (Satellite data)', linestyle = 'dashed', linewidth= 0.5, color = 'teal', ax=ax2)
ax2.set_xlim('2022-03-23', '2022-11-01')
ax2.set_ylabel('Energy Output [kWh]', fontsize=8)
ax2.set_xlabel('Months', fontsize=8)
ax2.xaxis.set_tick_params(labelsize=8)
ax2.yaxis.set_tick_params(labelsize=8)
ax2.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize=8)
plt.savefig('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/adjusted/KSAT_sg64_ws_sat_week.png', dpi=300, bbox_inches='tight')
#SG26 daily AC output
fig1, ax1 = plt.subplots(figsize=(6, 2), dpi=300)
(p_mp_ac['sg26'].resample('D').sum()/1000).plot(label='Inverter (Platåberget Weather station)', linestyle ='dotted', linewidth = 0.5, color = 'crimson', ax=ax1) #as the sum over the week is resampled, the unit is kWh
df_sg26['kWh'].resample('D').sum().plot(label='Measurements', color = 'black', linewidth = 0.5, ax=ax1)
ax1.set_xlim('2022-03-23', '2022-04-01')
ax1.set_ylabel('Energy Output [kWh]', fontsize=8)
ax1.set_xlabel('Months', fontsize=8)
ax1.xaxis.set_tick_params(labelsize=8)
ax1.yaxis.set_tick_params(labelsize=8)
ax1.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize=8)
plt.savefig('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/adjusted/KSAT_sg26_ws_day.png', dpi=300, bbox_inches='tight')
#SG64 daily AC output
fig2, ax2 = plt.subplots(figsize=(6, 2), dpi=300)
(p_mp_ac['sg64'].resample('D').sum()/1000).plot(label='Inverter (Platåberget Weather station)', linestyle ='dotted', linewidth = 0.5, color = 'crimson', ax=ax2)
df_sg64['kWh'].resample('D').sum().plot(label='Measurements', color = 'black', linewidth = 0.5, ax=ax2)
ax2.set_xlim('2022-03-23', '2022-04-01')
ax2.set_ylabel('Energy Output [kWh]', fontsize=8)
ax2.set_xlabel('Months', fontsize=8)
ax2.xaxis.set_tick_params(labelsize=8)
ax2.yaxis.set_tick_params(labelsize=8)
ax2.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fontsize=8)
plt.savefig('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/adjusted/KSAT_sg64_ws_day.png', dpi=300, bbox_inches='tight')

##### STATISTICAL MEASURES #####
ksat_ac = ksat_ac.dropna()
RMAE_sat_year = mean_absolute_error(ksat_ac['ac_measured_sg26'], df_sat['sg26_ac'])/np.mean(ksat_ac['ac_measured_sg26'])
RMAE_ws_year = mean_absolute_error(ksat_ac['ac_measured_sg26'], ksat_ac['sg26_ac'])/np.mean(ksat_ac['ac_measured_sg26'])

RRMSE_sat_year = mean_squared_error(ksat_ac['ac_measured_sg26'], df_sat['sg26_ac'], squared=False)/np.mean(ksat_ac['ac_measured_sg26'])
RRMSE_ws_year = mean_squared_error(ksat_ac['ac_measured_sg26'], ksat_ac['sg26_ac'], squared=False)/np.mean(ksat_ac['ac_measured_sg26'])

#write statistics to csv
file = open('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/RMAE_RRMSE_SG26_year.csv', 'w', newline = '')
with file:
    header = ['Weather Data', 'RMAE', 'RRMSE']
    writer = csv.DictWriter(file, fieldnames = header)

    writer.writeheader()
    writer.writerow({'Weather Data': 'Sat','RMAE': RMAE_sat_year, 'RRMSE': RRMSE_sat_year})
    writer.writerow({'Weather Data': 'WS','RMAE': RMAE_ws_year, 'RRMSE': RRMSE_ws_year})

RMAE_ws_march = mean_absolute_error(ksat_ac['ac_measured_sg26'][:9], ksat_ac['sg26_ac'][:9])/np.mean(ksat_ac['ac_measured_sg26'][:9])
RMAE_ws_april = mean_absolute_error(ksat_ac['ac_measured_sg26'][9:39], ksat_ac['sg26_ac'][9:39])/np.mean(ksat_ac['ac_measured_sg26'][9:39])
RMAE_ws_may = mean_absolute_error(ksat_ac['ac_measured_sg26'][39:70], ksat_ac['sg26_ac'][39:70])/np.mean(ksat_ac['ac_measured_sg26'][39:70])
RMAE_ws_june = mean_absolute_error(ksat_ac['ac_measured_sg26'][70:100], ksat_ac['sg26_ac'][70:100])/np.mean(ksat_ac['ac_measured_sg26'][70:100])
RMAE_ws_july = mean_absolute_error(ksat_ac['ac_measured_sg26'][100:131], ksat_ac['sg26_ac'][100:131])/np.mean(ksat_ac['ac_measured_sg26'][100:131])
RMAE_ws_august = mean_absolute_error(ksat_ac['ac_measured_sg26'][131:162], ksat_ac['sg26_ac'][131:162])/np.mean(ksat_ac['ac_measured_sg26'][131:162])
RMAE_ws_september = mean_absolute_error(ksat_ac['ac_measured_sg26'][162:192], ksat_ac['sg26_ac'][162:192])/np.mean(ksat_ac['ac_measured_sg26'][162:192])
RMAE_ws_october = mean_absolute_error(ksat_ac['ac_measured_sg26'][192:], ksat_ac['sg26_ac'][192:])/np.mean(ksat_ac['ac_measured_sg26'][192:])

RRMSE_ws_march = mean_squared_error(ksat_ac['ac_measured_sg26'][:9], ksat_ac['sg26_ac'][:9], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][:9])
RRMSE_ws_april = mean_squared_error(ksat_ac['ac_measured_sg26'][9:39], ksat_ac['sg26_ac'][9:39], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][9:39])
RRMSE_ws_may = mean_squared_error(ksat_ac['ac_measured_sg26'][39:70], ksat_ac['sg26_ac'][39:70], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][39:70])
RRMSE_ws_june = mean_squared_error(ksat_ac['ac_measured_sg26'][70:100], ksat_ac['sg26_ac'][70:100], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][70:100])
RRMSE_ws_july = mean_squared_error(ksat_ac['ac_measured_sg26'][100:131], ksat_ac['sg26_ac'][100:131], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][100:131])
RRMSE_ws_august = mean_squared_error(ksat_ac['ac_measured_sg26'][131:162], ksat_ac['sg26_ac'][131:162], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][131:162])
RRMSE_ws_september = mean_squared_error(ksat_ac['ac_measured_sg26'][162:192], ksat_ac['sg26_ac'][162:192], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][162:192])
RRMSE_ws_october = mean_squared_error(ksat_ac['ac_measured_sg26'][192:], ksat_ac['sg26_ac'][192:], squared=False)/np.mean(ksat_ac['ac_measured_sg26'][192:])


#write statistics to csv
file = open('/Users/KimCho/Library/Mobile Documents/com~apple~CloudDocs/Master UIW/04 UNIS/master thesis/result_figures 180323/RMAE_RRMSE_Sg26_month.csv', 'w', newline = '')
with file:
    header = ['Month', 'RMAE', 'RRMSE']
    writer = csv.DictWriter(file, fieldnames = header)

    writer.writeheader()
    writer.writerow({'Month': 'March', 'RMAE': RMAE_ws_march, 'RRMSE': RRMSE_ws_march})
    writer.writerow({'Month': 'April', 'RMAE': RMAE_ws_april, 'RRMSE': RRMSE_ws_april})
    writer.writerow({'Month': 'May', 'RMAE': RMAE_ws_may, 'RRMSE': RRMSE_ws_may})
    writer.writerow({'Month': 'June', 'RMAE': RMAE_ws_june, 'RRMSE': RRMSE_ws_june})
    writer.writerow({'Month': 'July', 'RMAE': RMAE_ws_july, 'RRMSE': RRMSE_ws_july})
    writer.writerow({'Month': 'August', 'RMAE': RMAE_ws_august, 'RRMSE': RRMSE_ws_august})
    writer.writerow({'Month': 'September', 'RMAE': RMAE_ws_september, 'RRMSE': RRMSE_ws_september})
    writer.writerow({'Month': 'October', 'RMAE': RMAE_ws_october, 'RRMSE': RRMSE_ws_october})
