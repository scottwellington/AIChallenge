import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import pdb
    
# Note:

# NOAA WOA18 Data covers the following:

# * Temperature (°C)
# * Salinity (unitless)
# * Density (kg/m3)
# * Conductivity (S/m)
# * Mixed Layer Depth (m)
# * Dissolved Oxygen (µmol/kg)
# * Percent Oxygen Saturation (%)
# * Apparent Oxygen Utilization (µmol/kg)
# * Silicate (µmol/kg)
# * Phosphate (µmol/kg)
# * Nitrate (µmol/kg)

# As the base for your data, we'll use the data for Saccharina latissima from 
# Venolia et al. (who in turn have compiled this data from 4 other courses).
# GitHub = https://github.com/CVenolia/SugarKelpDEB

class SugarKelpDEB:

    def __init__(self):
        self.read_path = os.path.join('.','df.csv')
        self.write_path = os.path.join('.','df_out.csv')

    def read_data(self):
        # Data output from R-script processing from Venolia et al.:
        # "Modeling the Growth of Sugar Kelp (Saccharina latissima)
        # in Aquaculture Systems using Dynamic Energy Budget Theory"

        df = pd.read_csv(self.read_path)
        
        # Environmental conditions:
        
        t_ = df['Temp_C'] # Temperature
        i_ = df['I'] # Irradiance
        d_ = df['CO_2'] # Dissolved Inorganic Carbon (DIC)
        n_ = df['N'] # Nitrate and nitrate concentration (nitrates)
        
        def get_d(array):
            y = []
            for i in range(1,len(array)):
                y.append(abs(array[i]-array[i-1]))
            y.insert(0, y[0])
            return np.array(y)
            
        def get_dd(d_array):
            y = []
            for i in range(1,len(d_array)):
                y.append(abs(d_array[i]-d_array[i-1]))
            y.insert(0, y[0])
            y.insert(0, y[0])
            return np.array(y)
        
        t_d = get_d(t_)
        t_dd = get_d(t_d)
        
        i_d = get_d(i_)
        i_dd = get_d(i_d)

        d_d = get_d(d_)
        d_dd = get_d(d_d)

        n_d = get_d(n_)
        n_dd = get_d(n_d)
        
        r = df['r'] # Net Specific Growth Rate (SGR)
        
        data = {'temperature':t_,
                'temperature_d':t_d,
                'temperature_dd':t_dd,
                'irradiance':i_,
                'irradiance_d':i_d,
                'irradiance_dd':i_dd,
                'DIC':d_,
                'DIC_d':d_d,
                'DIC_dd':d_dd,
                'nitrates':n_,
                'nitrates_d':n_d,
                'nitrates_dd':n_dd,
                'SGR':r,
                }
        
        df = pd.DataFrame(data, columns = [
                'temperature',
                'temperature_d',
                'temperature_dd',
                'irradiance',
                'irradiance_d',
                'irradiance_dd',
                'DIC',
                'DIC_d',
                'DIC_dd',
                'nitrates',
                'nitrates_d',
                'nitrates_dd',
                'SGR',
                ])
                
        return df
    
    def write_data(self, df):
        df.to_csv(self.write_path, index=False)
        
    def main(self):
        df = self.read_data()
        self.write_data(df)

latissima = SugarKelpDEB()
latissima.main()