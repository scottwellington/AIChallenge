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
        self.write_path = None

    def nan_helper(self, nparray):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(nparray), lambda z: z.nonzero()[0]


    def add_Gaussian_noise(self, nparray):
        
        _mean = 0
        _sd = .1
        _len = nparray.shape
        _scale = np.mean(nparray)

        return np.abs(nparray + (np.random.normal(_mean,_sd,_len)*_scale))


    def read_data(self):
        # Data output from R-script processing from Venolia et al.:
        # "Modeling the Growth of Sugar Kelp (Saccharina latissima)
        # in Aquaculture Systems using Dynamic Energy Budget Theory"

        df = pd.read_csv(self.read_path)
        
        return df


    def calculate_NP_ratio(self, df):
        
        _n = df['N'] # Nitrate and nitrate concentration (nitrates)
        _p = df['P']  # Phosphate and phosphate concentration (phosphates)

        _p = (_p/95)/10000 # convert from moles to micrograms
        _p = _p/16 # apply Redfield N:P ratio conversion

        _ = _n/16#  apply Redfield N:P ratio conversion
        _ = np.abs(_p + _/100) # apply as linear transform
        
        _p = self.add_Gaussian_noise(_) # apply Gaussian blur

        df['P'] = _p

        return df


    def replace_nul_with_nan(self, df):
        
        df = df.replace(0.,np.nan)

        return df


    def interpolate_nans(self, df):

        for i in df.columns:

            try:
                ii = np.array(df[i])
                nans, x = self.nan_helper(ii)
                ii[nans] = np.interp(x(nans), x(~nans), ii[~nans])
                df[i] = ii
            except:
                pass

        return df


    def synthesize(self, df):
        

        for i in df.columns[1:]: # ignore time

            try:
                ii = np.array(df[i])
                df[i] = self.add_Gaussian_noise(ii)
            except:
                pass

        return df


    def get_deltas(self, df):

        # Environmental conditions:
        
        t_ = df['Temp_C'] # Temperature
        i_ = df['I'] # Irradiance
        d_ = df['CO_2'] # Dissolved Inorganic Carbon (DIC)
        n_ = df['N'] # Nitrate and nitrate concentration (nitrates)
        p_ = df['P']  # Phosphate and phosphate concentration (phosphates)
        
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

        p_d = get_d(p_)
        p_dd = get_d(p_d)        
        
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
                'phosphates':p_,
                'phosphates_d':p_d,
                'phosphates_dd':p_dd,
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
                'phosphates',
                'phosphates_d',
                'phosphates_dd',
                'SGR',
                ])
        

        df = self.replace_nul_with_nan(df)
        df = self.interpolate_nans(df)

        return df
    

    def write_data(self, df):
        df.to_csv(self.write_path, index=False)
        

    def main(self):
        df = self.read_data()
        df = self.calculate_NP_ratio(df)
        df = self.replace_nul_with_nan(df)
        df = self.interpolate_nans(df)
        df = self.synthesize(df)

        self.write_path = os.path.join('.','df_all.csv')
        self.write_data(df)

        df = self.get_deltas(df)

        self.write_path = os.path.join('.','df_delta.csv')
        self.write_data(df)




latissima = SugarKelpDEB()
latissima.main()