# convenient logging imports
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',level=logging.INFO, stream=sys.stdout)
import warnings
warnings.filterwarnings("ignore")

# functional imports
import os
import shapefile as shp
import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
import pdb
from scipy import interpolate
import gzip
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim

class SHP:

	def __init__(self,
				temperature = None,
				salinity = None,
				nitrates = None,
				phosphates = None,
				conductivity = None,
				DIC = None,

				temperature_weight = 1,
				salinity_weight = 1,
				nitrates_weight = 1,
				phosphates_weight = 1,
				conductivity_weight = 1,
				DIC_weight = 1,

				temperature_path = None,
				salinity_path = None,
				nitrates_path = None,
				phosphates_path = None,	
				conductivity_path = None,
				DIC_path = None,

				latitude = None,
				longitude = None,
				distance = None,
				):

		self.temperature = temperature
		self.salinity = salinity
		self.nitrates = nitrates
		self.phosphates = phosphates
		self.conductivity = conductivity
		self.DIC = DIC

		self.temperature_weight = temperature_weight
		self.salinity_weight = salinity_weight
		self.nitrates_weight = nitrates_weight
		self.phosphates_weight = phosphates_weight
		self.conductivity_weight = conductivity_weight
		self.DIC_weight = DIC_weight

		self.tpath = temperature_path
		self.spath = salinity_path
		self.npath = nitrates_path
		self.ppath = phosphates_path
		self.cpath = conductivity_path
		self.dpath = DIC_path

		self.latitude = latitude
		self.longitude = longitude 
		self.distance = distance

	def nan_helper(self, y):
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
	    return np.isnan(y), lambda z: z.nonzero()[0]


	def normalise_data(self, data, env_var):

		data = np.append(data, np.array([np.mean(env_var)]), 0)
		data = (data-np.min(data))/(np.max(data)-np.min(data))

		return (data[:-1], data[-1]) # (normed_data, normed_optimal)

	def f_temperature(self, data, env_var, weight):

		m = np.mean(env_var)
		t_min = np.nonzero(data <= env_var[0])
		t_max = np.nonzero(data > env_var[1])

		data[t_min] = np.e**(-2.3*((data[t_min]-m)/(env_var[0]-m)))
		data[t_max] = np.e**(-2.3*((data[t_max]-m)/(env_var[1]-m)))

		nans, x = self.nan_helper(data)
		data[nans] = np.interp(x(nans), x(~nans), data[~nans])

		return data*weight

	def f_salinity(self, data, env_var, weight):
		
		m = np.mean(env_var)
		s_gt5 = np.nonzero(data >= 5)
		s_lt5 = np.nonzero(data < 5)

		# gt5 break
		smin = np.nonzero(data[s_gt5] < m)
		smax = np.nonzero(data[s_gt5] >= m)

		data[smin] = 1-(((data[smin]-m)/(min(data)-m))**2.5)
		data[smax] = 1-(((data[smax]-m)/(max(data)-m))**2)

		# lt5 break
		data[s_lt5] = (data[s_lt5]-min(data))/(m-min(data))

		nans, x = self.nan_helper(data)
		data[nans] = np.interp(x(nans), x(~nans), data[~nans])

		return data*weight

	def f_nitrates(self, data, env_var, weight, micro=None, dic=None):
		
		ratio_n = np.nonzero(data/micro < 12)
		keq = dic/(data+micro)

		data[ratio_n] = (data[ratio_n].astype(int)-env_var[0])/(
			keq[ratio_n]+(data[ratio_n].astype(int)-env_var[0]))

		nans, x = self.nan_helper(data)
		data[nans] = np.interp(x(nans), x(~nans), data[~nans])

		return data*weight

	def f_phosphates(self, data, env_var, weight, micro=None):

		ratio_p = np.nonzero(data/micro > 20)

		p_int = np.nonzero(data[ratio_p] < env_var[1])
		data[p_int] = data[p_int].astype(int)/env_var[1]
		p_int = np.nonzero(data[ratio_p] > env_var[1])
		data[p_int] = 1

		nans, x = self.nan_helper(data)
		data[nans] = np.interp(x(nans), x(~nans), data[~nans])

		return data*weight

	def f_conductivity(self, data, env_var, weight):

		m = np.mean(env_var)
		data  = (data/m)*np.e**(1-(data/m))

		nans, x = self.nan_helper(data)
		data[nans] = np.interp(x(nans), x(~nans), data[~nans])

		return data*weight

	def f_DIC(self, data, env_var, weight):

		nans, x = self.nan_helper(data)
		data[nans] = np.interp(x(nans), x(~nans), data[~nans])

		return data*weight

	def get_data(self, **kwargs):

		'''
		returns a dictionary of (lat,lon) coordinates
		based on which data has the most entries.
		This is a massive hack, and computationally inefficient,
		but of these environmental variables, we're going to 
		establish which one has the most coordinated data, and then
		interpolate the others into vectors of the same length
		'''

		d = {}

		for var in kwargs:

			d[var] = {}

			with gzip.open(kwargs[var], mode='rb') as f:
				csv = pd.read_csv(f, header=1) # read
				csv.columns = ['lat','lon','dat']+[i for i in csv.columns[3:]] # rename
				lat, lon = csv.lat.values, csv.lon.values # take as rote 
				dat = csv.dat.interpolate().values # interpolate NaNs first

			for i in range(lat.shape[0]): # arbitrary
				d[var][(lat[i], lon[i])] = dat[i] # add

		# # Get the coordinates of the table with the most data:
		# ltab = [len(d[i]) for i in d]
		# coords = [*d.values()][ltab.index(max(ltab))].keys()

		_ = []
		for i in d:
			_+=[*d[i].keys()]
		_ = set(sorted(_))
		coords = {var:[] for var in d}
		for i in _:
			for var in d:
				try:
					coords[var].append(d[var][i])
				except:
					coords[var].append(np.NaN)

		for var in coords:
			data = np.array(coords[var])
			nans, x = self.nan_helper(data)
			data[nans] = np.interp(x(nans), x(~nans), data[~nans])
			# Interpolation sanity check:
			if np.argwhere(np.isnan(data)).any():
				print('NaN in data; failed to interpolate')
				exit()
			coords[var] = data

		# IMPORTANT STEP: zip the coordinates and the normalised data together
		# (note the norm values here are currently a massive hack to be revised)
		# Currently these are being normed with the inclusion of the 'optimal' values,
		# but probably best in future revisions to norm these without the opt. vals.

		nitrates = coords['nitrates'] # store
		phosphates = coords['phosphates'] #store
		DIC = coords['DIC']

		for var in coords:
			var_opt = eval('self.'+str(var)) # as in, optimal values in self.[var]
			# No getting around this next step: depending on the environmental
			# variable in question, we need to run it through it's respective
			# modlling function. Afterwards, we'll weight these by SGR coefficents

			var_weight = eval('self.'+str(var)+'_weight')
			var_func = eval('self.f_'+str(var))

			if var == 'nitrates':
				coords[var] = var_func(coords[var], var_opt, var_weight, micro=phosphates, dic=DIC)
			elif var == 'phosphates':
				coords[var] = var_func(coords[var], var_opt, var_weight, micro=nitrates)
			else:
				coords[var] = var_func(coords[var], var_opt, var_weight)

			norm, norm_opt = self.normalise_data(coords[var], var_opt)
			data = {i:j for i, j, in zip(_,norm)}

			d[var] = norm_opt # repurpose d to hold the normed optimal values
			coords[var] = data

		# at this stage, we have a dictionary coords which has the enviromental vars
		# as its keys, each of which is a dictionary of coordinate:value pairs.
		# However, we need to return the lists (suitability,lat,lon), so...

		# (And here's the most disguisting hack: simply take the average of the
		# RMSEs for all environmental vars...:)

		l = len(_)
		s = np.zeros(l) # these will form our suitablility indices

		for var in coords:
			if var == 'DIC': # keq equation only
				pass
			else:
				t = np.full(l, d[var])
				v = np.array([*coords[var].values()])

				s += np.sqrt(((v - t) ** 2))

		s /= l

		s = (s-min(s))/(max(s)-min(s)) # minmax morm

		lat = [i[0] for i in _]
		lon = [i[1] for i in _]

		return coords, s, lat, lon


	def new_map(self, data, lat, lon):

		fig = plt.figure(figsize=(8, 8))
		m = Basemap(projection='lcc', resolution='h', 
            lat_0=self.latitude, lon_0=self.longitude,
            width=self.distance, height=self.distance)
		m.shadedrelief()
		#m.drawcoastlines(color='blue',linewidth=1)
		m.drawcountries(color='gray',linewidth=1)
		m.drawstates(color='gray')
		m.scatter(lon,lat,latlon=True,
          c=data,s=10,
          cmap='YlGnBu_r', alpha=0.5)
		plt.colorbar(label=r'Suitability Index')
		plt.clim(max(data),min(data))
		plt.show()


	def main(self):

		data, suits, slats, slons = self.get_data(
			temperature = self.tpath,
			salinity = self.spath,
			nitrates = self.npath,
			phosphates = self.ppath,
			conductivity = self.cpath,
			DIC = self.dpath,
			)

		self.new_map(suits, slats, slons)


def findGeocode(location):
    try:
        geolocator = Nominatim(user_agent="shp")
        return geolocator.geocode(location)
    except GeocoderTimedOut:
        return findGeocode(location)    
  
    if findGeocode(location) != None:
        return findGeocode(location)
    else:
    	exit('Unable to find location!')


# Initial user prompts:

loc = input('Location to map: ')
dist = input('Close (c), near (n) or far (f) zoom (¼° grid): ')
_dist = {'c':(1.0E6,30), 'n':(5.0E6,45), 'f':(10.0E6,60)}
dist = _dist[dist[0].lower()]
loc = findGeocode(loc)
print(f'Calculating S. latissima suitablility indices for {loc}')
latitude, longitude = loc.latitude, loc.longitude
print(f'(this will take {dist[1]} seconds...)')

# path files to data:

cwd = os.getcwd()

spring_temperatures = os.path.join(cwd,'spring_temperatures.csv.gz')
spring_salinity = os.path.join(cwd,'spring_salinity.csv.gz')
spring_nitrates = os.path.join(cwd,'spring_nitrates.csv.gz')
spring_phosphates = os.path.join(cwd,'spring_phosphates.csv.gz')
spring_conductivity = os.path.join(cwd,'spring_conductivity.csv.gz')
spring_DIC = os.path.join(cwd,'spring_DIC.csv.gz') # for k_eq equation

shp = SHP(

	temperature = [5,15], # degC
	salinity = [24,35], # spu
	nitrates = [5,20], # μM NO
	phosphates = [8,8.5], # μM PO
	conductivity = [2,12], # S/m # note no real upper threshold; guesstimated from max
	DIC = [8,30],

	temperature_weight = 15-5,
	salinity_weight = 35-24,
	nitrates_weight = 20-5,
	phosphates_weight = 8.5-8,
	conductivity_weight = 12-2,
	DIC_weight = 30-8,

	temperature_path = spring_temperatures,
	salinity_path = spring_salinity,
	nitrates_path = spring_nitrates,
	phosphates_path = spring_phosphates,
	conductivity_path = spring_conductivity,
	DIC_path = spring_DIC, # for k_eq equation

	latitude = latitude,
	longitude = longitude,
	distance = dist[0],
	)

shp.main()