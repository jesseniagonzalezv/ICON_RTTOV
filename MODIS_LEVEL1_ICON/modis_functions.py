import glob
import os
import numpy as np
import pandas as pd
import pprint
from pyhdf.SD import SD, SDC
import numpy.ma as ma
from netCDF4 import Dataset    # Note: python is case-sensitive!
import xarray as xr
import argparse
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker

def dataset_dic(file):
    """
    Prints information about the given file and the datasets it contains.

    Arguments:
    - file -- An instance of the netCDF4.Dataset class representing a netCDF file.

    Returns: None
    - Prints -- Information about the file such as its format, dimensions variables 
    """
    print(file.info())
    datasets_dic = file.datasets()
    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)
        
def read_coordinate(file_g):
    """
    Reads the latitude and longitude coordinates from a netCDF file.

    Arguments:
    - file_g: A netCDF4.Dataset file.

    Returns:
    - latitude, longitude -- A tuple containing the latitude and longitude coordinate arrays extracted from the file.
    """
    lat_2D = file_g.select('Latitude')
    lon_2D=file_g.select('Longitude')
    lat=lat_2D.get()
    lon=lon_2D.get()
    latitude=lat #.flatten()
    longitude=lon #.flatten()
    return latitude, longitude


def grid_coordinate(limit, gridSize_lat, gridSize_lon):
    """
    Generates a 2D grid of latitude and longitude coordinates within a given geographic limit.

    Parameters:
    - limit: A list containing the minimum and maximum latitude and longitude values in the format [minlat, maxlat, minlon, maxlon].
    - gridSize_lat: The grid cell size in the latitude direction, in degrees.
    - gridSize_lon: The grid cell size in the longitude direction, in degrees.

    Returns:
    - A tuple containing two 2D arrays representing the latitude and longitude coordinates of the generated grid. i = longitude and j = latitude
    """
    minlat = float(limit[0])
    maxlat = float(limit[1])
    minlon = float(limit[2])
    maxlon = float(limit[3])
    dx_lon = gridSize_lon
    dy_lat = gridSize_lat

    xdim=int(1+((maxlon-minlon)/dx_lon))
    ydim=int(1+((maxlat-minlat)/dy_lat))
    grdlat=np.full([xdim,ydim],-1.0)
    grdlon=np.full([xdim,ydim],-1.0)
    for i in range(xdim):
        for j in range(ydim):
            grdlon[i,j]=dx_lon*i+minlon
            grdlat[i,j]=dy_lat*j+minlat
    return grdlat,grdlon 



def read_level1_array(file_i, var_name, type_variable):
    '''
    Function to obtain the radiances or reflectances values
        Based on https://hdfeos.org/zoo/LAADS/MYD021KM.A2002226.0000.006.2012033082925.hdf.py

    Argguments:
     - file_i : name of the MODIS file
     - var_name : name of the varibale to read ('EV_250_Aggr1km_RefSB', 'EV_500_Aggr1km_RefSB', 'EV_1KM_RefSB', 'EV_1KM_Emissive')
     - type_variable: "reflectance" or "radiance" 
    
    Return: 
     - variable_raw (ChxHxW): MODIS values with any modification, raw data (H - Height , W - Width , Ch - channel)
     - variable_calibrated (ChxHxW): MODIS values after offset and scale
    '''

    variable_sds= file_i.select(var_name)    
    variable_raw = variable_sds.get()

    #get scale factor and fill value for reff field
    attributes = variable_sds.attributes()

    variable_calibrated = np.zeros(np.shape(variable_raw))

    fva = attributes["_FillValue"]
    FillValue = fva

    vra=attributes["valid_range"]
    valid_min = vra[0]        
    valid_max = vra[1]

    #print("_FillValue",fva )
    #print("min and max val", valid_min, valid_max)


    for i in range(len(variable_raw)):
        #print(i)

        data = variable_raw[i].astype(np.double)
        invalid = np.logical_or(data > valid_max,
                            data < valid_min)
        invalid = np.logical_or(invalid, data == FillValue)


        #### this i am comment to test 
        data[invalid] = np.nan #_FillValue #np.nan
    
        #print("type_variable",type_variable)
        if (type_variable == "reflectance"):
            scale_factor = attributes['reflectance_scales'][i]
            add_offset  = attributes['reflectance_offsets'][i]
        
        elif (type_variable == "radiance"):
            scale_factor = attributes['radiance_scales'][i]
            add_offset  = attributes['radiance_offsets'][i]
        #get the valid var
        data= (data - add_offset) * scale_factor
        variable_calibrated[i] = np.ma.masked_array(data, np.isnan(data)) 

        # print('data min and max read func',np.min(data), np.max(data) )
        # print('variable_calibrated min and max read func',np.min(variable_calibrated), np.max(variable_calibrated) )
        #variable_calibrated[i] = variable_calibrated[i].flatten()
        
    return variable_raw, variable_calibrated  


def read_bands(file_i, var_name):
    """
    Reads a variable from a netCDF file representing the values of an n-dimensional array.

    Arguments:
    - file_i: AnnetCDF4.Dataset file.
    - var_name: The name of the variable to be read from the file.

    Returns:
    - A numpy array containing the values of the specified variable in the file.
    """
    variable_sds= file_i.select(var_name)    
    bands = variable_sds.get()

    return bands  



def dataframe_csv(variable, colum, out_file):
    """
    Converts a 3D numpy array to a pandas DataFrame and saves its summary statistics to a CSV file.

    Argument:
    - variable: A 3D numpy array representing an n-dimensional array of data.
    - colum: A list of strings representing the column names for the resulting DataFrame.
    - out_file: The name of the output file where the DataFrame summary statistics will be saved.

    Returns:
    - None.
    """
  ### input (a,b,c) a will be the columns of the dataframe
  # datafram  row = b*c, colum = a  
    print('dataframe', np.shape(colum), np.shape(variable))
    X_flated = variable.transpose(1,2,0).reshape(-1,variable.shape[0]) # 
    #print(np.shape(X_flated))
    df=pd.DataFrame(X_flated) 
    for i in range(len(colum)):
        count_nan = df[i].isnull().sum()
        print ('In band {} values NaN: {}'.format(colum[i], count_nan))  

    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df.columns= colum

    df.describe().to_csv(out_file + ".csv")
    
    print("ok dataframe")


def grid(limit,  gsize_lat, gsize_lon, indata, inlat, inlon):
    """
    Returns a gridded dataset of average values, given a set of input data and grid specifications.

    Arguments:
    - limit (list): A list containing the minimum and maximum values of latitude and longitude, in the form [minlat, maxlat, minlon, maxlon].
    - gsize_lat (float): The desired grid cell size in the latitude dimension, in decimal degrees.
    - gsize_lon (float): The desired grid cell size in the longitude dimension, in decimal degrees.
    - indata (ndarray): The input data to be gridded.
    - inlat (ndarray): The latitude values corresponding to each element of the input data.
    - inlon (ndarray): The longitude values corresponding to each element of the input data.

    Returns:
    - ndarray: A gridded dataset of average values, with dimensions (ydim, xdim), where xdim and ydim are determined by the specified grid cell sizes and latitude/longitude limits.
    """
    dx = gsize_lon
    dy = gsize_lat
    minlat = float(limit[0])
    maxlat = float(limit[1])
    minlon = float(limit[2])
    maxlon = float(limit[3])
    xdim = int(1 + ((maxlon - minlon) / dx))  # 589
    ydim = int(1 + ((maxlat - minlat) / dy))  # 628
    
    sum_var  = np.zeros((xdim, ydim))
    count = np.zeros((xdim, ydim))
    avg_var = np.empty([xdim, ydim])

    indata[indata < 0] = 0

    #mask_re = np.where(indata != np.nan, 1, 0)  #con 0 tbn funcionaba  indata is a mask ...converting a masked element to nan


    for ii in range(len(indata)):

        if (inlat[ii] >= minlat and inlat[ii] <= maxlat and inlon[ii] >= minlon and inlon[ii] <= maxlon):

            i = round((inlon[ii] - minlon) / dx)
            i = int(i)
            j = round((inlat[ii] - minlat) / dy)
            j = int(j)
            if (indata[ii] != np.nan):
                sum_var[i, j] = sum_var[i, j] + indata[ii]
                count[i, j] = count[i, j] +1 
            # sum_var[i, j] =  indata[ii]

    # print(count)
    count[count == 0] = np.nan

    #count = np.ma.masked_equal(count, 0)

    avg_var = sum_var/count


    #print(avg_var,count)
    x_flat = np.reshape(avg_var, (avg_var.shape[0]*avg_var.shape[1],-1))
    df=pd.DataFrame(x_flat) 
    count_nan = df.isnull().sum()
    # print ('In band values NaN: {}'.format( count_nan))  
    # avg_var = np.ma.masked_equal(avg_var, -1)
  
    # print("values of zeros in the x", (avg_var <= 0).sum())   #.flatten()
    # ma.set_fill_value(avg_var,  np.nan)
    # print ('In band values NaN: 2 {}'.format( count_nan))  


    # print(np.shape(avg_var))


    return (avg_var).transpose(1,0)

def gridding_variable(variable, map_boundaries, gsize_lat, gsize_lon, allLat, allLon):
    """
    Gridded variable values for multiple variables.
    
    Arguments:
    - variable (list): A list of 2D arrays containing variable values to be gridded.
    - map_boundaries (list): A list containing the lat/lon limits of the region to be gridded.
            The list should contain the following elements in the order: [minlat, maxlat, minlon, maxlon]
    - gsize_lat (float): Grid cell size for latitude.
    - gsize_lon (float): Grid cell size for longitude.
    - allLat (array): Array containing the latitude values of all the data points.
    - allLon (array): Array containing the longitude values of all the data points.

    Returns:
    - valid_var_grid (array): A 3D array containing the gridded values for all variables in the input list.
     The dimensions are (n_variables, ydim, xdim), where n_variables is the number of variables, ydim is the number of grid cells in the y (latitude) direction, and xdim is the number of grid cells in the x (longitude) direction.
    """
    dx = gsize_lon
    dy = gsize_lat
    minlat = float(map_boundaries[0])
    maxlat = float(map_boundaries[1])
    minlon = float(map_boundaries[2])
    maxlon = float(map_boundaries[3])
    xdim = int(1 + ((maxlon - minlon) / dx))  # 589
    ydim = int(1 + ((maxlat - minlat) / dy))  # 628
    
    # check next https://notebook.community/dennissergeev/classcode/notebooks/01_MODIS_L1B#Reproject-MODIS-L1B-data-to-a-regular-grid
    #https://notebook.community/dennissergeev/classcode/notebooks/01_MODIS_L1B#Scale-factor-and-offset-value
    n_variables = len(variable)
    valid_var_grid=np.empty((n_variables, ydim, xdim)) # 628, 589#np.shape(allLat)[0] =lon, np.shape(allLat)[1]=lat)) #1001, 701)) ### como define it ???
    # print('valid_var_grid', np.shape(valid_var_grid))
    for i in range(n_variables):
        valid_var_grid[i] = grid(map_boundaries,  gsize_lat, gsize_lon, variable[i].flatten(), allLat.flatten(), allLon.flatten())
    
        # print("obtained gridded var {}, shape {}".format(i, np.shape(valid_var_grid[i])))
        #print("3 values of zeros in the x", (valid_var_grid[i] <= 0).sum())   #.flatten()
        # print("values of nan {}, i {}".format(i, np.isnan(valid_var_grid[i]).sum()))

    return valid_var_grid 

    

def save_ncfile(name_file, lat_array, lon_array, rad_var, refl_var, bands_array): #type_variable
    """
    Creates a NetCDF4 file and saves the provided latitude, longitude, radiance, and reflectance variables as variables
    in the file.

    Arguments:
    - name_file (str): The name of the NetCDF4 file to create.
    - lat_array (ndarray): A 1D numpy array containing latitude values.
    - lon_array (ndarray): A 1D numpy array containing longitude values.
    - rad_var (ndarray): A 3D numpy array containing radiance values for each band at each grid cell. The radiance variable, with dimensions (chan, lat, lon).
    - refl_var (ndarray): A 3D numpy array containing reflectance values for each band at each grid cell. The reflectance variable, with dimensions (chan, lat, lon).

    - bands_array (ndarray): A 1D numpy array containing band values.

    Returns:
    - None
    - Creation of a netcdf file with MODIS information on the Germany area ref_total and rad_total are the 2 variables in this file.
    """
    
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    #ncfile = Dataset('/home/jvillarreal/GRID_DOM3_new.nc',mode='w',format='NETCDF4') 
    ncfile = Dataset(name_file,mode='w',format='NETCDF4') 
    #print(ncfile)

    x_dim = ncfile.createDimension('lon', rad_var.shape[2])    # longitude axis
    y_dim = ncfile.createDimension('lat', rad_var.shape[1])     # latitude axis

    bands_dim = ncfile.createDimension('chan', np.size(bands_array))    # longitude axis

    ncfile.title='Germany MODIS'
    #print(ncfile.title)
    # nlon=589
    # nlat=628
    # ncout.createDimension('lat',nlat)
    # ncout.createDimension('lon',nlon)
    # lat_o=ncout.createVariable('lat',np.float32,('lon','lat'))
    # lon_o=ncout.createVariable('lon',np.float32,('lon','lat'))


    lat = ncfile.createVariable('lat', np.float64, ('lat'))
    lat.units = 'degrees_north'
    lat.standard_name = "latitude"
    lat.long_name = 'latitude'
    lat.axis = "Y" 

    lon = ncfile.createVariable('lon', np.float64, ('lon'))
    lon.units = 'degrees_east'
    lon.standard_name = "longitude" 
    lon.long_name = 'longitude'
    lon.axis = "X"
    
    bands = ncfile.createVariable('chan', np.float64, ('chan'))
    bands.standard_name = "channels" 

                
    var= ncfile.createVariable("ref_total", np.float64, ('chan','lat','lon'))  
        
    var2= ncfile.createVariable("rad_total", np.float64, ('chan','lat','lon'))  

    #units_str = ' '.join(units_names)
    var.units = 'none' 
    var2.units = 'W/m^2/um/sr' 
    
#     if (type_variable == 'radiance'):
#         var.units = 'Watts/m^2/micrometer/steradian'  
    
#     elif (type_variable == 'reflectance'):
#         var.units = 'none or Watts/m^2/micrometer/steradian' 

    lat[:]=lat_array
    lon[:]=lon_array
    bands[:]=bands_array
    
    var[:]= refl_var
    var2[:]= rad_var

    #dataframe_csv(variable = array, colum =range(np.shape(array)[0]), out_file = name_file)

    #close the Dataset.
    ncfile.close(); 
    print(' ------------------- Dataset was created! ',name_file)



#####################################################

def read_units(file_i, var_name, type_units):

    var = file_i.select(var_name)

    # Get the units attribute
    units_var = var.attributes()['{}_units'.format(type_units)]
    size_var = len(var.attributes()['{}_scales'.format(type_units)])
    #print(units_var)
    units_list = [units_var] * size_var

    return units_list 

########################################

from matplotlib.pyplot import figure 

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).

    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.

    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.

    Examples
    --------
    >>> img = array([[ 91.06794177,   3.39058326,  84.4221549 ],
                     [ 73.88003259,  80.91433048,   4.88878881],
                     [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)

    """
    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


def scale_image(image, x, y):
    #scaled = np.zeros((along_track, cross_trak), dtype=np.uint8)
    scaled = np.zeros((np.shape(image)[0], np.shape(image)[1]), dtype=np.uint8)


    for i in range(len(x)-1):
        x1 = x[i]
        x2 = x[i+1]
        y1 = y[i]
        y2 = y[i+1]
        m = (y2 - y1) / float(x2 - x1)
        b = y2 - (m *x2)
        mask = ((image >= x1) & (image < x2))
        scaled = scaled + mask * np.asarray(m * image + b, dtype=np.uint8)

    mask = image >= x2
    scaled = scaled + (mask * 255)

    return scaled

import cartopy.crs as ccrs
from scipy.interpolate import griddata
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree


def plot_rgb_MODIS(myd021_filepath, myd03_filepath, outpath):
    '''
    Code based on https://www.moonbooks.org/Jupyter/plot_rgb_image_from_modis_myd021km_products/
    code to create the plot of the RGB - MODIS 
    Argument:
    myd021_filepath: path of the hdf file (MODIS LEVEL 1 - MYD021KM)
    myd03_filepath: path of the hdf has the lat, lon values (MODIS - MYD03)
    outpath: path of the folder to save the plot
    '''
     #   myd021_filepath = '/home/jvillarreal/Documents/phd/dataset/MYD021KM.A2013122.1140.061.2018046032403.hdf'
     #   myd03_filepath = '/home/jvillarreal/Documents/phd/dataset/MYD03.A2013122.1140.061.2018046005026.hdf'
        #myd021_file_name = '/home/b/b381589/MODIS_DATA/MOD021KM.A2013122.0955.061.2017298011523.hdf'

    file = SD(myd021_filepath, SDC.READ)

    # print(file.info())

    #################################################################################################################################
    # Print SDS names
    # ---------------

    datasets_dic = file.datasets()

    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)

    #################################################################################################################################
    # Get data - Load the three channels into appropriate R, G, and B variables
    # --------------------------------------------------------------

    selected_sds = file.select('EV_250_Aggr1km_RefSB')

    selected_sds_attributes = selected_sds.attributes()

    for key, value in selected_sds_attributes.items():
        if key == 'reflectance_scales':
            reflectance_scales_250_Aggr1km_RefSB = np.asarray(value)
        if key == 'reflectance_offsets':
            reflectance_offsets_250_Aggr1km_RefSB = np.asarray(value)

    sds_data_250_Aggr1km_RefSB = selected_sds.get()

    # --------------------------------------------------------------
    selected_sds = file.select('EV_500_Aggr1km_RefSB')

    selected_sds_attributes = selected_sds.attributes()

    for key, value in selected_sds_attributes.items():
        if key == 'reflectance_scales':
            reflectance_scales_500_Aggr1km_RefSB = np.asarray(value)
        if key == 'reflectance_offsets':
            reflectance_offsets_500_Aggr1km_RefSB = np.asarray(value)

    sds_data_500_Aggr1km_RefSB = selected_sds.get()

    data_shape = sds_data_250_Aggr1km_RefSB.shape

    # print(reflectance_scales_500_Aggr1km_RefSB.shape)
    #################################################################################################################################
    # Load the three channels into appropriate R, G, and B variables
    # --------------------------------------------------------------

    along_track = data_shape[1]
    cross_track = data_shape[2]

    z = np.zeros((along_track, cross_track, 3))

    # Red channel
    z[:, :, 0] = (sds_data_250_Aggr1km_RefSB[0, :, :] - reflectance_offsets_250_Aggr1km_RefSB[0]) * \
      reflectance_scales_250_Aggr1km_RefSB[0]

    # Green channel
    z[:, :, 1] = (sds_data_500_Aggr1km_RefSB[1, :, :] - reflectance_offsets_500_Aggr1km_RefSB[1]) * \
      reflectance_scales_500_Aggr1km_RefSB[1]

    # Blue channel
    z[:, :, 2] = (sds_data_500_Aggr1km_RefSB[0, :, :] - reflectance_offsets_500_Aggr1km_RefSB[0]) * \
      reflectance_scales_500_Aggr1km_RefSB[0]

    # print('min RGB:',np.min(rgb),'max RGB:',np.max(rgb))

    # fig = figure(num=None, figsize=(12, 10), dpi=100, facecolor=fig_style_dict['facecolor'], edgecolor='k')
    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(111)

    # ax.imshow(np.fliplr(rgb), interpolation='nearest', origin='lower')
    # plt.show()
    # plt.close()
    # sys.exit()

    #################################################################################################################################
    #
    # --------------------------------------------------------------

    # myd03_file_name = '/home/b/b381589/MODIS_DATA/MOD03.A2013122.0955.061.2017298002914.hdf'

    myd03 = SD(myd03_filepath, SDC.READ)

    datasets_dic = myd03.datasets()

    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)

    myd03_Latitude = myd03.select('Latitude')
    myd03_Longitude = myd03.select('Longitude')

    myd03_Latitude_data = myd03_Latitude.get()
    myd03_Longitude_data = myd03_Longitude.get()

    myd03_Latitude_data = np.fliplr(myd03_Latitude_data)
    myd03_Longitude_data = np.fliplr(myd03_Longitude_data)

    myd03_Latitude_shape = myd03_Latitude_data.shape

    # print('myd03_Latitude_shape',myd03_Latitude_shape)

    norme = 1 #0.2 #0.8  # 0.2 # factor to increase the brightness ]0,1] in the simulations i didnt apply norm

    rgb = np.zeros((along_track, cross_track, 3))

    rgb = z / norme

    rgb[rgb > 1] = 1.0
    rgb[rgb < 0] = 0.0
    
    # ================= option 1 ======= increase the resolution
#     x = np.array([0,  30,  60, 120, 190, 255], dtype=np.uint8)
#     y = np.array([0, 110, 160, 210, 240, 255], dtype=np.uint8)
        
#     z_color_enh = np.zeros((along_track, cross_trak,3), dtype=np.uint8)
#     z_color_enh[:,:,0] = scale_image(bytescale(z[:,:,0]), x, y, along_track, cross_trak)
#     z_color_enh[:,:,1] = scale_image(bytescale(z[:,:,1]), x, y,along_track, cross_trak)
#     z_color_enh[:,:,2] = scale_image(bytescale(z[:,:,2]), x, y,along_track, cross_trak)
#     z = z_color_enh / 256.0
    
    # ================ option 2 ======== increase the resolution
    gamma = 2.2
    z = np.power(rgb, 1 / gamma)
    # ========================

    # norme = 0.7 # factor to increase the brightness ]0,1]
    # z = z/norme

    z = np.fliplr(z)

    proj = ccrs.PlateCarree()  # PlateCarree #Mercator

    along_track = myd03_Latitude_shape[0]
    cross_track = myd03_Latitude_shape[1]

    lat_long_grid = proj.transform_points(
    x=myd03_Longitude_data,
    y=myd03_Latitude_data,
    src_crs=proj)

    # print('lat_long_grid shape',lat_long_grid.shape)
    # print('lat_long_grid',lat_long_grid)

    x_igrid = lat_long_grid[:, :, 0]  ## long
    y_igrid = lat_long_grid[:, :, 1]  ## lat

    # print('x_igrid.shape',x_igrid.shape)

    # print('min myd03_Latitude_data',np.min(myd03_Latitude_data))
    # print('max myd03_Latitude_data',np.max(myd03_Latitude_data))

    # print('min myd03_Longitude_data',np.min(myd03_Longitude_data))
    # print('min myd03_Longitude_data',np.max(myd03_Longitude_data))

    # geod = ccrs.Geodetic()
    geod = ccrs.PlateCarree()

    xul, yul = proj.transform_point(
    x=myd03_Longitude_data[0, 0],
    y=myd03_Latitude_data[0, 0],
    src_crs=geod)

    xlr, ylr = proj.transform_point(
    x=myd03_Longitude_data[myd03_Latitude_shape[0] - 1, myd03_Latitude_shape[1] - 1],
    y=myd03_Latitude_data[myd03_Latitude_shape[0] - 1, myd03_Latitude_shape[1] - 1],
    src_crs=geod)

    print(xul, xlr, yul, ylr)

    xul = np.min(myd03_Longitude_data)
    xlr = np.max(myd03_Longitude_data)

    yul = np.min(myd03_Latitude_data)
    ylr = np.max(myd03_Latitude_data)

    z_igrid_01 = np.zeros((along_track, cross_track))
    z_igrid_02 = np.zeros((along_track, cross_track))
    z_igrid_03 = np.zeros((along_track, cross_track))

    z_igrid_01[:, :] = z[:, :, 0]
    z_igrid_02[:, :] = z[:, :, 1]
    z_igrid_03[:, :] = z[:, :, 2]

    x1_igrid = x_igrid.ravel()
    y1_igrid = y_igrid.ravel()

    z_igrid_01 = z_igrid_01.ravel()
    z_igrid_02 = z_igrid_02.ravel()
    z_igrid_03 = z_igrid_03.ravel()

    xy1_igrid = np.vstack((x1_igrid, y1_igrid)).T
    #    xi, yi = np.mgrid[xul:xlr:1000j, yul:ylr:1000j]
    xi, yi = np.mgrid[xul:xlr:628j, yul:ylr:589j]

    z_01 = griddata(xy1_igrid, z_igrid_01, (xi, yi), method='nearest')
    z_02 = griddata(xy1_igrid, z_igrid_02, (xi, yi), method='nearest')
    z_03 = griddata(xy1_igrid, z_igrid_03, (xi, yi), method='nearest')

    THRESHOLD = 0.2

    tree = cKDTree(xy1_igrid)
    arr_x = _ndim_coords_from_arrays((xi, yi))
    dists, indexes = tree.query(arr_x)

    z_01[dists > THRESHOLD] = np.nan
    z_02[dists > THRESHOLD] = np.nan
    z_03[dists > THRESHOLD] = np.nan

    rgb_projected = np.zeros((628, 589, 3))

    rgb_projected[:, :, 0] = z_01[:, :]
    rgb_projected[:, :, 1] = z_02[:, :]
    rgb_projected[:, :, 2] = z_03[:, :]

    # print('rgb_projected.shape:',rgb_projected.shape)

    whereAreNaNs = np.isnan(rgb_projected)
    rgb_projected[whereAreNaNs] = 0.

    # print('rgb_projected:',rgb_projected)

    # print('max_rgb_projected:',np.max(rgb_projected))

    min_long = 4.5  # np.min(myd03_Longitude_data)
    max_long = 14.5  # np.max(myd03_Longitude_data)
    min_lat = 47.599  # np.min(myd03_Latitude_data)
    max_lat = 54.5  # np.max(myd03_Latitude_data)
 #   map_boundaries[0] = 47.599 #llcrnrlat


    plt.figure(figsize=(10, 10))

    proj = ccrs.PlateCarree()  # Mercator

    offset = 0.0

    ease_extent = [min_long - offset,
        max_long + offset,
        min_lat - offset,
        max_lat + offset]

    ax = plt.axes(projection=proj)
    # ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(ease_extent, crs=proj)
    # ax.set_extent(ease_extent, crs=ccrs.PlateCarree())
    # ax.set_extent([4.5, 14.5, 47.8, 54.5], crs=ccrs.Mercator()) #PlateCarree

    swe_extent = [xul, xlr, yul, ylr]
    # swe_extent = [4.5, 14.5, 47.8, 54.5]

    ax.imshow(np.rot90(np.fliplr(rgb_projected)), extent=swe_extent, transform=proj, origin='lower',
    aspect='1.5')  # aspect=1.7 #aspect=9./6.

    # ax.gridlines(color='gray', linestyle='--')

    # ax.coastlines()
    # ax.coastlines(resolution='10m', color='black', linewidth=1)

    # plt.tight_layout()

    # plt.show()

    # sys.exit()

    # ax.coastlines(resolution='10m', color='black', linewidth=1)
    # ax.add_feature(ccrs.cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', color='black', linewidth=1)) #name='admin_0_boundary_lines_land'
    # ax.add_feature(ccrs.cartopy.feature.COASTLINE, resolution='10m', color='black', linewidth=1)
    # ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.5)
    ax.add_feature(ccrs.cartopy.feature.BORDERS, color='black', linewidth=1)
    ax.add_feature(ccrs.cartopy.feature.COASTLINE, color='black', linewidth=1)

    ax.set_xticks([5., 6., 7., 8., 9., 10., 11., 12., 13., 14.], crs=ccrs.PlateCarree())  # PlateCarree
    ax.set_yticks([48., 49., 50., 51., 52., 53., 54.], crs=ccrs.PlateCarree())  # PlateCarree
    ax.set_xticklabels([5., 6., 7., 8., 9., 10., 11., 12., 13., 14.], color='black', fontsize=16)
    ax.set_yticklabels([48., 49., 50., 51., 52., 53., 54.], color='black', fontsize=16)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    title = 'Real_MODIS_aqua_true_color_image_05022013_1140utc'
    # title = 'Real_MODIS_Terra_true_color_image_05022013_0955utc_irs_2022.png'
    plt.title('True Color from AQUA - MODIS  (Real data)', loc='center', fontweight='bold', fontsize=20)
    # print("almost")
    plt.savefig("{}/{}.png".format(outpath, title), dpi=600)
    print("{}/{}.png".format(outpath, title))
    plt.show()

    # file.close()
    # myd03.close()

########################################## Not used ######################################

def plot_rgb_image(along_track, cross_trak, z, out_file, name_plot, lat, lon):
        #https://moonbooks.org/Jupyter/deep_learning_with_tensorflow_for_modis_multilayer_clouds/
# check it https://moonbooks.org/Jupyter/plot_rgb_image_from_modis_myd021km_products/
    norme = 0.7#0.4 # factor to increase the brightness ]0,1]

    rgb = np.zeros((along_track, cross_trak,3))

    rgb = z / norme

    rgb[ rgb > 1 ] = 1.0
    rgb[ rgb < 0 ] = 0.0

#######
    x = np.array([0,  30,  60, 120, 190, 255], dtype=np.uint8)
    y = np.array([0, 110, 160, 210, 240, 255], dtype=np.uint8)
        
    z_color_enh = np.zeros((along_track, cross_trak,3), dtype=np.uint8)
    z_color_enh[:,:,0] = scale_image(bytescale(z[:,:,0]), x, y, along_track, cross_trak)
    z_color_enh[:,:,1] = scale_image(bytescale(z[:,:,1]), x, y,along_track, cross_trak)
    z_color_enh[:,:,2] = scale_image(bytescale(z[:,:,2]), x, y,along_track, cross_trak)

########3

    fig_style_dict = {}

    #fig_style_dict['facecolor'] = 'white'
#     fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

    fig = figure(num=None, figsize=(12, 10), dpi=100,  edgecolor='k') #facecolor=fig_style_dict['facecolor'],
    ax = fig.add_subplot(111)
    

    if name_plot == "simulation":
        m = Basemap(projection='merc', llcrnrlon=4.5, llcrnrlat=47.8, urcrnrlon=14.5,urcrnrlat=54.5, ax=ax, resolution='l') #Germany f
#         m.drawcoastlines(linewidth=0.8)
#         m.drawcountries()
#         m.drawstates()
#         m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
#         m.drawmeridians(np.arange(-180., 181., 45.), labels=[0, 0, 0, 1])
    
        m.imshow(z_color_enh, interpolation='nearest', origin='lower')
    

        l = [int(i) for i in np.linspace(0,cross_trak,6)]
        plt.xticks(l, [i for i in reversed(l)], rotation=0, fontsize=11 )

        l = [int(i) for i in np.linspace(0,along_track,9)]
        plt.yticks(l, l, rotation=0, fontsize=11 )


    if name_plot == "simulation":
        plt.title('Simulation RGB reflectance', fontsize=16)

    # ax.set_yticklabels([])
    # ax.set_xticklabels([])

    plt.tight_layout()

    #plt.show()    
    name_output = "{}/{}_RGB_image.png".format(out_file,name_plot) 
    fig.savefig(name_output) 

    fig.savefig(name_output) 
    print("Plotted RGB:",name_output)
    plt.close()          
    

def get_filenames(path_dataset, file_type, output_filename):
    '''Save the filenames per line 
        Input: path of the dataset to read
        Outpur: .txt file with the names
    '''
    os.chdir(path_dataset)
    output_file_path=os.sep.join([path_dataset,output_filename])
    hdf_files=glob.glob(file_type)
    with open(output_file_path, "w") as output:
        for name_file in hdf_files:
            s = "".join(map(str,name_file))
            print(s)
            output.write((s)+'\n')
    print("files {} saved in {}".format(file_type, output_file_path))


def get_data_geolocation_names(path_dataset, filename_datas, geolocation_filenames):
    get_filenames(path_dataset=path_dataset,file_type= filename_datas, output_filename="file_list.txt")
    get_filenames(path_dataset=path_dataset ,file_type= geolocation_filenames, output_filename="geolocation_files_list.txt")

                            

def save_data(name_variable, name_file, array, lat, lon, bands):
    # check next https://clouds.eos.ubc.ca/~phil/courses/atsc301/coursebuild/html/modis_multichannel.html
    fileName=name_file  #https://clouds.eos.ubc.ca/~phil/courses/atsc301/coursebuild/html/modis_level1b_read.html
    print(fileName)
    filehdf = SD(fileName, SDC.WRITE | SDC.CREATE)

    # Create a dataset
    sds = filehdf.create(name_variable, SDC.FLOAT64, array.shape)
    print(np.shape(sds))

    sds2 = filehdf.create("bands", SDC.FLOAT64, np.shape(bands)) #bands.shape)
    sds3 = filehdf.create("lat", SDC.FLOAT64, lat.shape)
    sds4 = filehdf.create("lon", SDC.FLOAT64, lon.shape)

    # Fill the dataset with a fill value
    sds.setfillvalue(0)

    # Set dimension names
    dim1 = sds.dim(0)
    dim1.setname("bands")
    dim2 = sds.dim(1)
    dim2.setname("10*nscans:MODIS_SWATH_Type_L1B") #"lon") #x
    dim3 = sds.dim(2)
    dim3.setname("Max_EV_frames:MODIS_SWATH_Type_L1B") #"lat") #y

    # sds3.dim(0).setname('\10\*nscans\:MODIS_SWATH_Type_L1B') #("x")
    # sds3.dim(1).setname('Max_EV_frames\:MODIS_SWATH_Type_L1B')  # ("y")

    # sds4.dim(0).setname('\10\*nscans\:MODIS_SWATH_Type_L1B') #("x")
    # sds4.dim(1).setname('Max_EV_frames\:MODIS_SWATH_Type_L1B')  #("y")

    sds2.dim(0).setname("index")

    # Assign an attribute to the dataset
    sds.units =  'Watts/m^2/micrometer/steradian' #"W/m^2/micron/sr"

    # Write data
    sds[:] = array
    sds2[:] = bands
    sds3[:] = lat
    sds4[:] = lon

   # print(np.shape(sds))

    # Close the dataset
    sds.endaccess()
    sds2.endaccess()
    sds3.endaccess()
    sds4.endaccess()


    # Flush and close the HDF file
    filehdf.end()            
                            
                            
# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap
# import numpy as np
# from matplotlib.ticker import MaxNLocator

def visulize_sat(variable, bands, lat, lon, cbar_label,
             figure_name, map_limit):
    # fs_titel = 20
    # fs_label = 20

    #to plot > 1 figures


    #################3
    # cmap = [(0.0,0.0,0.0)] + [(cm.jet(i)) for i in range(1,256)]
    # cmap = mpl.colors.ListedColormap(cmap)
    # bounds = bounds
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # cmap_2 = LinearSegmentedColormap.from_list("", ["white","lightskyblue","steelblue","green","yellowgreen","yellow","gold","red","firebrick","darkred"])
    # cmap_2 = LinearSegmentedColormap.from_list("",["lightskyblue", "steelblue", "green", "yellowgreen", "yellow",
    #                                             "gold", "red", "firebrick", "darkred"])
    # cmap_test=plt.get_cmap('jet')
    # m = Basemap(ax=ax, projection='merc', llcrnrlat= map_limit[0], urcrnrlat=map_limit[1],\
    #        llcrnrlon=map_limit[2], urcrnrlon=map_limit[3], resolution='c')
    # m.drawmeridians(np.arange(-180., 180.,2.), linewidth=1.2, labels=[0, 0, 0, 1], color='grey', zorder=3, latmax=90,
    #              )
    # m.drawparallels(np.arange(0., 85., 1.), linewidth=1.2, labels=[1, 0, 0, 0], color='grey', zorder=2, latmax=90,
    #               )
    # m.drawcoastlines()
    # m.drawcountries()
    # x, y = m(longrid, latgrid)
    # ax.set_title(titel_figure, fontsize=fs_titel)
    # l1=m.pcolormesh(x, y, var, cmap=cmap_test, norm=norm)
    # cbar = plt.colorbar(l1, ax=ax)
    # cbar_bounds = bounds
    # cbar_ticks =  bounds
    # cbar.set_label(cbar_label, fontsize= fs_label)
    # cbar.ax.tick_params(labelsize='xx-large')
    ###########################33
    n_bands = len(bands) #np.size(bands)
    print(n_bands)

    if(n_bands>8):
       nrows = 2
    else:
       nrows = 1

    ncols = n_bands // nrows + (n_bands % nrows > 0)

    #fig,axes = plt.subplots(nrows,ncols) #,figsize = (32,20))

    fig = plt.figure(figsize=(5*ncols, 5*nrows))   #(4*ncols, 4*nrows)) #for the subset
   # fig, axes = plt.subplots(nrows,ncols, figsize=(7, 7)) #width of 15 inches and 7 inches in height.

    #fig, axes = plt.subplots(nrows,ncols, figsize=(32, 4)) #width of 15 inches and 7 inches in height.

    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    label= cbar_label #'$Radiances/(W\,m^{-2}\,\mu m^{-1}\,sr^{-1})$'
    
    #axes = axes.ravel()
    for i in range(n_bands):
        
        #ax = axes[i]
        ax = plt.subplot(nrows, ncols, i + 1)
        print('ploting band {}'.format(bands[i]))

        map = Basemap(projection='merc',llcrnrlon=map_limit[2],llcrnrlat=map_limit[0],urcrnrlon=map_limit[3],urcrnrlat=map_limit[1],resolution='f',ax=ax) #Germany
        map.drawcoastlines()
        map.drawcountries()
        map.drawstates()
        map.drawparallels(np.arange(-90.,91.,1.),labels=[1,0,0,1],fontsize=10)
        map.drawmeridians(np.arange(-180.,181.,2.),labels=[1,0,0,1],fontsize=10)

        # lat = ds_array['lat']
        # lon = ds_array['lon']

        #lons,lats = np.meshgrid(lon,lat) #manoosh no lo usa ver para q sirve????
        # x,y = map(lons,lats)
        x,y = map(lon,lat)



        data = variable[i]#(chan, lat, lon) #check it when it is nan or inf np.ma.masked_where((ds_array[variable][col]< 2000), ds_array[variable][col])                
        
        print("data visuali ------------", np.min(data),np.max(data))
        pprint.pprint(data)

        #_FillValue= 999999# 0 #65535  #decide if it is needed to pass as parameter
        
        data = np.ma.masked_equal(data, 0) # np.ma.masked_array(data,np.isnan(data)) # data== _FillValue)
                
        print("data visuali 2------------", np.min(data),np.max(data))

        pprint.pprint(data)

        #levels = MaxNLocator(nbins=15).tick_values(np.nanmin(data),np.nanmax(data))

        #print(levels.max(), levels.min())
        extend = 'both' #min,max,both,neither
        cmap=plt.get_cmap('jet') #du bleu au rouge
                        
        #cs = map.contourf(x,y, y_filtered,levels,extend=extend,cmap=cmap) 
        cs = map.pcolormesh(x,y,data, cmap=cmap,shading='auto')
    
        cbar = fig.colorbar(cs, ax=ax,label=label,shrink=0.75) #location="right",pad="5%",ticks=[270,275,280,285,290,295,300],
        cbar.ax.tick_params(size=0,labelsize=10)

        ax.set_title('MODIS Channel %d (11:40/2/5/2013)'% (bands[i]),fontsize=14)
        ax.set_xlabel('Longitude', labelpad=20,fontsize=14)
        ax.set_ylabel('Latitude', labelpad=33,fontsize=14)

    # fig.delaxes(axes[-1])
    # fig.delaxes(axes[-2])
    # fig.delaxes(axes[-3])
    # fig.delaxes(axes[-4])

    plt.tight_layout()
   
    fig.savefig("{} in bands {}.png".format(figure_name, bands)) 
    plt.close()                               
########################################## End ot used ####################################


