import glob
import os
import numpy as np
import pandas as pd
import pprint

from pyhdf.SD import SD, SDC


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


def dataset_dic(file):
    print(file.info())
    datasets_dic = file.datasets()
    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)


def grid_coordinate(limit,gridSize ):
    minlat = float(limit[0])
    maxlat = float(limit[1])
    minlon = float(limit[2])
    maxlon = float(limit[3])
    dx = gridSize
    xdim=int(1+((maxlon-minlon)/dx))
    ydim=int(1+((maxlat-minlat)/dx))
    grdlat=np.full([xdim,ydim],-1.0)
    grdlon=np.full([xdim,ydim],-1.0)
    for i in range(xdim):
        for j in range(ydim):
            grdlon[i,j]=dx*i+minlon
            grdlat[i,j]=dx*j+minlat
    return grdlat,grdlon 



def grid(limit, gsize, indata, inlat, inlon):
    dx = gsize
    dy = gsize
    minlat = float(limit[0])
    maxlat = float(limit[1])
    minlon = float(limit[2])
    maxlon = float(limit[3])
    xdim = int(1 + ((maxlon - minlon) / dx))
    ydim = int(1 + ((maxlat - minlat) / dy))
    sum_var  = np.zeros((xdim, ydim))
    count = np.zeros((xdim, ydim))
    avg_var = np.full([xdim, ydim], -1.0)

#pv= np.nan #_FillValue=999999 #0 #65535  #decide if it is needed to pass as parameter or other value 0 ? nan?

    #indata[indata < 0] = 0
    
    # print("-------------------------------------indata grid ini-------------------",np.min(indata),np.max(indata), "max min indata-----")
    # pprint.pprint(indata)
    # print("-------------------------------------fin indata grid ini-------------------",np.min(indata),np.max(indata), "max min indata-----")

    mask_re = np.where(indata != np.nan, 1, 0)  #con 0 tbn funcionaba  indata is a mask ...converting a masked element to nan

    # print('mask_re valuesn min, max:',mask_re, np.min(mask_re),np.max(mask_re))
    # pprint.pprint(mask_re)
    
    # print("-------------------------------------indata-------------------",np.min(indata),np.max(indata), "max min indata-----")
    # pprint.pprint(indata)
    # print("----------------------------fin---------indata-------------------")


    for ii in range(len(indata)):

        if (inlat[ii] >= minlat and inlat[ii] <= maxlat and inlon[ii] >= minlon and inlon[ii] <= maxlon):

            i = round((inlon[ii] - minlon) / dx)
            i = int(i)
            j = round((inlat[ii] - minlat) / dy)
            j = int(j)
            #sum_var[i, j] = sum_var[i, j] + indata[ii]
            sum_var[i, j] =  indata[ii]
            #count[i, j] += mask_re[ii]

    #count = np.ma.masked_equal(count, 0)  #### xq divide a la sumaa


    # print(count)
    avg_var = sum_var #/ count

    #print('------------------------------DATAFRAMA---------------------')
    #print(np.shape(avg_var))
    #x_flat = avg_var.reshape(avg_var.shape[0]*avg_var.shape[0],-1) #
    x_flat = np.reshape(avg_var, (avg_var.shape[0]*avg_var.shape[1],-1))

    #print(np.shape(avg_var))
 
    # df=pd.DataFrame(x_flat) 
    # count_nan = df[i].isnull().sum()
    # print ('In band values NaN: {}'.format( count_nan))  

    # df.describe().to_csv("avr before mask.csv")

    avg_var = np.ma.masked_equal(avg_var, -1)
  
    print(np.shape(avg_var))

    # x_flat = avg_var.reshape(-1,1) # 
    # df=pd.DataFrame(x_flat) 
    # count_nan = df[i].isnull().sum()
    # print ('In band  values NaN: {}'.format( count_nan))  

    # df.describe().to_csv("avr after mask.csv")


    # print("---------avg_var out grid---------",np.max(avg_var),np.min(avg_var),"finish max, min")
    # pprint.pprint(avg_var)
    

    return (avg_var)

def gridding_variable(variable, map_boundaries, gridSize, allLat, allLon):
    
    n_variables = len(variable)
    valid_var_grid=np.zeros((n_variables, 1001, 701)) ### como define it ???
    
    for i in range(n_variables):
        valid_var_grid[i] = grid(map_boundaries, gridSize, variable[i].flatten(), allLat.flatten(), allLon.flatten())
    

    return valid_var_grid 


def read_coordinate(file_g):
    lat_2D = file_g.select('Latitude')
    lon_2D=file_g.select('Longitude')
    lat=lat_2D.get()
    lon=lon_2D.get()
    latitude=lat #.flatten()
    longitude=lon #.flatten()
    return latitude, longitude

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.ticker import MaxNLocator

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


def read_level1_radiances(file_i, var_name):

    variable_sds= file_i.select(var_name)    
    variable = variable_sds.get()

    #get scale factor and fill value for reff field
    attributes = variable_sds.attributes()

    valid_var=np.zeros(np.shape(variable))

    fva=attributes["_FillValue"]
    FillValue = fva

    vra=attributes["valid_range"]
    valid_min = vra[0]        
    valid_max = vra[1]

    print("_FillValue",fva )
    print("min and max val", valid_min, valid_max)



    for i in range(len(variable)):
        print(i)

        data = variable[i].astype(np.double)
        invalid = np.logical_or(data > valid_max,
                            data < valid_min)
        invalid = np.logical_or(invalid, data == FillValue)


        
        data[invalid] = np.nan #_FillValue #np.nan
    

        scale_factor = attributes['radiance_scales'][i]
        add_offset  = attributes['radiance_offsets'][i]
        #get the valid var
        data= (data-add_offset)*scale_factor


        data = np.ma.masked_array(data, np.isnan(data)) 
        valid_var[i] = data

        # print('data min and max read func',np.min(data), np.max(data) )

        # print('valid_var min and max read func',np.min(valid_var), np.max(valid_var) )


        print("fin reading")
        #valid_var[i] = valid_var[i].flatten()
    return variable, valid_var  


def read_bands(file_i, var_name):

    variable_sds= file_i.select(var_name)    
    bands = variable_sds.get()

    return bands  

from netCDF4 import Dataset    # Note: python is case-sensitive!
import xarray as xr
import numpy as np
import argparse




def save_ncfile(name_variable, name_file, array, lat_array, lon_array, bands_array):

    
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    #ncfile = Dataset('/home/jvillarreal/GRID_DOM3_new.nc',mode='w',format='NETCDF4') 
    ncfile = Dataset(name_file,mode='w',format='NETCDF4') 
    print(ncfile)

    #print(ncfile)

    x_dim = ncfile.createDimension('Max_EV_frames:MODIS_SWATH_Type_L1B', array.shape[2])    # longitude axis
    y_dim = ncfile.createDimension('10*nscans:MODIS_SWATH_Type_L1B', array.shape[1])     # latitude axis

    bands_dim = ncfile.createDimension('bands', np.size(bands_array))    # longitude axis

    for dim in ncfile.dimensions.items():
        print(dim)

    ncfile.title='MODIS and simulations'
    #print(ncfile.title)

    lat = ncfile.createVariable('lat', np.float64, ('10*nscans:MODIS_SWATH_Type_L1B','Max_EV_frames:MODIS_SWATH_Type_L1B'))
    lat.units = 'degrees_north'
    lat.standard_name = "latitude"
    lat.long_name = 'latitude'
    lat.axis = "Y" 

    lon = ncfile.createVariable('lon', np.float64, ('10*nscans:MODIS_SWATH_Type_L1B','Max_EV_frames:MODIS_SWATH_Type_L1B'))
    lon.units = 'degrees_east'
    lon.standard_name = "longitude" 
    lon.long_name = 'longitude'
    lon.axis = "X"
    
    bands = ncfile.createVariable('bands', np.float64, ('bands'))
    bands.standard_name = "bands" 

                
    var= ncfile.createVariable(name_variable, np.float64, ('bands','10*nscans:MODIS_SWATH_Type_L1B','Max_EV_frames:MODIS_SWATH_Type_L1B'))   
    var.units = 'Watts/m^2/micrometer/steradian'  

    lat[:]=lat_array
    lon[:]=lon_array
    bands[:]=bands_array
    var[:]= array


    #close the Dataset.
    ncfile.close(); 
    print('Dataset was created!',name_file)





def save_data(name_variable, name_file, array, lat, lon, bands):
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


def dataframe_csv(variable, colum, out_file):
  ### input (a,b,c) a will be the columns of the dataframe
  # datafram  row = b*c, colum = a  
    print('dataframe', np.shape(colum), np.shape(variable))
    X_flated = variable.transpose(1,2,0).reshape(-1,variable.shape[0]) # 
    print(np.shape(X_flated))
    df=pd.DataFrame(X_flated) 
    for i in range(len(colum)):
        count_nan = df[i].isnull().sum()
        print ('In band {} values NaN: {}'.format(colum[i], count_nan))  

    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df.columns= colum

    df.describe().to_csv(out_file + ".csv")
    
    print("ok dataframe")

        # convert to Dataframe 


from matplotlib.pyplot import figure 

def plot_rgb_image(along_track, cross_trak, z, out_file, name_plot):
    
    norme = 0.8#0.4 # factor to increase the brightness ]0,1]

    rgb = np.zeros((along_track, cross_trak,3))

    rgb = z / norme

    rgb[ rgb > 1 ] = 1.0
    rgb[ rgb < 0 ] = 0.0

    fig = figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

    ax = fig.add_subplot(111)

    img = ax.imshow(np.fliplr(rgb), interpolation='nearest', origin='lower')
#    img = plt.imshow((rgb), interpolation='nearest', origin='lower')
    
    l = [int(i) for i in np.linspace(0,cross_trak,6)]
    plt.xticks(l, [i for i in reversed(l)], rotation=0, fontsize=11 )

    l = [int(i) for i in np.linspace(0,along_track,9)]
    plt.yticks(l, l, rotation=0, fontsize=11 )

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.title('', fontsize=16)

    #plt.show()    
    name_output = "{}/{}_RGB_image.png".format(out_file,name_plot) 
    fig.savefig(name_output) 

    fig.savefig(name_output) 
    print("Plotted RGB:",name_output)
    plt.close()          
 

def rgb_image(out_file, myd021km_file):
    #https://moonbooks.org/Jupyter/deep_learning_with_tensorflow_for_modis_multilayer_clouds/
    selected_sds = myd021km_file.select('EV_250_Aggr1km_RefSB')
    selected_sds_attributes = selected_sds.attributes()

    for key, value in selected_sds_attributes.items():
        if key == 'reflectance_scales':
            reflectance_scales_250_Aggr1km_RefSB = np.asarray(value)
        if key == 'reflectance_offsets':
            reflectance_offsets_250_Aggr1km_RefSB = np.asarray(value)

    sds_data_250_Aggr1km_RefSB = selected_sds.get()


    selected_sds = myd021km_file.select('EV_500_Aggr1km_RefSB')

    selected_sds_attributes = selected_sds.attributes()

    for key, value in selected_sds_attributes.items():
        if key == 'reflectance_scales':
            reflectance_scales_500_Aggr1km_RefSB = np.asarray(value)
        if key == 'reflectance_offsets':
            reflectance_offsets_500_Aggr1km_RefSB = np.asarray(value)

    sds_data_500_Aggr1km_RefSB = selected_sds.get()

    print( reflectance_scales_500_Aggr1km_RefSB.shape)


    data_shape = sds_data_250_Aggr1km_RefSB.shape

    along_track = data_shape[1]
    cross_trak = data_shape[2]

    z = np.zeros((along_track, cross_trak,3))

    z[:,:,0] = ( sds_data_250_Aggr1km_RefSB[0,:,:] - reflectance_offsets_250_Aggr1km_RefSB[0] ) * reflectance_scales_250_Aggr1km_RefSB[0]  
    z[:,:,1] = ( sds_data_500_Aggr1km_RefSB[1,:,:] - reflectance_offsets_500_Aggr1km_RefSB[1] ) * reflectance_scales_500_Aggr1km_RefSB[1]  
    z[:,:,2] = ( sds_data_500_Aggr1km_RefSB[0,:,:] - reflectance_offsets_500_Aggr1km_RefSB[0] ) * reflectance_scales_500_Aggr1km_RefSB[0] 
    
    #z[:,:,1] = ( sds_data_500_Aggr1km_RefSB[1,:,:] - reflectance_offsets_250_Aggr1km_RefSB[1] ) * reflectance_scales_500_Aggr1km_RefSB[1]  
    # R = z[:,:,0]
    # G_true = z[:,:,1]
    # B = 0.5 *(R + G_true)

        # Apply the gamma correction
       

    R = z[:,:,0]
    G = z[:,:,1]
    B = z[:,:,2]

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    gamma = 2.2
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)
    B = np.power(B, 1/gamma)

    # Calculate the "True" Green
    #G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)

    # The final RGB array :)
    RGB = np.dstack([R, G_true, B])

    
    plot_rgb_image(along_track, cross_trak, RGB, out_file, name_plot = "total")

    return z
    #https://proj.org/operations/projections/eqc.html  satpy