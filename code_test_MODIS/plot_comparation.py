
import xarray as xr
import numpy as np
import argparse
import pprint

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_MODIS_simulation(variable, band, lat, lon, title_subplot, map_limit, fig, ax, max_colorbar, min_colorbar,type_variable):

    #axes = axes.ravel()
        if (type_variable == "refl_emis" and (band >= 1 and band <= 20 or band == 26)):
            cbar_label = 'none'
        else:
            cbar_label = 'Radiance $(W\,m^{-2}\,sr^{-1}\,\mu m^{-1})$'

        print(' ================= ploting band {} ================= '.format(band))

        map = Basemap(projection='merc',llcrnrlon=map_limit[2],llcrnrlat=map_limit[0],urcrnrlon=map_limit[3],urcrnrlat=map_limit[1],resolution='f',ax=ax) #Germany
        map.drawcoastlines()
        map.drawcountries()
        map.drawstates()
        map.drawparallels(np.arange(-90.,91.,1.),labels=[1,0,0,1],fontsize=10)
        map.drawmeridians(np.arange(-180.,181.,2.),labels=[1,0,0,1],fontsize=10)

        if(title_subplot == "RTTOV (12:30/2/5/2013)-Channel:"):
            lon,lat = np.meshgrid(lon,lat) ###funcionara sin esto revisar !!!!!!!!!!!!!!1
        x,y = map(lon,lat)  

        data = variable#(chan, lat, lon) #check it when it is nan or inf np.ma.masked_where((ds_array[variable][col]< 2000), ds_array[variable][col])                
        
        print(" ================= RTTOV data min and max: ", np.min(data),np.max(data))
        #pprint.pprint(data)

        #_FillValue= 999999# 0 #65535  #decide if it is needed to pass as parameter
        
        if(title_subplot == "MODIS (11:40/2/5/2013)-Channel:"):
            data = np.ma.masked_equal(data, 0) # np.ma.masked_array(data,np.isnan(data)) # data== _FillValue)
                
        print(" ================= MODIS data min and max: ", np.min(data),np.max(data))

        #pprint.pprint(data)

        #extend = 'both' #min,max,both,neither
        cmap=plt.get_cmap('jet') #du bleu au rouge
                        
        #cs = map.contourf(x,y, y_filtered,levels,extend=extend,cmap=cmap) 

        print(' ================= PLOT colobar min and max', min_colorbar, max_colorbar)
        cs = map.pcolormesh(x,y,data, cmap=cmap,shading='auto', vmin = min_colorbar, vmax = max_colorbar)
    
        cbar = fig.colorbar(cs, ax=ax,label=cbar_label,shrink=0.75) #location="right",pad="5%",ticks=[270,275,280,285,290,295,300],
        cbar.ax.tick_params(size=0,labelsize=10)

        ax.set_title("{}{}".format(title_subplot, band),fontsize=14)
        ax.set_xlabel('Longitude', labelpad=20,fontsize=14)
        ax.set_ylabel('Latitude', labelpad=33,fontsize=14)

        


    



       # visulize_sat(grid_valid_variable, bands_variable, gr_lat, gr_lon, cbar_label, titel, map_boundaries )

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--rttov-path-refl-emmis', type = str, default = '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-131-data-icon-1to19-26-T12.nc', help = 'Path of the dataset with only reflectances 1-19 and 26')
    #arg('--rttov-path', type = str, default = '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-msg-1-to-36.nc', help = 'Path of the dataset')
    arg('--rttov-path-rad', type = str, default = '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc', help = 'Path of the dataset with only radiances')
        
    arg('--MODIS-path', type = str, default = '/home/jvillarreal/Documents/phd/dataset/MODIS_Germany_radiances_MYD021KM.A2013122.1140.061.2018046032403.nc', help = 'Path of the output' )
    arg('--path-output', type = str, default = '/home/jvillarreal/Documents/phd/output', help = 'Path of the output' )
    arg('--type-variable', type = str, default = 'radiance', help = 'radiance, refl_emiss, rgb' )

        
    args = parser.parse_args()

    
    map_boundaries=np.zeros(4)
    map_boundaries[0] = 47.5 #llcrnrlat
    map_boundaries[1] = 54.5 # 496 #urcrnrlatMODIS_variable
    map_boundaries[2] = 4.5 #llcrnrlon
    map_boundaries[3] = 14.5 #496 #urcrnrlon
    
    rttov_ds_rad = xr.open_dataset(args.rttov_path_rad).compute()
    rttov_ds_refl_emmi = xr.open_dataset(args.rttov_path_refl_emmis).compute()

    MODIS_ds = xr.open_dataset(args.MODIS_path).compute()
    
    rttov_variable = np.zeros((np.shape(rttov_ds_rad['Y'].values)))
    print( "MODIS shape:", np.shape(MODIS_ds['MODIS_Germany_refl_emis'].values), "RTTOV Simulation shape:", np.shape(rttov_variable))

    MODIS_variable = np.zeros((3, np.shape(MODIS_ds['MODIS_Germany_refl_emis'].values)[1], np.shape(MODIS_ds['MODIS_Germany_refl_emis'].values)[2]))

    if (args.type_variable == "refl_emis"):
        MODIS_variable = MODIS_ds['MODIS_Germany_refl_emis']
#         rttov_variable =rttov_ds['bt_refl_total'][:3]
        rttov_variable[:19] = rttov_ds_refl_emmi['bt_refl_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
        rttov_variable[19:25] = rttov_ds_rad['Y'][19:25]
        rttov_variable[25] = rttov_ds_refl_emmi['bt_refl_total'][19] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
        rttov_variable[26:36] = rttov_ds_rad['Y'][26:36]

    elif (args.type_variable == "radiance"):
        MODIS_variable =MODIS_ds['MODIS_Germany_radiances']  
        rttov_variable =rttov_ds_rad['Y'] #'y' neceisto cambiar el nombre de esta variable

    
    
    MODIS_bands =MODIS_ds['bands'].values

    rttov_bands =rttov_ds_rad['chan'].values  ##check it becasue i am using 2 files

#    rttov_variable =rttov_ds['Y_clear'][:]
    MODIS_lat = MODIS_ds['lat'].values
    MODIS_lon = MODIS_ds['lon'].values

    rttov_lat = rttov_ds_rad['lat'].values
    rttov_lon = rttov_ds_rad['lon'].values
    n_bands = len(MODIS_bands)*2 #np.size(bands)
    #ncols = 2 #
    nrows = 2 #
    #print(n_bands)

    #nrows = n_bands // ncols + (n_bands % ncols > 0) #tbn pujede ser entrada
    ncols = n_bands // nrows + (n_bands % nrows > 0) #tbn pujede ser entrada

    fig = plt.figure(figsize=(5*ncols, 5*nrows))   
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    position =list(range(1,n_bands+1)) #[1:76]
    position_rttov = position[:39] #position[::2]
    position_MODIS = position[38:] #[1::2]


    for i in range(len(MODIS_bands)): 
    #for i in range(2):
        bnd=MODIS_bands[i].astype(int) # hacer que cuando sea igual al indice seleccionar ese cuando la dim chan ==i


        variable = np.ma.masked_array(MODIS_variable[i],  np.isnan(MODIS_variable[i]))
        position =position_MODIS[i]
        

        variable_simulation = np.ma.masked_array(rttov_variable[bnd-1], np.isnan(rttov_variable[bnd-1])) 
            
        min_colorbar = np.min([np.min(variable), np.min(variable_simulation)])
        max_colorbar = np.max([np.max(variable), np.max(variable_simulation)])
        # print('============modis colobar min and max',np.min(MODIS_variable[i]))
        # print('============rttov colobar min and max', np.min(rttov_variable[bnd-1]))

        # print('============code colobar min and max', min_colorbar, max_colorbar)

            
        
        title_subplot= "MODIS (11:40/2/5/2013)-Channel:"
        ax = plt.subplot(nrows, ncols, position)
        plot_MODIS_simulation(variable = variable, #in the below area there are missing data i cut it, 
                              band = MODIS_bands[i],   
                              lat = MODIS_lat, 
                              lon = MODIS_lon, 
                              title_subplot = title_subplot, 
                              map_limit = map_boundaries, 
                              fig = fig, 
                              ax = ax, 
                              max_colorbar = max_colorbar, 
                              min_colorbar = min_colorbar,
                              type_variable = args.type_variable)
        
        title_subplot= "RTTOV (12:30/2/5/2013)-Channel:"
        position = position_rttov[i] #position+38
        ax = plt.subplot(nrows, ncols, position)
        plot_MODIS_simulation(variable = variable_simulation, #in the below area there are missing data i cut it, 
                              band = rttov_bands[bnd-1], 
                              lat = rttov_lat, 
                              lon = rttov_lon, 
                              title_subplot = title_subplot, 
                              map_limit = map_boundaries, 
                              fig = fig, 
                              ax = ax, 
                              max_colorbar = max_colorbar, 
                              min_colorbar = min_colorbar,
                              type_variable = args.type_variable)


    plt.tight_layout()
    figure_name = '{}/MODIS and RTTOV output {} in all bands.png'.format(args.path_output, args.type_variable) #aca pasarr con todo path
                   
    fig.savefig(figure_name) 
    plt.close()    

if __name__ == '__main__':
    main()