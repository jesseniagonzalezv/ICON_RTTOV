
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

def visulize_sat(variable, band, lat, lon, title_subplot, map_limit, fig,ax):

    #axes = axes.ravel()
        cbar_label = 'Radiance $(W\,m^{-2}\,sr^{-1}\,\mu m^{-1})$'

        print('ploting band {}'.format(band))

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
        
        print("data visuali ------------", np.min(data),np.max(data))
        pprint.pprint(data)

        #_FillValue= 999999# 0 #65535  #decide if it is needed to pass as parameter
        
        if(title_subplot == "MODIS (11:40/2/5/2013)-Channel:"):
            data = np.ma.masked_equal(data, 0) # np.ma.masked_array(data,np.isnan(data)) # data== _FillValue)
                
        print("data visuali 2------------", np.min(data),np.max(data))

        pprint.pprint(data)

        extend = 'both' #min,max,both,neither
        cmap=plt.get_cmap('jet') #du bleu au rouge
                        
        #cs = map.contourf(x,y, y_filtered,levels,extend=extend,cmap=cmap) 
        cs = map.pcolormesh(x,y,data, cmap=cmap,shading='auto')
    
        cbar = fig.colorbar(cs, ax=ax,label=cbar_label,shrink=0.75) #location="right",pad="5%",ticks=[270,275,280,285,290,295,300],
        cbar.ax.tick_params(size=0,labelsize=10)

        ax.set_title("{}{}".format(title_subplot, band),fontsize=14)
        ax.set_xlabel('Longitude', labelpad=20,fontsize=14)
        ax.set_ylabel('Latitude', labelpad=33,fontsize=14)

        


    



       # visulize_sat(grid_valid_variable, bands_variable, gr_lat, gr_lon, cbar_label, titel, map_boundaries )

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--rttov-path', type = str, default = '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc', help = 'Path of the dataset')
    #arg('--rttov-path', type = str, default = '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-msg-1-to-36.nc', help = 'Path of the dataset')

    arg('--MODIS-path', type = str, default = '/home/jvillarreal/Documents/phd/dataset/MODIS_Germany_radiances_MYD021KM.A2013122.1140.061.2018046032403.nc', help = 'Path of the output' )

    args = parser.parse_args()

    rttov_ds = xr.open_dataset(args.rttov_path).compute()
    MODIS_ds = xr.open_dataset(args.MODIS_path).compute()

    MODIS_variable =MODIS_ds['MODIS_Germany_radiances']
    MODIS_bands =MODIS_ds['bands'].values

    rttov_variable =rttov_ds['Y']
    rttov_bands =rttov_ds['chan'].values

#    rttov_variable =rttov_ds['Y_clear'][:]
    MODIS_lat = MODIS_ds['lat'].values
    MODIS_lon = MODIS_ds['lon'].values

    rttov_lat = rttov_ds['lat'].values
    rttov_lon = rttov_ds['lon'].values
    n_bands = len(MODIS_bands)*2 #np.size(bands)
    ncols = 2 #
    print(n_bands)

    nrows = n_bands // ncols + (n_bands % ncols > 0) #tbn pujede ser entrada
    fig = plt.figure(figsize=(5*ncols, 5*nrows))   
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    position =list(range(1,n_bands+1))
    position_rttov = position[::2]
    positon_MODIS = position[1::2]
    

    figure_name = 'MODIS and RTTOV output radiances' #aca pasarr con todo path
    map_boundaries=np.zeros(4)
    map_boundaries[0] = 47.5 #llcrnrlat
    map_boundaries[1] = 54.5 # 496 #urcrnrlat
    map_boundaries[2] = 4.5 #llcrnrlon
    map_boundaries[3] = 14.5 #496 #urcrnrlon


    for i in range(len(MODIS_bands)): 
    #for i in range(2):

        variable = MODIS_variable[i]
        position =positon_MODIS[i]
        title_subplot= "MODIS (11:40/2/5/2013)-Channel:"
        ax = plt.subplot(nrows, ncols, position)
        visulize_sat(variable, MODIS_bands[i], MODIS_lat, MODIS_lon, title_subplot, map_boundaries, fig, ax)

        bnd=MODIS_bands[i].astype(int) # hacer que cuando sea igual al indice seleccionar ese cuando la dim chan ==i
        variable = rttov_variable[bnd-1]
        position =position_rttov[i]
        title_subplot= "RTTOV (12:30/2/5/2013)-Channel:"
        ax = plt.subplot(nrows, ncols, position)
        visulize_sat(variable, rttov_bands[bnd-1], rttov_lat, rttov_lon, title_subplot, map_boundaries, fig, ax)



    plt.tight_layout()
    fig.savefig("{} in all bands.png".format(figure_name)) 
    plt.close()    

if __name__ == '__main__':
    main()