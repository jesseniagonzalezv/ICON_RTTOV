'''
Input: MYD021KM modis radiances level 1B
Output: Modis radiances in 36 channels in values  W/m2 -µm-sr
# Reference https://hdfeos.org/zoo/LAADS/MYD021KM.A2002226.0000.006.2012033082925.hdf.py
'''

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import pprint
#%matplotlib inline
import argparse
import sys

# plottinh
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs


def histrogram_raw_plot(out_file,sds_list,band_list): 
    fig,axes = plt.subplots(5,8,figsize = (32,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    i=0
    for x in sds_list.keys():
        n_bands=len(band_list['band_'+x].get())
        for band in range(n_bands):
            a=sds_list[x][band]
            n, bins, patches = axes[i].hist(a)
            axes[i].set_title('Histrogram \n in channel %0.1f max: %0.1f mean: %0.1f' %((band_list['band_'+x][band]),(a.max()),(a.mean())))
            i+=1
    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    plt.tight_layout()
    fig.savefig(out_file+"/MODIS_raw_histogram_by_channel.png")  
    plt.close()

def raw_plot(out_file,sds_list,band_list):
    fig,axes = plt.subplots(5,8,figsize = (32,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    i=0
    for x in sds_list.keys():
        n_bands=len(band_list['band_'+x].get())
        for band in range(n_bands):
    # radiance_list['radiance_'+ x][band] = (sds_list[x][band] - offset_list['radiance_offsets_'+ x][band]) * scale_list['radiance_scales_'+ x][band]

                #axes[i].imshow(X_train[:,:,i],cmap='gray', vmin=0, vmax=1)
        #axes[i].imshow(X_data[i,:,:],cmap='gray')
            pcm = axes[i].imshow(np.fliplr(sds_list[x][band]),  origin='lower') #interpolation='nearest'
            #pcm = axes[i].imshow(np.fliplr(radiance_list['radiance_'+ x][band]),  origin='lower') #interpolation='nearest'

            cbar=fig.colorbar(pcm, ax=axes[i])
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label('$(W\,m^{-2}\,\mu m^{-1}\,sr^{-1})$ \n \n', rotation=270)

            #out2=cax.ax.set_ylabel('$(W\,m^{-2}\,\mu m^{-1}\,sr^{-1})$ \n \n')
            #out2.set_rotation(270)
            axes[i].set_title('Raw \n in channel %0.1f'% (band_list['band_'+x][band]))
            i+=1
    #fig,axes = plt.subplots(5,8,figsize = (40,20))
    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    plt.tight_layout()
    fig.savefig(out_file+"/MODIS_raw_by_channel.png")  
    plt.close()

'''
mask values higher than value_max before get radiances and plot histogram of radiances
'''
def mask_radiances_values(sds_list,band_list,offset_list,scale_list,radiance_list,out_file):
        ##all have the same FillValue and valid range
    attrs = sds_list['sds_250_RefSB'].attributes(full=1)
    fva=attrs["_FillValue"]
    _FillValue = fva[0]
    vra=attrs["valid_range"]
    valid_min = vra[0][0]        
    valid_max = vra[0][1]
    ua=attrs["radiance_units"]
    units = ua[0]

    ### mask values higher than value_max
    fig,axes = plt.subplots(5,8,figsize = (32,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    i=0

    for x in sds_list.keys():
        n_bands=len(band_list['band_'+x].get())
        for band in range(n_bands):
            data=sds_list[x][band].astype(np.double)
            invalid = np.logical_or(data > valid_max,
                            data < valid_min)
            invalid = np.logical_or(invalid, data == _FillValue)
            data[invalid] = np.nan
            data= (data- offset_list['radiance_offsets_'+ x][band]) * scale_list['radiance_scales_'+ x][band]
            data = np.ma.masked_array(data, np.isnan(data))

            n, bins, patches = axes[i].hist(data)
            axes[i].set_title('Histrogram radiances \n in channel %0.1f min: %0.1f max: %0.1f' %((band_list['band_'+x][band]),(data.min()),(data.max())))
            i+=1
            
            radiance_list['radiance_'+ x][band]=data
            #print(x,band,data.min(),data.max(),data.mean())
        
    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    plt.tight_layout()
    fig.savefig(out_file+"/Only_valid_range_MODIS_radiances_histogram_by_channel.png")  
    plt.close()

    return radiance_list

'''
get radiances 
'''
def radiances_values(sds_list,band_list,offset_list,scale_list,radiance_list):
    for x in sds_list.keys():
        n_bands=len(band_list['band_'+x].get())
        for band in range(n_bands):
            radiance_list['radiance_'+ x][band] = (sds_list[x][band] - offset_list['radiance_offsets_'+ x][band]) * scale_list['radiance_scales_'+ x][band]
    return radiance_list

def radiances_histogram(sds_list,band_list,radiance_list):
    #### histrogram radiances
    fig,axes = plt.subplots(5,8,figsize = (32,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    i=0
    for x in sds_list.keys():
        n_bands=len(band_list['band_'+x].get())
        for band in range(n_bands):
            r=radiance_list['radiance_'+ x][band] 
            r = np.ma.masked_array(r, np.isnan(r))
            n, bins, patches = axes[i].hist(r)
            axes[i].set_title('Histrogram radiances \n in channel %0.1f min: %0.1f max: %0.1f' %((band_list['band_'+x][band]),(r.min()),(r.max())))
            i+=1
    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    plt.tight_layout()
    fig.savefig("MODIS_radiances_histogram_by_channel.png")  
    plt.close()

    
def radiances_plot(data_radiances_38bands,bands_radiances_38bands,name_out,out_file):
    fig,axes = plt.subplots(5,8,figsize = (30,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    n_bands_sample=38 #7
    for i in range(n_bands_sample):
        #axes[i].imshow(X_train[:,:,i],cmap='gray', vmin=0, vmax=1)
        #axes[i].imshow(X_data[i,:,:],cmap='gray')
        pcm = axes[i].imshow(np.fliplr(data_radiances_38bands[i]),  origin='lower') #interpolation='nearest'

        cbar=fig.colorbar(pcm, ax=axes[i])
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label('$(W\,m^{-2}\,\mu m^{-1}\,sr^{-1})$ \n \n', rotation=270)

        #out2=cax.ax.set_ylabel('$(W\,m^{-2}\,\mu m^{-1}\,sr^{-1})$ \n \n')
        #out2.set_rotation(270)
        axes[i].set_title('Radiances \n in channel %0.1f'% (bands_radiances_38bands[i]))
        
    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    plt.tight_layout()
    fig.savefig(out_file+name_out)  
    plt.close()


# it is not used
def projection_plot(myd03_Latitude_data,myd03_Longitude_data,subset_data_radiances_38bands,data_radiances_38bands):

    subset_latitud=myd03_Latitude_data[1000:1600,400:1000]
    subset_longitude=myd03_Longitude_data[1000:1600,400:1000]
    data = subset_data_radiances_38bands[11,:,:]

    #m = Basemap(projection='cyl', resolution='l', llcrnrlat=-90, urcrnrlat = 90, llcrnrlon=-180, urcrnrlon = 180)
    #m = Basemap(width=6000000,height=4500000,resolution='c',projection='aea',lat_1=30.,lat_2=90,lon_0=-45,lat_0=90)
    m = Basemap(projection='cyl', resolution='l', llcrnrlat=30, urcrnrlat = 90, llcrnrlon=-0, urcrnrlon = 90)
    m.drawcoastlines(linewidth=0.5)


    m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180., 181., 45.), labels=[0, 0, 0, 1])
    x, y = m(subset_longitude, subset_latitud)
    m.pcolormesh(x, y, data)
    #https://www.humbleisd.net/cms/lib2/TX01001414/Centricity/Domain/3635/Map%20Projections.pdf
    #https://matplotlib.org/basemap/users/mapsetup.html
    #https://rabernat.github.io/research_computing/intro-to-basemap.html

    #########################################################################################
    m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=30.)
    m.bluemarble()

    data=rgb_out[1000:1600,400:1000,1]
    x, y = m(subset_longitude, subset_latitud)
    m.pcolormesh(x,y,data)
    plt.show()
    
    ########################################################################
    data_test=data_radiances_38bands[1]
    # Find middle location.
    lat_m = np.nanmean(myd03_Latitude_data)
    lon_m = np.nanmean(myd03_Longitude_data)

    # Let's use ortho projection.
    orth = ccrs.Orthographic(central_longitude=lon_m,
                             central_latitude=lat_m,
                             globe=None)
    ax = plt.axes(projection=orth)

    # Set global view. You can comment it out to get zoom-in view.
    #ax.set_global()

    # If you want to get global view, you need to subset.
    p = plt.pcolormesh(myd03_Longitude_data[::5][::5],
                       myd03_Latitude_data[::5][::5],
                       data_test[::5][::5],
                       transform=ccrs.PlateCarree())
    ax.gridlines()
    #ax.coastlines(resolution="110m")
    cb = plt.colorbar(p)
    cb.set_label(units, fontsize=8)



'''
Function to calculate the reflectances values after mask the values and plot the histogram of it
'''
def mask_refletances_values(sds_list,band_list,offset_list,scale_list,out_file):
    #reflectance_list= { 'reflectance_sds_250_RefSB': [], 'reflectance_sds_500_RefSB': [],'reflectance_sds_1_RefSB':[], 
    #            'reflectance_sds_1_Ems':[]}
    attrs = sds_list['sds_250_RefSB'].attributes(full=1)
    fva=attrs["_FillValue"]
    _FillValue = fva[0]
    vra=attrs["valid_range"]
    valid_min = vra[0][0]        
    valid_max = vra[0][1]
    ua=attrs["radiance_units"]
    units = ua[0]

    reflectance_list= {}
    for x in sds_list.keys(): #for key, value in sds_list.items():
        if (x != 'sds_1_Ems'):
            name='reflectance_'+x
            reflectance_list[name] = np.zeros(np.shape(sds_list[x]))
            print(name,np.shape(reflectance_list[name] ))
    ### mask values higher than value_max
    fig,axes = plt.subplots(3,8,figsize = (32,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    i=0
    for x in sds_list.keys():
        if (x != 'sds_1_Ems'):
            n_bands=len(band_list['band_'+x].get())
            for band in range(n_bands):
                data=sds_list[x][band].astype(np.double)
                invalid = np.logical_or(data > valid_max,
                                data < valid_min)
                invalid = np.logical_or(invalid, data == _FillValue)
                data[invalid] = np.nan
                data = (data - offset_list['reflectance_offsets_'+ x][band]) * scale_list['reflectance_scales_'+ x][band]
                data = np.ma.masked_array(data, np.isnan(data))
                reflectance_list['reflectance_'+x][band]=data
                
                n, bins, patches = axes[i].hist(data)
                axes[i].set_title('Histogram reflectances\n in channel %0.1f min: %0.1f max: %0.1f' %((band_list['band_'+x][band]),(data.min()),(data.max())))
                
                #print(x,band,data.min(),data.max(),data.mean())

                i+=1
                
    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    plt.tight_layout()
    fig.savefig(out_file+"/Only_valid_range_MODIS_reflectance_histogram_by_channel.png")  
    plt.close()

    return reflectance_list

'''
Read reflectances with mask and 
'''
def rgb_reflectances(sds_list,offset_list,scale_list,out_file):

   # reflectance_RGB = np.zeros((2030, 1354,3))
   # name_data= 'sds_250_RefSB'
   # reflectance_RGB[:,:,0]= ((sds_list[name_data][0] - offset_list['reflectance_offsets_'+ name_data][0]) * scale_list['reflectance_scales_'+ name_data][0])
   # name_data= 'sds_500_RefSB'
   # reflectance_RGB[:,:,1]= ((sds_list[name_data][1] - offset_list['reflectance_offsets_'+ name_data][1]) * scale_list['reflectance_scales_'+ name_data][1]) 
   # reflectance_RGB[:,:,2]= ((sds_list[name_data][0] - offset_list['reflectance_offsets_'+ name_data][0]) * scale_list['reflectance_scales_'+ name_data][0]) 
    reflectance_RGB = np.zeros((2030, 1354,3))
    name_data= 'sds_250_RefSB'
    reflectance_RGB[:,:,0]= ((sds_list[name_data][0] - offset_list['reflectance_offsets_'+ name_data][0]) * scale_list['reflectance_scales_'+ name_data][0])
    #name_data= 'sds_1_RefSB'
    reflectance_RGB[:,:,1]= ((sds_list[name_data][1] - offset_list['reflectance_offsets_'+ name_data][1]) * scale_list['reflectance_scales_'+ name_data][1]) 
    reflectance_RGB[:,:,2]= 0.5*(reflectance_RGB[:,:,0]+reflectance_RGB[:,:,1])
    print("RGB min, max, mean",reflectance_RGB.min(), reflectance_RGB.max(),reflectance_RGB.mean())


    norme = 0.6 #0.2 # 0,4factor to increase the brightness ]0,1] https://moonbooks.org/Codes/Plot-MODIS-granule-RGB-image-using-python/
    
    rgb_out= np.zeros((2030, 1354,3))

    rgb_out= reflectance_RGB / norme

    rgb_out[ rgb_out > 1 ] = 1.0
    rgb_out[ rgb_out < 0 ] = 0.0


    fig,ax = plt.subplots(1,1,figsize = (10,10))
    CS=ax.imshow(np.fliplr(rgb_out),  origin='lower') #interpolation='nearest' #reflectance_RGB/reflectance_RGB.max()*6)


    l = [int(i) for i in np.linspace(0,1354,6)]
    plt.xticks(l, [i for i in reversed(l)], rotation=0, fontsize=7 )

    l = [int(i) for i in np.linspace(0,2030,9)]
    plt.yticks(l, l, rotation=0, fontsize=7 )

    #cax=fig.colorbar(CS)
    ax.set_title('RGB reflectance')

    #out=cax.ax.set_ylabel('RGB reflectance') #out=cax.ax.set_ylabel('RGB radiance $(W\,m^{-2}\,\mu m^{-1}\,sr^{-1})$')
    #out.set_verticalalignment('bottom')
    #out.set_rotation(270)

    plt.savefig(out_file+"/modis_granule_rgb.png", bbox_inches='tight', dpi=100)
    plt.close()


def save_data(name_file,array,bands,out_file):
    fileName=out_file+name_file+'_MYD021KM.A2013122.1140.L1B.hdf' #https://clouds.eos.ubc.ca/~phil/courses/atsc301/coursebuild/html/modis_level1b_read.html
    
    filehdf = SD(fileName, SDC.WRITE | SDC.CREATE)

    # Create a dataset
    sds = filehdf.create(name_file, SDC.FLOAT64, array.shape)
    print(np.shape(sds))
    sds2 = filehdf.create("bands_radiances_38bands", SDC.FLOAT64, bands.shape)

    # Fill the dataset with a fill value
    sds.setfillvalue(0)

    # Set dimension names
    #dim1 = sds.dim(0)
    #dim1.setname("row")
    #dim2 = sds.dim(1)
    #dim2.setname("col")

    # Assign an attribute to the dataset
    sds.units = "W/m^2/micron/sr"



    # Write data
    sds[:] = array
    sds2[:] = bands

   # print(np.shape(sds))

    # Close the dataset
    sds.endaccess()
    sds2.endaccess()


    # Flush and close the HDF file
    filehdf.end()




def verify_copy(name_file,array,out_file):
    # Radiances dataset -all
    modis_file = out_file + name_file+"_MYD021KM.A2013122.1140.L1B.hdf"
    filehdf_sample_radiances = SD(modis_file, SDC.READ)
    datasets_dict = filehdf_sample_radiances.datasets()

    for idx,sds in enumerate(datasets_dict.keys()):
        print(idx,sds)
        
    file_data_sample = filehdf_sample_radiances.select(name_file) # select sds
    file_values=file_data_sample.get()
    file_values= np.ma.masked_array(file_values, np.isnan(file_values))

    print(name_file, 'copied correctly',(array==file_data_sample).all())   

    print(name_file,"max and min:",file_values.max(),file_values.min())
    print("Array values max and min:",array.max(),array.min())

    filehdf_sample_radiances.end()

################################################################################################################################

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-in', type=str, default='/home/jvillarreal/Documents/phd/dataset', help='path of the dataset is')
    arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/ouput', help='path of the output data is' )
    args = parser.parse_args()

    fname_in = args.path_in 
    out_file = args.path_out 
    
    
    sys.stdout = open(out_file+'/log_MODIS_LEVEL1B.txt','wt')

    #os.chdir(fname_in)
    geo_file = fname_in+'/MYD03.A2013122.1140.061.2018046005026.hdf'

    ######################################################
   # Geo File #
    print("geo_file")
    file_geo = SD(geo_file, SDC.READ)
    print("Reading MODIS LEVEL 1B + Geo File",geo_file)
    print(file_geo.info())

    datasets_dic = file_geo.datasets()

    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)

    lat = file_geo.select('Latitude')
    myd03_Latitude_data = lat[:,:]
    lon = file_geo.select('Longitude')
    myd03_Longitude_data = lon[:,:]

    #fig,ax = plt.subplots(1,1,figsize = (6,7))
        ##ax.plot(myd03_Longitude_data[900:940,900:940],myd03_Latitude_data[900:940,900:940],'b+');
        #ax.plot(myd03_Longitude_data,myd03_Latitude_data,'b+');
        #ax.set(xlabel='longitude (deg W)',ylabel='latitude (deg N)');
        
    print ('lat min, max', myd03_Latitude_data.min(), myd03_Latitude_data.max())
    print ('lon min, max', myd03_Longitude_data.min(), myd03_Longitude_data.max())

    ######################################################
    print("MODIS_level1B")
    
    modis_file = fname_in+'/MYD021KM.A2013122.1140.061.2018046032403.hdf'
    file = SD(modis_file, SDC.READ)

    print(file.info())

    datasets_dic = file.datasets()

    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)

    ###############################
    # Read bands.
    ###############################
    band_sds_250_RefSB = file.select('Band_250M')
    band_sds_500_RefSB = file.select('Band_500M')
    band_sds_1_RefSB = file.select('Band_1KM_RefSB')
    band_sds_1_Ems = file.select('Band_1KM_Emissive')

    band_list ={'band_sds_250_RefSB':band_sds_250_RefSB , 'band_sds_500_RefSB': band_sds_500_RefSB,'band_sds_1_RefSB':band_sds_1_RefSB, 
                'band_sds_1_Ems':band_sds_1_Ems}

    for x in band_list.keys():
        a=band_list[x].attributes()['long_name']
        print(f'modis {a}\n {band_list[x].get()}')  #get=values

    ###############################
    # Read Radiances
    ###############################
    print("Reading the radiances")
    sds_250_RefSB = file.select('EV_250_Aggr1km_RefSB')
    sds_data_250_RefSB=sds_250_RefSB.get()  #values
    sds_500_RefSB = file.select('EV_500_Aggr1km_RefSB')
    sds_1_RefSB = file.select('EV_1KM_RefSB')
    sds_1_Ems = file.select('EV_1KM_Emissive')
    #help(sds_250_RefSB.info)

    sds_list= { 'sds_250_RefSB': sds_250_RefSB, 'sds_500_RefSB': sds_500_RefSB,'sds_1_RefSB':sds_1_RefSB, 
                'sds_1_Ems':sds_1_Ems}
    #np.shape(sds_250_RefSB),sds_250_RefSB
    #sds_250_RefSB.attributes()
    for x in sds_list.keys():
        print(x)

    scale_list= { 'reflectance_scales_sds_250_RefSB':[] , 'reflectance_scales_sds_500_RefSB':[] ,
                'reflectance_scales_sds_1_RefSB':[],
                'radiance_scales_sds_250_RefSB':[] , 'radiance_scales_sds_500_RefSB': [],
                'radiance_scales_sds_1_RefSB':[], 'radiance_scales_sds_1_Ems':[]
                }
    offset_list= {'reflectance_offsets_sds_250_RefSB':[] , 'reflectance_offsets_sds_500_RefSB':[] ,
                'reflectance_offsets_sds_1_RefSB':[],
                'radiance_offsets_sds_250_RefSB':[] , 'radiance_offsets_sds_500_RefSB': [],
                'radiance_offsets_sds_1_RefSB':[], 'radiance_offsets_sds_1_Ems':[]
                }


    for x in sds_list.keys():   
        for out_type in ['radiance', 'reflectance']:
            #print(np.shape(sds_list[x]))
            
            if( not(x == 'sds_1_Ems' and out_type =='reflectance')):
                #print(x)
                scale_name=out_type+'_scales'+ '_'+x
                offset_name=out_type+'_offsets'+ '_'+x   
                scale_list[scale_name]=sds_list[x].attributes()[out_type+'_scales'] 
                offset_list[offset_name]=sds_list[x].attributes()[out_type+'_offsets']     

    #pprint.pprint ("Values scales  :" + str(dict(scale_list)))

    ######################################################
    #### histogram
    print("MODIS_raw_histogram_by_channel plotting")
    histrogram_raw_plot(out_file,sds_list,band_list)
          
    print("MODIS_raw_by_channel plotting")
    raw_plot(out_file,sds_list,band_list)

    radiance_list= { 'radiance_sds_250_RefSB': [], 'radiance_sds_500_RefSB': [],'radiance_sds_1_RefSB':[], 
            'radiance_sds_1_Ems':[]}

    for x in sds_list.keys(): #for key, value in sds_list.items():
        radiance_list['radiance_'+x] = np.zeros(np.shape(sds_list[x]))
        print('radiance_'+x,np.shape(radiance_list['radiance_'+x] ))
        
    ######################################################
    ###radiance_list=radiances_values(sds_list,band_list,offset_list,scale_list,radiance_list)
    ###radiances_histogram(sds_list,band_list,radiance_list)
          
    ### eliminate values out the range
    print("Masking the radiance nan values and plotting histograms")
    radiance_list=mask_radiances_values(sds_list,band_list,offset_list,scale_list,radiance_list,out_file)

    ######################################################
    print("Concatenation of the data: radiances 38")
    data_radiances_38bands = np.concatenate((radiance_list['radiance_sds_250_RefSB'],radiance_list['radiance_sds_500_RefSB'],radiance_list['radiance_sds_1_RefSB'],radiance_list['radiance_sds_1_Ems']))
    data_radiances_38bands= np.ma.masked_array(data_radiances_38bands, np.isnan(data_radiances_38bands))

    bands_radiances_38bands = np.concatenate((band_list['band_sds_250_RefSB'],band_list['band_sds_500_RefSB'],band_list['band_sds_1_RefSB'],band_list['band_sds_1_Ems']))
    print('Ready_data_radiances_38bands',np.shape(data_radiances_38bands))

    print("Plotting of the radiances: radiances 38")
    name_out ="/Only_valid_range_MODIS_radiances_by_channel.png"
    radiances_plot(data_radiances_38bands,bands_radiances_38bands,name_out,out_file)

    ######################################################
    print("Concatenation of the subset data: radiances 38")
    #subset
    subset_data_radiances_38bands=data_radiances_38bands[:,1000:1600,400:1000]
    print(subset_data_radiances_38bands.shape)
    
    print("Plotting of the subset radiances: radiances 38")
    name_out ="/Only_valid_range_MODIS_subset_radiances_by_channel.png"
    radiances_plot(subset_data_radiances_38bands,bands_radiances_38bands,name_out,out_file)

    #projection_plot(myd03_Latitude_data,myd03_Longitude_data,subset_data_radiances_38bands):

    ######################################################
    #Saving data
    print("Saving the data")
    save_data("subset_radiances_38bands",subset_data_radiances_38bands,bands_radiances_38bands,out_file)
    save_data("radiances_38bands",data_radiances_38bands,bands_radiances_38bands,out_file)

    verify_copy("subset_radiances_38bands",subset_data_radiances_38bands,out_file)
    verify_copy("radiances_38bands",data_radiances_38bands,out_file)
    print("Ready saved data")

    ######################################################
    print("Masking the nan reflectances values and plotting histograms")
    reflectance_list=mask_refletances_values(sds_list,band_list,offset_list,scale_list,out_file)
        
    print("RGB reflectances plotting")
    rgb_reflectances(sds_list,offset_list,scale_list,out_file)

    sys.stdout.close()

if __name__ == '__main__':
    main()