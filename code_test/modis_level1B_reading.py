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

# plottinh
from mpl_toolkits.basemap import Basemap


def histrogram_raw_plot(out_file): 
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

def histrogram_reflectances_plot(out_file): 
    fig,axes = plt.subplots(3,8,figsize = (32,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

#    for x in sds_list.keys():
#     if (x != 'sds_1_Ems'):
#         n_bands=len(band_list['band_'+x].get())
#         for band in range(n_bands):
#                 data =reflectance_list['reflectance_'+x][band]
#                 data = np.ma.masked_array(data, np.isnan(data))

#                 print(x,data.min())

    i=0
    for x in sds_list.keys():
        if (x != 'sds_1_Ems'):
            n_bands=len(band_list['band_'+x].get())
            for band in range(n_bands):
                a=reflectance_list['reflectance_'+x][band]
                n, bins, patches = axes[i].hist(a)
                axes[i].set_title('Histrogram \n in channel %0.1f max: %0.1f mean: %0.1f' %((band_list['band_'+x][band]),(a.max()),(a.mean())))
                i+=1
    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    plt.tight_layout()
    fig.savefig(out_file+"/MODIS_reflectance_histogram_by_channel.png")  
    plt.close()

def raw_plot(out_file):
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

def radiances_plot(data_radiances_38bands,out_file)
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
    fig.savefig(out_file+"Ony_valid_range_MODIS_radiances_by_channel.png")  
    plt.close()

def radiances_plot(data_radiances_38bands,out_file)
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
    fig.savefig(out_file+"Ony_valid_range_MODIS_radiances_by_channel.png")  
    plt.close()


# it is not used
def projection_plot(myd03_Latitude_data,myd03_Longitude_data,subset_data_radiances_38bands):

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

    m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=30.)
    m.bluemarble()

    data=rgb_out[1000:1600,400:1000,1]
    x, y = m(subset_longitude, subset_latitud)
    m.pcolormesh(x,y,data)
    plt.show()



'''
Function to calculate the reflectances values after mask the values and plot the histogram of it
'''
def mask_refletances_values(sds_list,band_list,offset_list,scale_list,out_file):
    #reflectance_list= { 'reflectance_sds_250_RefSB': [], 'reflectance_sds_500_RefSB': [],'reflectance_sds_1_RefSB':[], 
    #            'reflectance_sds_1_Ems':[]}
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
                axes[i].set_title('Histrogram reflectances\n in channel %0.1f min: %0.1f max: %0.1f' %((band_list['band_'+x][band]),(data.min()),(data.max())))
                
                print(x,band,data.min(),data.max(),data.mean())

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

def save_data(name_file,array,out_file):
    fileName=out_file + '/sample_'+name_file+'.MYD021KM.A2013122.1140.L1B.hdf' #https://clouds.eos.ubc.ca/~phil/courses/atsc301/coursebuild/html/modis_level1b_read.html
    filehdf = SD(fileName, SDC.WRITE | SDC.CREATE)

    # Create a dataset
    sds = filehdf.create(name_file, SDC.FLOAT64, array.shape)
    print(np.shape(sds))
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
    print(np.shape(sds))

    # Close the dataset
    sds.endaccess()

    # Flush and close the HDF file
    filehdf.end()

def verify_copy(name_file,array):
    # Radiances dataset -all
    modis_file = out_file + '/sample_'+name_file+".MYD021KM.A2013122.1140.L1B.hdf"
    filehdf_sample_radiances = SD(modis_file, SDC.READ)
    datasets_dict = filehdf_sample_radiances.datasets()

    for idx,sds in enumerate(datasets_dict.keys()):
        print(idx,sds)
        
    file2_data_subset = filehdf_sample_radiances.select(name_file) # select sds

    print(name_file, 'copied correctly',(array==file2_data_subset).all())   

    print(file2_data_subset.get().max(),file2_data_subset.get().min())
    print(array.max(),array.min())

    filehdf_sample_radiances.end()



def main():
    parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg('--path-in', type=str, default='/home/jvillarreal/Documents/phd/dataset', help='path of the dataset is')
	arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/dataset', help='path of the output data is' )
	args = parser.parse_args()
	
    fname_in = args.path_in 
    out_file = args.path_out 

    os.chdir(fname_in)
    geo_file = 'MYD03.A2013122.1140.061.2018046005026.hdf'

   # Geo File #
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

   # Radiances File #
    # Read dataset.
    modis_file = 'MYD021KM.A2013122.1140.061.2018046032403.hdf'
    file = SD(modis_file, SDC.READ)

    print(file.info())

    datasets_dic = file.datasets()

    for idx, sds in enumerate(datasets_dic.keys()):
        print(idx, sds)

    # Read bands.
    band_sds_250_RefSB = file.select('Band_250M')
    band_sds_500_RefSB = file.select('Band_500M')
    band_sds_1_RefSB = file.select('Band_1KM_RefSB')
    band_sds_1_Ems = file.select('Band_1KM_Emissive')

    band_list ={'band_sds_250_RefSB':band_sds_250_RefSB , 'band_sds_500_RefSB': band_sds_500_RefSB,'band_sds_1_RefSB':band_sds_1_RefSB, 
                'band_sds_1_Ems':band_sds_1_Ems}

    for x in band_list.keys():
        a=band_list[x].attributes()['long_name']
        print(f'modis {a}\n {band_list[x].get()}')  #get=values

    # Read Radiances
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

    #### histogram
    histrogram_raw_plot(out_file)
    raw_plot(out_file)

    radiance_list= { 'radiance_sds_250_RefSB': [], 'radiance_sds_500_RefSB': [],'radiance_sds_1_RefSB':[], 
            'radiance_sds_1_Ems':[]}

    for x in sds_list.keys(): #for key, value in sds_list.items():
        radiance_list['radiance_'+x] = np.zeros(np.shape(sds_list[x]))
        print('radiance_'+x,np.shape(radiance_list['radiance_'+x] ))

    #radiance_list=radiances_values(sds_list,band_list,offset_list,scale_list,radiance_list):

    ### eliminate values out the range
    radiance_list=mask_radiances_values(sds_list,band_list,offset_list,scale_list,radiance_list,out_file):

    data_radiances_38bands = np.concatenate((radiance_list['radiance_sds_250_RefSB'],radiance_list['radiance_sds_500_RefSB'],radiance_list['radiance_sds_1_RefSB'],radiance_list['radiance_sds_1_Ems']))
    data_radiances_38bands= np.ma.masked_array(data_radiances_38bands, np.isnan(data_radiances_38bands))

    bands_radiances_38bands = np.concatenate((band_list['band_sds_250_RefSB'],band_list['band_sds_500_RefSB'],band_list['band_sds_1_RefSB'],band_list['band_sds_1_Ems']))
    print('Ready_data_radiances_38bands',np.shape(data_radiances_38bands))

    radiances_plot(data_radiances_38bands)

    #subset
    subset_data_radiances_38bands=data_radiances_38bands[:,1000:1600,400:1000]
    print(subset_data_radiances_38bands.shape)

    #projection_plot(myd03_Latitude_data,myd03_Longitude_data,subset_data_radiances_38bands):


    reflectance_list=refletances(sds_list,band_list,offset_list,scale_list,out_file)
    rgb_reflectances(sds_list,offset_list,scale_list,out_file)

    save_data("subset_radiances_38bands",subset_data_radiances_38bands,out_file)
    save_data("radiances_38bands",data_radiances_38bands,out_file)

    verify_copy("subset_radiances_38bands",subset_data_radiances_38bands)
    verify_copy("radiances_38bands",data_radiances_38bands)


if __name__ == '__main__':
    main()