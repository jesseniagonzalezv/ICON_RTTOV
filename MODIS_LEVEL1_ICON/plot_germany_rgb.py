

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import argparse
from modis_functions import scale_image, plot_rgb_image

import xarray as xr



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-in', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/rttov-131-data-icon-1to19-26-T12.nc', help='path of the dataset is')
    arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/output', help='path of the output data is' )
    arg('--type', type = str, default = 'modis', help = 'simulation modis' )

    args = parser.parse_args()

    fname_in = args.path_in 
    out_file = args.path_out 
    
    
#     name_file = fname_in 
#     filehdf = SD(fname_in, SDC.READ)
#     datasets_dict = filehdf.datasets()
    


    
    if  args.type == "simulation":
        data_ds = xr.open_dataset(args.path_in).compute()
        data =data_ds['bt_refl_total']
        name_plot = "simulation"
#         data= filehdf.select('bt_refl_total').get() #radiances

            
#         data_reduced = np.zeros((data.shape[0],628,data.shape[2])) 


        data_reduced = data[:,9:,:] #in the below area there are missing data i cut it
        print('shape data reduced',np.shape(data_reduced))
        

    elif args.type == "modis":
        data_ds = xr.open_dataset(args.path_in).compute()
        data =data_ds['MODIS_Germany_refl_emis'] #['MODIS_Germany_radiances'] #["subset_radiances_38bands"] #['MODIS_Germany_refl_emis']
#         data= filehdf.select('subset_radiances_38bands').get() #radiances
#         data= filehdf.select('MODIS_Germany_radiances').get() #radiances

        data_reduced = data
        name_plot = "modis"
        
    lat = data_ds['lat'].values
    lon = data_ds['lon'].values

        
    data_reduced = np.ma.masked_array(data_reduced,  np.isnan(data_reduced))

    along_track = np.shape(data_reduced)[1] #2030
    cross_trak = np.shape(data_reduced)[2] #1354

    z = np.zeros((along_track, cross_trak,3))

    z[:,:,0]  = data_reduced[0,:,:]  #Channels 1,3,4 (0,1,2) then change for RGB: 1,4,3  (0,2,1)
    z[:,:,1]  = data_reduced[3,:,:]
    z[:,:,2]  = data_reduced[2,:,:]
           


    plot_rgb_image(along_track, cross_trak, z, out_file, name_plot,lon, lat)

        #plot_rgb_image(along_track, cross_trak, z, out_file, name_plot = "simulation")




#     plot_image_rgb(image_3bands = z_color_enh,  out_file = out_file)

########3


#     R = np.clip(R, 0, 1)
#     G = np.clip(G, 0, 1)
#     B = np.clip(B, 0, 1)
    
#     gamma = 2.2
#     R = np.power(R, 1/gamma)
#     G = np.power(G, 1/gamma)
#     B = np.power(B, 1/gamma)

#     # Calculate the "True" Green
#     G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
#     #G_true = 0.45 * R + 0.1 * G + 0.45 * B
#     G_true = np.clip(G_true, 0, 1)

#     # The final RGB array :)
#     RGB = np.dstack([R, G_true, B])
#     #RGB = np.dstack([R, G, B])
#     print("RGB:", np.shape(RGB))

#     plot_image_rgb(image_3bands = RGB,  out_file = out_file)

    #####################################################


        
if __name__ == '__main__':
    main()