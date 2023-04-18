import argparse
import glob
import os
import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


from modis_functions import dataset_dic,grid_coordinate, grid, gridding_variable, read_bands, read_coordinate, read_level1_array, visulize_sat, save_data, dataframe_csv,save_ncfile, rgb_image, plot_rgb_image, read_units

#%matplotlib inline

def get_spectral(data_file, path_dataset, map_boundaries, gridSize_lat, gridSize_lon, allLat, allLon, file_data, list_variable_name, band_list, MODIS_dataset_output, MODIS_dataset_output_germany, MODIS_bands_output, type_data):
    """
    Obtain values of reflectances or radiances of the MODIS
    Arguments:
         map_boundaries -- limit of the lat and lon, min and max values for both
         gridSize_lat -- gridsize of the latitude
         gridSize_lon -- gridsize of the longitude
         allLat -- values of lat of the file MODIS geo
         allLon -- values of lon of the file MODIS geo
         type_data -- dictionary with type_reflectance or type_radiance
         file_data -- modis file already read with netcdf4
         list_variable_name -- dictionary which has the names of the variables to be process
         band_list -- dictionary with the names of the bands
    Argument/ Return:
        MODIS_dataset_output -- MODIS with values of reflectances or radiances (it depends of the type_data provided)
        MODIS_dataset_output_germany -- MODIS only of the area of Germany (reflectances or radiances)
        bands --
    """
    for index_name in type_data: 
        variable_name = list_variable_name[index_name]
        bands_name = band_list[index_name]

        # if task == "get_refl_emiss":
        #     type_variable = type_reflectance[index_name]
        #     titel = 'MODIS_reflectance_emissivity' #aca pasarr con todo path
        #     cbar_label = 'Reflectance - emissivity $none units$'
        # else:
        #     type_variable = type_radiance[index_name]
        #     titel = 'MODIS_radiances' #aca pasarr con todo path
        #     cbar_label = 'Radiance $(W\,m^{-2}\,sr^{-1}\,\mu m^{-1})$'

        variable_raw, valid_variable = read_level1_array(file_data, variable_name, type_data[index_name])
        print((" ================= {} read in {}").format(type_data[index_name], variable_name))

        MODIS_dataset_output= np.dstack((MODIS_dataset_output.transpose(1,2,0), valid_variable.transpose(1,2,0))).transpose(2,0,1) # ChxHxW => HxWxCh => HxWxCh 
        bands_variable = read_bands(file_data,bands_name)
        print((" ------------ Bands: {}").format(bands_variable))

        MODIS_bands_output = np.concatenate((MODIS_bands_output, bands_variable), axis=0) 
        #units_variable = read_units(file_data,variable_name, type_variable)
        #units_names.extend(units_variable)

        print(' ------------  Gridding MODIS to get Germany area  ')

        grid_valid_variable = gridding_variable(valid_variable, map_boundaries, gridSize_lat, gridSize_lon, allLat, allLon)


        MODIS_dataset_output_germany= np.dstack((MODIS_dataset_output_germany.transpose(1,2,0), grid_valid_variable.transpose(1,2,0))).transpose(2,0,1)

#             #bounds it give the range of the colors but i want it get adapted to the data
#            # visulize_sat(grid_valid_variable, bands_variable, gr_lat, gr_lon, cbar_label, titel, map_boundaries )

#         if task == "get_refl_emiss":
#             type_output = "refl_emis"

#         else:
#             type_output = "radiances"

    MODIS_dataset_output = np.delete( MODIS_dataset_output, 0,axis=0)
    MODIS_dataset_output_germany = np.delete( MODIS_dataset_output_germany, 0,axis=0)

    
    print(' ======= Dataframe MODIS raw data {} ======= '.format(type_data[index_name]))
    name_file = "{}/MODIS_{}_{}".format(path_dataset, type_data[index_name], data_file[:-4]) #data_file with -4 to dont include hdf

    dataframe_csv(variable = MODIS_dataset_output, colum = MODIS_bands_output, out_file = name_file)

    print(' ======= Dataframe Germany MODIS {} ======= '.format(type_data[index_name]))
    name_file = "{}/MODIS_Germany_{}_{}".format(path_dataset, type_data[index_name], data_file[:-4]) #dont include hdf
    dataframe_csv(variable = MODIS_dataset_output_germany, colum = MODIS_bands_output, out_file = name_file)

    return MODIS_dataset_output, MODIS_dataset_output_germany, MODIS_bands_output

            

            
            
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-dataset', type = str, default = '/home/jvillarreal/Documents/phd/dataset', help = 'Path of the dataset')
    arg('--path-output', type = str, default = '/home/jvillarreal/Documents/phd/output', help = 'Path of the output' )
    arg('--task', type = str, default = 'spectral', help = 'spectral, rgb' )

    args = parser.parse_args()
    task = args.task 
    path_dataset = args.path_dataset
    path_output = args.path_output
    data_file = 'MYD021KM.A2013122.1140.061.2018046032403.hdf'
    geolocation_file = 'MYD03.A2013122.1140.061.2018046005026.hdf'
   
    #Selecting the area of Germany
    map_boundaries=np.zeros(4)  ##
    map_boundaries[0] = 47.599 #llcrnrlat
    map_boundaries[1] = 54.5 #urcrnrlat
    map_boundaries[2] = 4.5 #llcrnrlon
    map_boundaries[3] = 14.5 #urcrnrlon
    # gridSize_lon=0.017 # 1 grado = 100 km => 0.01 grados = 1 km
    # gridSize_lat=0.011 # 1 grado = 100 km => 0.01 grados = 1 km
    gridSize_lon=0.017 # 1 grado = 100 km => 0.01 grados = 1 km
    gridSize_lat=0.011 # 1 grado = 100 km => 0.01 grados = 1 km
    
    print(" ********** MODIS level1B data **********")
    file_data = SD(os.sep.join([path_dataset,data_file]), SDC.READ)
    dataset_dic(file_data)
      
    #set the modis file L1 & geo
    print(" ********** Geolocation file **********")
    file_geo = SD(os.sep.join([path_dataset,geolocation_file]), SDC.READ)
    dataset_dic(file_geo)
        

    allLat, allLon = read_coordinate(file_geo)
    gr_lat , gr_lon = grid_coordinate(map_boundaries, gridSize_lat, gridSize_lon)  
        
    print(" ********** EV_250_Aggr1km_RefSB [1. 2.] bands ********")
    print(" ********** EV_500_Aggr1km_RefSB [3. 4. 5. 6. 7.] bands ********")
    print(" ********** EV_1KM_RefSB [ 8.   9.  10.  11.  12.  13.  13.5 14.  14.5 15.  16.  17.  18.  19. 26.] bands ********")
    print(" ********** EV_1KM_Emissive [20. 21. 22. 23. 24. 25. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36.] bands ********")
      
    list_variable_name= { 0 : 'EV_250_Aggr1km_RefSB', 1 : 'EV_500_Aggr1km_RefSB', 2 : 'EV_1KM_RefSB', 3 : 'EV_1KM_Emissive'}
    band_list ={ 0 : 'Band_250M', 1 : 'Band_500M', 2 : 'Band_1KM_RefSB', 3 : 'Band_1KM_Emissive'}
    
    type_reflectance = {0:"reflectance", 1:"reflectance", 2:"reflectance"}
    #type_reflectance_emi = {0:"reflectance", 1:"reflectance", 2:"reflectance", 3:"radiance"}
    type_radiance= {0:"radiance", 1:"radiance", 2:"radiance", 3:"radiance"}
    # index_variables = [0,1,2,3]  
        
    if task == "spectral": #task == "get_refl_emiss":

        rad_MODIS_bands_output = []
        refl_MODIS_bands_output = []

        # MODIS_dataset_output_germany = np.empty((1,np.shape(gr_lat)[1],np.shape(gr_lat)[0])) #ch, lat,lon
        # MODIS_dataset_output = np.empty((1,np.shape(allLat)[0],np.shape(allLat)[1])) #ch, lat,lon
        # units_names = []
        refl_MODIS_dataset_output_germany = np.empty((1,np.shape(gr_lat)[1],np.shape(gr_lat)[0])) #ch, lat,lon
        refl_MODIS_dataset_output = np.empty((1,np.shape(allLat)[0],np.shape(allLat)[1])) #ch, lat,lon
        rad_MODIS_dataset_output_germany = np.empty((1,np.shape(gr_lat)[1],np.shape(gr_lat)[0])) #ch, lat,lon
        rad_MODIS_dataset_output = np.empty((1,np.shape(allLat)[0],np.shape(allLat)[1])) #ch, lat,lon
      
        print(" ********** Obtaining radiances ******************** ")
        rad_MODIS_dataset_output, rad_MODIS_dataset_output_germany, rad_MODIS_bands_output = get_spectral(data_file, path_dataset, map_boundaries, gridSize_lat, gridSize_lon, 
                      allLat, allLon, 
                      file_data, 
                      list_variable_name, band_list,
                      MODIS_dataset_output = rad_MODIS_dataset_output, 
                      MODIS_dataset_output_germany = rad_MODIS_dataset_output_germany,
                      MODIS_bands_output = rad_MODIS_bands_output,
                      type_data =type_radiance)
                

        print(" ********** Obtaining reflectances ******************** ")
        refl_MODIS_dataset_output, refl_MODIS_dataset_output_germany, refl_MODIS_bands_output = get_spectral(data_file, path_dataset, map_boundaries, gridSize_lat, gridSize_lon, 
                      allLat, allLon, 
                      file_data, 
                      list_variable_name, band_list,
                      MODIS_dataset_output = refl_MODIS_dataset_output, 
                      MODIS_dataset_output_germany = refl_MODIS_dataset_output_germany,
                      MODIS_bands_output = refl_MODIS_bands_output,
                      type_data =type_reflectance)
        
        
        ch_thermal_zeros = np.zeros((16,np.shape(gr_lat)[1],np.shape(gr_lat)[0])) #ch, lat,lon

        refl_MODIS_dataset_output_germany= np.dstack((refl_MODIS_dataset_output_germany.transpose(1,2,0), ch_thermal_zeros.transpose(1,2,0))).transpose(2,0,1) 
        
        

        print(('================= Saving and dataframe {} Germany===================').format(task))

        name_file = "{}/MODIS_T1140_Germany_{}.nc".format(path_dataset, data_file[:-4])
        #save_data('MODIS_Germany_radiances', name_file, MODIS_dataset_output_germany, gr_lat, gr_lon, MODIS_bands_output)
        save_ncfile(name_file,
                    gr_lat[0,:], gr_lon[:,0], 
                    rad_MODIS_dataset_output_germany,
                    refl_MODIS_dataset_output_germany,
                    rad_MODIS_bands_output)  
                    # units_names)

        print((' ===========deleting 13.5 and 14.5 and organizing the channels Germany ============='))
        modis_ds = xr.open_dataset(name_file)
        modis_ds = modis_ds.drop_sel(chan =[13.5, 14.5])
        # Define the desired channel order
        desired_channels = list(range(1, 37))

        # Reorganize the channels in the desired order using reindex
        organized_data = modis_ds.reindex(chan=desired_channels)

        # Check the resulting order of channels
        print(organized_data.chan.values)
        # Save the organized MODIS data to a netCDF file
        name_file = "{}/organized_chann_MODIS_T1140_Germany_{}.nc".format(path_dataset, data_file[:-4])
        organized_data.to_netcdf(name_file)
        print(' ------------------- Dataset was created! ', name_file)
        modis_ds.close()
        
        print((' ========================'))

        # visulize_sat(data_MODIS_38bands,bands_radiances_38bands, allLat, allLon, cbar_label, titel, map_boundaries)


        # print("*********** testt********************************************************")
        # map_boundaries=np.zeros(4)
        # map_boundaries[0] = allLat.min() #38.61691 
        # map_boundaries[1] = allLat.max() #60.485966
        # map_boundaries[2] = allLon.min() #-3.8685935 
        # map_boundaries[3] = allLon.max() #35.7941

        # titel = 'MODIS_radiances_total_EV_1KM_RefSB' #aca pasarr con todo path
        # visulize_sat(valid_var_EV_1KM_Emissive[:4],bands_EV_1KM_Emissive[:4], allLat, allLon, cbar_label, titel, map_boundaries)

        # print("***********fin testt********************************************************")

        #sys.stdout.close()

        

    elif task == "get_RGB":
        reflectance_rgb = rgb_image(out_file = path_output, myd021km_file = file_data, lat = allLat, lon = allLon) #reflectance_rgb 2030,1354,3  x,y,ch
        
#         rgb_germany = np.empty((1,np.shape(gr_lat)[0],np.shape(gr_lat)[1]))
        
#         print(np.shape(reflectance_rgb),len(reflectance_rgb.transpose(1,2,0)))  
#         grid_valid_variable = gridding_variable(reflectance_rgb.transpose(2,0,1), map_boundaries, gridSize, allLat, allLon)
#         rgb_germany= np.dstack((rgb_germany.transpose(1,2,0), grid_valid_variable.transpose(1,2,0))).transpose(2,0,1)
#         print(np.shape(rgb_germany),len(reflectance_rgb.transpose(1,2,0)))  

#         rgb_germany = np.delete( rgb_germany, 0,axis=0)
        
#         print(np.shape(rgb_germany),len(reflectance_rgb.transpose(1,2,0)))  

#         data_shape = rgb_germany.shape
#         rgb_germany = np.ma.masked_equal(rgb_germany, 0) 
#         plot_rgb_image(along_track = data_shape[1] , 
#                        cross_trak = data_shape[2] , 
#                        z = rgb_germany.transpose(1,2,0), 
#                        out_file = args.path_output, 
#                        name_plot = "Germany")
        

    #file_germany = SD(os.sep.join([args.path_dataset,args.germany_area_MODIS]), SDC.READ)
    #selected_sds = file_germany.select('MODIS_Germany_radiances')
    
    # Close the netCDF file
    file_data.end()
    file_geo.end()


if __name__ == '__main__':
    main()