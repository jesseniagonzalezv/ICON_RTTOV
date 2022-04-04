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

from modis_functions import dataset_dic,grid_coordinate, grid, gridding_variable, read_bands, read_coordinate, read_level1_radiances, visulize_sat, save_data, dataframe_csv,save_ncfile, rgb_image, plot_rgb_image

#%matplotlib inline


    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-dataset', type = str, default = '/home/jvillarreal/Documents/phd/dataset', help = 'Path of the dataset')
    arg('--path-output', type = str, default = '/home/jvillarreal/Documents/phd/output', help = 'Path of the output' )
    arg('--task', type = str, default = 'get_radiances', help = 'option to only get radiances or rgb' )

    args = parser.parse_args()

    data_file = 'MYD021KM.A2013122.1140.061.2018046032403.hdf'
    geolocation_file = 'MYD03.A2013122.1140.061.2018046005026.hdf'
   

    print("Reading MODIS level1B data")
    file_data = SD(os.sep.join([args.path_dataset,data_file]), SDC.READ)
    dataset_dic(file_data)
    
    
    #set the modis file L1 & geo
    print("Reading geolocation file")
    file_geo = SD(os.sep.join([args.path_dataset,geolocation_file]), SDC.READ)
    dataset_dic(file_geo)
    
    #Selecting the area of Germany
    map_boundaries=np.zeros(4)  ##
    map_boundaries[0] = 47.5 #llcrnrlat
    map_boundaries[1] = 54.5 #urcrnrlat
    map_boundaries[2] = 4.5 #llcrnrlon
    map_boundaries[3] = 14.5 #urcrnrlon
    gridSize=0.01 # 1 grado = 100 km => 0.01 grados = 1 km
    
    allLat, allLon = read_coordinate(file_geo)
    gr_lat , gr_lon = grid_coordinate(map_boundaries,gridSize)  ###!!!!!! verifiy thisss como se que cuadra exacto ..poner mas decimales
        
    if args.task == "get_radiances":
        #sys.stdout = open(os.sep.join([args.path_output,'log_MODIS_LEVEL1B.txt']) ,'w')

        # #save the names of the data and geolocation (.txt file)
        # get_data_geolocation_names(path_dataset= args.path_dataset, filename_datas= "MYD021KM*.hdf", geolocation_filenames= "MYD03*.hdf")

        # data_files=os.sep.join([args.path_dataset,'file_list.txt']) 
        # geolocation_files=os.sep.join([args.path_dataset,'geolocation_files_list.txt']) 

        # #--opening L1 and geo file list
        # data_list=open(data_files, 'r').read().splitlines()
        # geo_list=open(geolocation_files, 'r').read().splitlines()


        print("********** EV_250_Aggr1km_RefSB [1. 2.] ********")
        print("********** EV_500_Aggr1km_RefSB [3. 4. 5. 6. 7.] ********")
        print("********** EV_1KM_RefSB [ 8.   9.  10.  11.  12.  13.  13.5 14.  14.5 15.  16.  17.  18.  19. 26.] ********")
        print("********** EV_1KM_Emissive [20. 21. 22. 23. 24. 25. 27. 28. 29. 30. 31. 32. 33. 34. 35. 36.] ********")

        #bounds it give the range of the colors but i want it get adapted to the data
        titel = 'MODIS_radiances' #aca pasarr con todo path
        cbar_label = 'Radiance $(W\,m^{-2}\,sr^{-1}\,\mu m^{-1})$'

        list_variable_name= { 0 : 'EV_250_Aggr1km_RefSB', 1 : 'EV_500_Aggr1km_RefSB', 2 : 'EV_1KM_RefSB', 3 : 'EV_1KM_Emissive'}
        band_list ={ 0 : 'Band_250M', 1 : 'Band_500M', 2 : 'Band_1KM_RefSB', 3 : 'Band_1KM_Emissive'}


        simulate_variable = [0,1,2,3]  # [2] [0] #




        MODIS_dataset_output_germany = np.empty((1,np.shape(gr_lat)[0],np.shape(gr_lat)[1]))
        MODIS_bands_output = []
        MODIS_dataset_output = np.empty((1,np.shape(allLat)[0],np.shape(allLat)[1])) 


        for index_name in simulate_variable: 
            var_name = list_variable_name[index_name]
            bands_name = band_list[index_name]

            variable_raw, valid_variable = read_level1_radiances(file_data, var_name)  #talves puedo gurarde este output y las bands en otro array x si quiero usar
            MODIS_dataset_output= np.dstack((MODIS_dataset_output.transpose(1,2,0), valid_variable.transpose(1,2,0))).transpose(2,0,1)

            bands_variable = read_bands(file_data,bands_name)
            MODIS_bands_output = np.concatenate((MODIS_bands_output, bands_variable), axis=0) 

            grid_valid_variable = gridding_variable(valid_variable, map_boundaries, gridSize, allLat, allLon)
            MODIS_dataset_output_germany= np.dstack((MODIS_dataset_output_germany.transpose(1,2,0), grid_valid_variable.transpose(1,2,0))).transpose(2,0,1)

            #print('+++++++++++++++++++++++++++++values after gridding+++++++++++++++++++++++++++')
            #pprint.pprint(grid_valid_variable)

           # visulize_sat(grid_valid_variable, bands_variable, gr_lat, gr_lon, cbar_label, titel, map_boundaries )
        

        print("saving ")    
        MODIS_dataset_output = np.delete( MODIS_dataset_output, 0,axis=0)
        MODIS_dataset_output_germany = np.delete( MODIS_dataset_output_germany, 0,axis=0)

        print('save in netcdf')


        name_file = "{}/MODIS_radiances_{}".format(args.path_dataset, data_file)
        print('=================data output  total===================',MODIS_dataset_output[0],np.shape(MODIS_dataset_output), np.min(MODIS_dataset_output),np.max(MODIS_dataset_output))

        #save_data(name_variable = 'MODIS_radiances', name_file = name_file, array = MODIS_dataset_output, lat =allLat, lon = allLon, bands = MODIS_bands_output)

        variable_output = MODIS_dataset_output
        bands = MODIS_bands_output
        name_file = "{}/MODIS_radiances_{}".format(args.path_dataset, data_file[:-4]) #dont include hdf
        dataframe_csv(variable = variable_output, colum = bands, out_file = name_file)

    ####
        print('=================data output  germany===================',MODIS_dataset_output_germany[0],np.shape(MODIS_dataset_output_germany), 'gr_lat:', np.shape(gr_lat), 'gr_lon:', np.shape(gr_lon), np.min(MODIS_dataset_output_germany),np.max(MODIS_dataset_output_germany))

        name_file = "{}/MODIS_Germany_radiances_{}.nc".format(args.path_dataset, data_file[:-4])

        #save_data('MODIS_Germany_radiances', name_file, MODIS_dataset_output_germany, gr_lat, gr_lon, MODIS_bands_output)
        save_ncfile('MODIS_Germany_radiances', name_file, MODIS_dataset_output_germany, gr_lat, gr_lon, MODIS_bands_output)

        variable_output = MODIS_dataset_output_germany
        bands = MODIS_bands_output
        name_file = "{}/MODIS_Germany_radiances_{}".format(args.path_dataset, data_file[:-4]) #dont include hdf
        dataframe_csv(variable = variable_output, colum = bands, out_file = name_file)








        ###check if i want to save #talves puedo gurarde este output y las bands en otro array x si quiero usar  ..deberia para analizar 



        print("*******************************************************************")
        titel = 'Modis_initial_values' #aca pasarr con todo path
        cbar_label = '-'

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


    elif args.task == "get_RGB":
        reflectance_rgb = rgb_image(out_file = args.path_output, myd021km_file = file_data) #reflectance_rgb 2030,1354,3  x,y,ch
        
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

if __name__ == '__main__':
    main()