import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import argparse
import pandas as pd
from eofs.xarray import Eof
from sklearn import preprocessing
from sklearn.decomposition import PCA

from lwp_nd import lwp_nd_input_ICON
from PCA_radiances import PCA_calculation, variance_sklearn_plot, dataframe_csv #, convert_3D

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, check_scoring, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os

import joblib
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor


'''
def get_rmse_array(truth, pred):
    print(truth.shape, pred.shape)
    weights = np.cos(np.deg2rad(truth.lat))

    return np.sqrt(((truth - pred.reshape(-1, 96,144))**2).weighted(weights).mean(['lat', 'lon'])).data.mean()


def get_training_inputs(path_ICON):
    ds = xr.open_dataset(path_ICON).compute()
   
    p_2013 = ds.pres.values #/100 #convert to hPa
    T_2013 = ds.ta.values
    q_2013 = ds.hus.values
    max_cdnc_2013_cm = ds.Nd_max.values
    lwp_2013 = ds.lwp.values
    #lat = ds.lat
    #lon = ds.lot
    #height = ds.height

    # call pca, new to run it again and kmwasn but also adapt inputs array of inputs, array of outpus, usig of reference code of clibenc
    # train_files = [ "T09", "T12" ]
    # train_inputs_variables_1d = [ 'height','lat', 'lon']
    # train_inputs_variables_2d = [ "t2m", 'q2m', 'p_surf', 'u_surf', 'v_surf', 't_skin', 'h_surf', 'lat', 'lon', 'qnc', 'lwp']
    # train_inputs_variables_3d = [ "p", "t", "q", 'tca', 'lwc', 'iwc']
    
    train_inputs_variables_2D = { "Nd_max": Nd_max, "lwp": lwp}  # ['Nd_max', 'lwp'] #(lat, lon)
    train_inputs_variables_3D = { "pres": p_2013, "ta": T_2013, "hus":q_2013} #(height, lat, lon)
    
    ############ variables 2D ######################################
    df_train_inputs_variables_2D= pd.DataFrame({  #falta normalizar
    "Nd_max" : ds.Nd_max.values.flatten(),
    "LWP" : ds["lwp"].values.flatten() }) 

    print("============== 2D variables statistics ==================== ")
    print("LWP shape: {}".format(np.shape(ds["lwp"].values)))
    print(df_train_inputs_variables_2D.describe())#if i mask i will have nan instead of Nd<2 and LWP<o
    # print("LWP min: {}, max: {}".format(ds["lwp"].values.min(), np.max(ds["lwp"].values)))
    # print("Nd_max min: {}, max: {}".format(ds["Nd_max"].values.min(), np.max(ds["Nd_max"].values)))
    count_nan_in_df = pd.DataFrame(df_train_inputs_variables_2D).isnull().sum()
    print ("=========print NaNs: \n", count_nan_in_df)  #can i used 0 in the lwp and nd?
    # df = df.fillna(0) #there is not nan

    print("============== 3D variables statistics ==================== ")
    print("pres min: {}, max: {}, mean: {}".format(np.min(p_2013), np.max(p_2013), np.mean(p_2013)))
    print("ta min: {}, max: {}, mean: {}".format(ds["ta"].values.min(), np.max(ds["ta"].values),ds["ta"].values.mean()))
    print("hus min: {}, max: {}, mean: {}".format(ds["hus"].values.min(), np.max(ds["hus"].values), ds["hus"].values.mean()))

    return df_train_inputs_variables_2D, train_inputs_variables_2D, train_inputs_variables_3D 

    

### the next should be the last version 
def read_data_refl_emiss_rttov(path_rttov_test):
    
    #output: refl_emmiss #(HxWxCH)
    
    rttov_ds_refl_emmi = xr.open_dataset(path_rttov_test).compute()
    rttov_variable = np.zeros((np.shape(rttov_ds_refl_emmi['radiances_total'].values)))
    
    
    print("****************variables shape ", np.shape(rttov_ds_refl_emmi['bt_refl_total'].values), np.shape(rttov_variable), np.shape(rttov_ds_refl_emmi['radiances_total'].values))
    

            
    rttov_variable[:19] = rttov_ds_refl_emmi['bt_refl_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
    rttov_variable[19:25] = rttov_ds_refl_emmi['radiances_total'][19:25]
    rttov_variable[25] = rttov_ds_refl_emmi['bt_refl_total'][25] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    rttov_variable[26:36] = rttov_ds_refl_emmi['radiances_total'][26:36]  

    print('rttov_variable',np.shape(rttov_variable))
    
    rttov_ds_refl_emmi.close()
        
    #rttov_bands =rttov_ds_rad['chan'].values
    # print('rttov_variable',np.shape(rttov_variable))
    refl_emmiss =  rttov_variable[:,9:,] #.transpose(1,2,0) #output then I dont neet to cut ######################
    
    rttov_variable_ds = xr.DataArray(rttov_variable, dims=['chan','lat','lon'], coords= [rttov_ds_refl_emmi.chan.data, rttov_ds_refl_emmi.lat.data, rttov_ds_refl_emmi.lon.data ])
        

    return  rttov_variable_ds

    #return  refl_emmiss, rttov_bands

    
def old_scaler_PCA_input(train_inputs_2D, train_inputs_3D, path_output):
    
    #input: dataframe input with 2D and 3D
    #output: x_train_df dataframe with the variable 2D and 3D PCAs
    
    
    n_pca_variables_3D = { "pres": 5, "ta": 10, "hus":24} #(height, lat, lon)
        
#     scaler_2D = preprocessing.StandardScaler()# Fit on training set only.
#     scaler_2D.fit(df_train_inputs_variables_2D)
    scaler_2D = preprocessing.StandardScaler().fit(df_train_inputs_variables_2D)

        
    training_input_variables_df = pd.DataFrame(scaler_2D.transform(df_train_inputs_variables_2D), columns = ["Nd_max", "lwp"]) 

    # Get PC of the variables 3D
    scaler_3D = []
    pca_3D = []

    for key, var in train_inputs_variables_3D.items():  
        var = var.values.transpose(1,2,0) #lat,lot,heigh
        var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
        var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
        # scaler = preprocessing.StandardScaler().fit(var_df) 
        scaler = preprocessing.StandardScaler()# Fit on training set only. 
        scaler.fit(var_df)  #antes esto estaba despues 
        #norm = preprocessing.Normalizer().fit(xtrain)
        scaler_3D.append(scaler)
        
        var_scaled = scaler.transform(var_df)
        
        
        n_pca = n_pca_variables_3D[key] #150
        name_plot= "{}_Explained_variance_{}_variable".format(n_pca, key)
        print("============== Variable: {}  ========================================".format(key))

        X_pca, pca = PCA_calculation(var_scaled, name_plot,n_pca, path_output)
        pca_3D.append(pca)
        
        # print( 'Original shape: {}'.format(str(PC.shape)))
        print( 'Original shape: {}'.format(str(var.shape)))
        print( 'Reduced shape: {}'.format(str(X_pca.shape)))
        principalDf = pd.DataFrame(data = X_pca
             , columns = [f"{key}_PCA_{i}" for i in range(n_pca)])
        training_input_variables_df=pd.concat([training_input_variables_df, principalDf], axis=1)



        # training_input_variables_df=pd.concat([training_input_variables_df, var_df], axis=1)



    # for key, var in train_inputs_variables_3D.items():  
    #     var = var.transpose(1,2,0)
    #     var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
    #     var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
    #     training_input_variables_df=pd.concat([training_input_variables_df, var_df], axis=1)


    print("================ dataframe =============================")
    training_input_variables_df.describe().to_csv(path_output + "/inputs_after_PCA_StandardScaler.csv")    
    # count_nan 
    count_nan_in_df = training_input_variables_df.isnull().sum()
    print("================ values of Nan in training_input_variables_df =====================")

    print (count_nan_in_df)  #can i used 0 in the lwp and nd?
    
    #X_train_np = np.array([X_train_reshaped['pres'].data, X_train_reshaped['ta'].data, X_train_reshaped['hus'].data, X_train_reshaped['qnc'].data])

    # pd.set_option('display.float_format', lambda x: '%.1f' % x)
    # df.columns= colum
    

    return training_input_variables_df, scaler_2D, scaler_3D, pca_3D, n_pca_variables_3D

 
  
'''
 

    

def read_data_refl_emiss_rttov( rttov_path_rad, rttov_path_refl_emmis): #rttov_path
  
    #input: path of the radiances and reflectances
    #output: refl_emmiss CHxHxW  
    
    rttov_ds_rad = xr.open_dataset(rttov_path_rad).compute()  # write read rttov in a function
    rttov_ds_refl_emmi = xr.open_dataset(rttov_path_refl_emmis).compute()    
    # rttov_ds = xr.open_dataset(rttov_path).compute()    
    
    
    rttov_variable = np.zeros((np.shape(rttov_ds_rad['Y'].values)))
    # rttov_variable = np.zeros((np.shape(rttov_ds['Radiance_total'].values)))

    # print("****************variables shape ", np.shape(rttov_ds_refl_emmi['bt_refl_total'].values), np.shape(rttov_variable), np.shape(rttov_ds_rad['Y'].values))


    rttov_variable[:19] = rttov_ds_refl_emmi['bt_refl_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
    rttov_variable[19:25] = rttov_ds_rad['Y'][19:25]
    rttov_variable[25] = rttov_ds_refl_emmi['bt_refl_total'][19] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    rttov_variable[26:36] = rttov_ds_rad['Y'][26:36]
    
    # rttov_variable[:19] = rttov_ds['BRF_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
    # rttov_variable[19:25] = rttov_ds['Radiance_total'][19:25]
    # rttov_variable[25] = rttov_ds['BRF_total'][25] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    # rttov_variable[26:36] = rttov_ds['Radiance_total'][26:36]
    

    
    print("===================================== Training output ====================================== ")

    print("****************variables shape training_output", np.shape(rttov_variable))
    
    rttov_ds_rad.close()
    rttov_ds_refl_emmi.close()
    # rttov_ds.close()
     
    #rttov_bands =rttov_ds_rad['chan'].values
    # print('rttov_variable',np.shape(rttov_variable))
    
    #output then I dont neet to cut   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rttov_variable_ds = xr.DataArray( rttov_variable[:,9:,], dims=['channel','lat','lon'], coords= [rttov_ds_rad.chan.data, rttov_ds_refl_emmi.lat.data[9:], rttov_ds_refl_emmi.lon.data ])
    # rttov_variable_ds = xr.DataArray( rttov_variable, dims=['channel','lat','lon'], coords= [rttov_ds.Channel.data, rttov_ds.Latitude.data, rttov_ds.Longitude.data])      

    return  rttov_variable_ds  
   
 
def get_split_data_xarray(path_ICON, path_output, k_fold, rttov_path_rad, rttov_path_refl_emmis): #rttov_path
    '''
    Input: path of the x  and y  (x = ICON-LES, y = RTTOV-radiances/reflectances)
           path_ICON:
                   x: 3D variables height: 150, lat: 628, lon: 589
                      2D variables lat: 628, lon: 589
           rttov_path_rad, rttov_path_refl_emmis:
                   y: 3D variables Channel: 36, Latitude: 628, Longitude: 589
                      reflectances: 1-19,26  Channel  
                      radiances: 20-25,27-36 Channel 
           path_output: path in which the output will be saved

    Output:
    
    dataframes 
    x_train_2D, x_train_3D (height, lat, lon) = ICON-LES lat < threshold
    x_test_2D, x_test_3D (height, lat, lon) =  ICON-LES lat > threshold
    y_train, y_test (Channel: 36, Latitude, Longitude)  = radiances/reflectances 
    df_x_train, df_x_test: dataframes of the ICON-LES inputs
                           dataframe 2D variables (row: lat x lon(H*W))
                           dataframe 3D variables (row: lat x lon(H*W), columns(height))
    df_y_train, df_y_test: dataframes of the reflectances/radiances
                           dataframe variables (row: Latitude x Longitude(H*W), columns(Channel))

    '''
    ds = xr.open_dataset(path_ICON).compute()
    # rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path)
    rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path_rad, rttov_path_refl_emmis)


    #================================= X ========================================

    # ds_x_train = ds.sel(lat=slice(50,55))  #(0,53))  #  # I am keeping the higher area
    # ds_x_test = ds.sel(lat=slice(48,50)) #(53.01,60)) 

    # ds_x_train = ds.sel(lon=slice(0,11))  #(0,53))  #  # I am keeping the higher area
    # ds_x_test = ds.sel(lon=slice(11,16)) #(53.01,60)) 
    # ds_x_train = ds.sel(lat=slice(0,49))  #(0,53))  #  # I am keeping the higher area
    # ds_x_test = ds.sel(lat=slice(49,55)) #(53.01,60)) 
    
    # ################### K folds ###############################
    if (k_fold == 5):
        ds_x_train = ds.sel(lat=slice(48.97, 54.47))
        ds_x_test = ds.sel(lat=slice(47.5, 48.97) )
        
        y_train = rttov_variable_ds.sel(lat=slice(48.97, 54.47) ) #(0,53)) 
        y_test = rttov_variable_ds.sel(lat=slice(47.5, 48.97)) #(53.01, 60)) 
    
    elif (k_fold == 4):
        ds_x_train = xr.concat([ds.sel(lat=slice(47.5, 48.97)), ds.sel(lat=slice(50.34, 54.47))], dim="lat")
        ds_x_test = ds.sel(lat=slice(48.97, 50.34))

        y_train = xr.concat([rttov_variable_ds.sel(lat=slice(47.5, 48.97)), rttov_variable_ds.sel(lat=slice(50.34, 54.47))], dim="lat")  
        y_test = rttov_variable_ds.sel(lat=slice(48.97, 50.34)) #(53.01, 60))  
   
    elif (k_fold == 3):      
        ds_x_train = xr.concat([ds.sel(lat=slice(47.5, 50.34)), ds.sel(lat=slice(51.72, 54.47))], dim="lat")
        ds_x_test = ds.sel(lat=slice(50.34, 51.72))
         
        y_train = xr.concat([rttov_variable_ds.sel(lat=slice(47.5, 50.34)), rttov_variable_ds.sel(lat=slice(51.72, 54.47))], dim="lat") 
        y_test = rttov_variable_ds.sel(lat=slice(50.34, 51.72)) #(53.01, 60))  
    
    elif (k_fold == 2):
        ds_x_train = xr.concat([ds.sel(lat=slice(47.5, 51.72)), ds.sel(lat=slice(53.09, 54.47))], dim="lat")
        ds_x_test = ds.sel(lat=slice(51.72, 53.09))
        
        y_train = xr.concat([rttov_variable_ds.sel(lat=slice(47.5, 51.72)), rttov_variable_ds.sel(lat=slice(53.09, 54.47))], dim="lat") 
        y_test = rttov_variable_ds.sel(lat=slice(51.72, 53.09)) #(53.01, 60))  
    
    elif (k_fold == 1):
        ds_x_train = ds.sel(lat=slice(47.5, 53.09))
        ds_x_test = ds.sel(lat=slice(53.09, 54.47) )
    
        y_train = rttov_variable_ds.sel(lat=slice(47.5, 53.09) ) #(0,53)) 
        y_test = rttov_variable_ds.sel(lat=slice(53.09, 54.47)) #(53.01, 60)) 
    
    
    ds.close()
    #================================= X ========================================    
    # y_train = rttov_variable_ds.sel(lat=slice(50,55)) #(0,53)) 
    # y_test = rttov_variable_ds.sel(lat=slice(48,50)) #(53.01, 60))  
    
    # y_train = rttov_variable_ds.sel(lon=slice(0,11)) #(0,53)) 
    # y_test = rttov_variable_ds.sel(lon=slice(11,16)) #(53.01, 60)) 
#     y_train = rttov_variable_ds.sel(lat=slice(0,49)) #(0,53)) 
#     y_test = rttov_variable_ds.sel(lat=slice(49,55)) #(53.01, 60))

    # ################################################################################
 
    
    lat_test_ds = y_test.lat
    lon_test_ds = y_test.lon

    
        
    # df_x_train.drop(df_x_train[(df_x_train['Nd_max'] <400) & (df_x_train['lwp'] < 600)].index, inplace=True)
   
    #=============================Dataframe X  ====================================    

    
#     x_train_2D = { "Nd_max": np.log((ds_x_train.Nd_max.values/np.max(ds_x_train.Nd_max.values)) + 1.0e-16), "lwp": np.log(ds_x_train.lwp.values/np.max(ds_x_train.lwp.values) + 1.0e-16)}  #, "clc": ds_x_train.clc.values*100
    
        
#     x_train_3D = { "pres": np.log(ds_x_train.pres.values/np.max(ds_x_train.pres.values)), "ta": np.log(ds_x_train.ta.values / np.max(ds_x_train.ta.values)), "hus": np.log(ds_x_train.hus.values/np.max(ds_x_train.hus.values))}
    
 
     # x_test_2D = { "Nd_max": np.log((ds_x_test.Nd_max.values / np.max(ds_x_test.Nd_max.values)) + 1.0e-16) , "lwp": np.log((ds_x_test.lwp.values/np.max(ds_x_test.lwp.values)) + 1.0e-16)}  #, "clc": ds_x_test.clc.values*100
#     x_test_3D = { "pres": np.log(ds_x_test.pres.values/ np.max(ds_x_test.pres.values)), "ta": np.log(ds_x_test.ta.values/np.max(ds_x_test.ta.values)), "hus": np.log(ds_x_test.hus.values/ np.max(ds_x_test.hus.values))}


    x_train_2D = { "Nd_max": np.log(ds_x_train.Nd_max.values+ 1.0e-16), "lwp": np.log(ds_x_train.lwp.values+ 1.0e-16)}  
    x_train_3D = { "pres": ds_x_train.pres.values, "ta": ds_x_train.ta.values, "hus": ds_x_train.hus.values}
    
    # print(x_train_2D.describe())


    
    x_test_2D = { "Nd_max": np.log(ds_x_test.Nd_max.values+ 1.0e-16), "lwp": np.log(ds_x_test.lwp.values+ 1.0e-16)}  
    x_test_3D = { "pres": ds_x_test.pres.values, "ta": ds_x_test.ta.values, "hus": ds_x_test.hus.values}
    
    
    ds_x_train.close()
    ds_x_test.close()
    
    #=============================Dataframe X  ====================================    
    df_x_train = pd.DataFrame()
    df_x_test = pd.DataFrame()
    
    for key, var in x_train_2D.items():  
        df = pd.DataFrame(data = var.flatten()
             , columns =  [key])
        df_x_train = pd.concat([df_x_train, df], axis=1)   
    for key, var in x_train_3D.items():  
        var = var.transpose(1,2,0) #lat,lot,heigh
        var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
        var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
        df_x_train=pd.concat([df_x_train, var_df], axis=1)
    
    print("=================== Dataframe training =======================================")    
    name_file = 'inputs_statistics_training'  
    df_x_train.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    
    for key, var in x_test_2D.items():  
        df = pd.DataFrame(data = var.flatten()
             , columns = [key])
        df_x_test = pd.concat([df_x_test, df], axis=1)   
    for key, var in x_test_3D.items():  
        var = var.transpose(1,2,0) #lat,lot,heigh
        var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
        var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
        df_x_test=pd.concat([df_x_test, var_df], axis=1)
        
    print("=================== Dataframe testing =======================================")    
    name_file = 'inputs_statistics_testing'  
    df_x_test.describe().to_csv("{}/{}.csv".format(path_output, name_file))    
    
    #================================= Y ========================================

    
    ###############################Dataframe Y_data##################################
    name_file = 'refl_emiss_statistics'
    df_y = dataframe_csv(variable = rttov_variable_ds.transpose('lat', 'lon', 'channel').values, 
                  column = rttov_variable_ds.channel.values,  
                  path_output = path_output, 
                  name_file = name_file)
    rttov_variable_ds.close()
           
    ############## convert img_rttov: because the input for the dataframe_csv need to be HxWxCH I use transpose ##########################
    name_file = 'refl_emiss_statistics_train'
    df_y_train = dataframe_csv(variable = y_train.transpose('lat', 'lon', 'channel').values, 
                  column = y_train.channel.values,  
                  path_output = path_output, 
                  name_file = name_file)
     
    
    # # df_y_train = preprocessing.normalize(df_y_train)

        
    name_file = 'refl_emiss_statistics_test'
    df_y_test = dataframe_csv(variable = y_test.transpose('lat', 'lon', 'channel').values, 
                  column = y_test.channel.values, 
                  path_output = path_output, 
                  name_file = name_file)
    
    # df_y_test = preprocessing.normalize(df_y_test)

    ###################################################################################
        

    y_train.close()
    y_test.close()
    
    return x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds



def get_split_data_xarray_alldata(path_ICON, rttov_path_rad, rttov_path_refl_emmis, path_output):
    '''
    Input: path of the X and Y. X = ICON-LES, Y = RTTOV-radiances/reflectances
    Output: xarray.DataArray  chan: 36, lat: 637, lon: 589
    
    dataframes 
    x_train_2D, x_train_3D (height, lat, lon) = ICON-LES lat < 53.01
    x_test_2D, x_test_3D (height, lat, lon) =  ICON-LES lat > 53.01
    y_train, y_test (chan: 36, lat, lon)  = radiances/reflectances 
    
    df_y_train: dataframe  row = H*W, colum = CH  
    '''
    #================================= x ========================================
    ds = xr.open_dataset(path_ICON).compute()
    #================================= y ========================================
    rttov_variable_ds = read_data_refl_emiss_rttov_old(rttov_path_rad, rttov_path_refl_emmis)  
    #=============================Dataframe X  ====================================    
    x_train_2D_all = { "Nd_max": ds.Nd_max.values, "lwp": ds.lwp.values}          
    x_train_3D_all = { "pres": ds.pres.values, "ta": ds.ta.values, "hus": ds.hus.values}
    ds.close()    
    #=============================Dataframe X  ====================================    
    df_x_all = pd.DataFrame()
    
    for key, var in x_train_2D_all.items():  
        df = pd.DataFrame(data = var.flatten()
             , columns =  [key])
        df_x_all = pd.concat([df_x_all, df], axis=1)   
        
    for key, var in x_train_3D_all.items():  
        var = var.transpose(1,2,0) #lat,lot,heigh
        var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
        var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
        df_x_all=pd.concat([df_x_all, var_df], axis=1)
    
    print("=================== Dataframe all =======================================")    
    name_file = 'inputs_statistics_all_data'  
    df_x_all.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    
    ###############################Dataframe Y_data##################################
    name_file = 'refl_emiss_statistics'
    df_y_all = dataframe_csv(variable = rttov_variable_ds.transpose('lat', 'lon', 'chan').values, 
                  column = rttov_variable_ds.chan.values,  
                  path_output = path_output, 
                  name_file = name_file)
    
#     df_permutated = df.sample(frac=1)

#     train_size = 0.8
#     train_end = int(len(df_permutated)*train_size)

#     df_train = df_permutated[:train_end]
#     df_test = df_permutated[train_end:]

    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x_all, df_y_all, test_size=0.33, random_state=42)
    # print("==========================convert dataframe ======================")
    # print(x_train)
    df_x_train.columns = df_x_all.columns
    df_x_test.columns = df_x_all.columns
    df_y_train.columns = df_y_all.columns
    df_y_test.columns = df_y_all.columns

    # df_x_train = x_train
    # df_x_test = x_test
    # df_y_train = y_train
    # df_y_test = y_test
    print("=================== Dataframe training saved =======================================")    
    name_file = 'inputs_statistics_training'  
    df_x_train.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    
    print("=================== Dataframe testing  saved=======================================")    
    name_file = 'inputs_statistics_testing'  
    df_x_test.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    

    return x_train_2D_all, x_train_3D_all,  df_x_train, df_x_test, df_y_train, df_y_test



  
def scaler_PCA_input(df_x_train, path_output):
    '''
    input: dataframe input with 2D and 3D
    output: x_train_df dataframe with the variable 2D and 3D PCAs
    '''
    variables_2D = { "Nd_max", "lwp"}  
    n_pca_variables_3D = { "pres": 5, "ta": 10, "hus":24} #(height, lat, lon)
    # scaler_variables = { "Nd_max", "lwp", "pres", "ta", "hus"}
    training_df_x_train = pd.DataFrame()

    scaler = preprocessing.StandardScaler().fit(df_x_train)   
    df = pd.DataFrame(scaler.transform(df_x_train), columns = df_x_train.columns) 

    print("=================== After scaler dataframe training saved=======================================")    
    name_file = 'after_scaler_inputs_statistics_training'  
    df.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    
  
    
    pca_3D = {}
    
    for key in variables_2D: 
        training_df_x_train=pd.concat([training_df_x_train, df[key]], axis=1)

    for key, var in n_pca_variables_3D.items():  
        var_scaled = df.filter(like=key)                 
        n_pca = var 
        name_plot= "{}_Explained_variance_{}_variable".format(n_pca, key)
        print("============== Variable: {}  ========================================".format(key))
        X_pca, pca = PCA_calculation(var_scaled.to_numpy(), name_plot,n_pca, path_output)
        #pca_3D.append(pca)
        pca_3D[key] = pca

        # print( 'Original shape: {}'.format(str(PC.shape)))
        print( 'Original shape: {}'.format(str(var_scaled.shape)))
        print( 'Reduced shape: {}'.format(str(X_pca.shape)))
        principalDf = pd.DataFrame(data = X_pca
             , columns = [f"{key}_PCA_{i}" for i in range(n_pca)])
        training_df_x_train = pd.concat([training_df_x_train, principalDf], axis=1)

    print("================ dataframe all after PCA saved=============================")
    training_df_x_train.describe().to_csv(path_output + "/inputs_after_PCA_StandardScaler.csv")    
    # count_nan 
    count_nan_in_df = training_df_x_train.isnull().sum()
    print("================ values of Nan in training_input_variables_df =====================")
    print (count_nan_in_df)  #can i used 0 in the lwp and nd?


    return training_df_x_train, scaler, pca_3D, n_pca_variables_3D

def scaler_PCA_input_all_3D(df_x_train, path_output):
    '''
    input: dataframe input with 2D and 3D
    output: x_train_df dataframe with the variable 2D and 3D PCAs
    '''
    variables_2D = { "Nd_max", "lwp"}  
    # n_pca_variables_3D = { "pres": 5, "ta": 10, "hus":24} #(height, lat, lon)

    # scaler_variables = { "Nd_max", "lwp", "pres", "ta", "hus"}
    training_df_x_train = pd.DataFrame()

    scaler = preprocessing.StandardScaler().fit(df_x_train)   
    df = pd.DataFrame(scaler.transform(df_x_train), columns = df_x_train.columns) 

    print("=================== After scaler dataframe training saved=======================================")    
    name_file = 'after_scaler_inputs_statistics_training'  
    df.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    
  

    # pca_3D = {}
    
    for key in variables_2D: 
        training_df_x_train=pd.concat([training_df_x_train, df[key]], axis=1)

        
    X = df.drop(columns =["lwp","Nd_max"])
    name_plot= "Explained_variance__variable_all"
    n_pca = 26 
    n_pca_variables_3D = n_pca
    X_pca, pca = PCA_calculation(X.to_numpy(), name_plot,n_pca, path_output)
    pca_3D = pca

    principalDf = pd.DataFrame(data = X_pca
         , columns = [f"3D_PCA_{i}" for i in range(n_pca)])
    training_df_x_train = pd.concat([training_df_x_train, principalDf], axis=1)

        
#     for key, var in n_pca_variables_3D.items():  
#         var_scaled = df.filter(like=key)                 
#         n_pca = var 
#         name_plot= "{}_Explained_variance_{}_variable".format(n_pca, key)
#         print("============== Variable: {}  ========================================".format(key))
#         X_pca, pca = PCA_calculation(var_scaled.to_numpy(), name_plot,n_pca, path_output)
#         #pca_3D.append(pca)
#         pca_3D[key] = pca

#         # print( 'Original shape: {}'.format(str(PC.shape)))
#         print( 'Original shape: {}'.format(str(var_scaled.shape)))
#         print( 'Reduced shape: {}'.format(str(X_pca.shape)))
#         principalDf = pd.DataFrame(data = X_pca
#              , columns = [f"{key}_PCA_{i}" for i in range(n_pca)])
#         training_df_x_train = pd.concat([training_df_x_train, principalDf], axis=1)

    print("================ dataframe all after PCA saved=============================")
    training_df_x_train.describe().to_csv(path_output + "/inputs_after_PCA_StandardScaler.csv")    
    # count_nan 
    count_nan_in_df = training_df_x_train.isnull().sum()
    print("================ values of Nan in training_input_variables_df =====================")
    print (count_nan_in_df)  #can i used 0 in the lwp and nd?


    return training_df_x_train, scaler, pca_3D, n_pca_variables_3D



def PCA_read_input_target(df_y_train, path_output):
    '''
    PC_output (latxlon, number_pcs)
    '''
###########################################
    df_y_train_normalized = preprocessing.normalize(df_y_train)
###########################################3
    
    # scaler_y = preprocessing.StandardScaler().fit(df_y_train)  #Standardize features by removing the mean and scaling to unit variance
    # X_scaled = scaler_y.transform(df_y_train)

    scaler_y = preprocessing.StandardScaler().fit(df_y_train_normalized)  #Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler_y.transform(df_y_train_normalized)

    ###### analysis  PCA###########################
    name_plot= "Explained_variance_refl_emiss"
    n_pca = 6 
    n_bands = len(df_y_train.columns) # rttov_bands) #2 #test JQ     
    X_reduced_output, pca_y = PCA_calculation(X_scaled, name_plot, n_pca, path_output) #PC_output_all

    principalDf = pd.DataFrame(data = X_reduced_output
             , columns = [f"PCA_{i}" for i in range(np.shape(X_reduced_output)[1])], index = df_y_train.index)
    print("columns output", principalDf.columns)
        
    print("=================== After scaler and pca output training Dataframe =======================================")   
    name_file = 'after_scaler_pca_output_statistics_training'  
    principalDf.describe().to_csv("{}/{}.csv".format(path_output, name_file)) 
    
    
    return scaler_y, principalDf, pca_y



    
##################### 
def get_test_input(path_output, df_x_test, scaler, pca_3D, n_pca_variables_3D):
    '''
    PC_output (latxlon, number_pcs)
    '''
    testing_df_x_test = pd.DataFrame()
    
    variables_2D = { "Nd_max", "lwp"}  
    # variables_3D = { "pres", "ta", "hus"} 

        
    df = pd.DataFrame(scaler.transform(df_x_test), columns = df_x_test.columns) 
    
    print("=================== After scaler Dataframe testing =======================================")    
    name_file = 'after_scaler_inputs_statistics_testing'  
    df.describe().to_csv("{}/{}.csv".format(path_output, name_file))  
    
    
    for key in variables_2D: 
        testing_df_x_test=pd.concat([testing_df_x_test, df[key]], axis=1)

        
    for key in n_pca_variables_3D:  
        var_scaled = df.filter(like=key)   
        principalDf = pd.DataFrame(data = pca_3D[key] .transform(var_scaled.to_numpy()) #pca_3D [i] i de 1 
             , columns = [f"{key}_PCA_{i}" for i in range(n_pca_variables_3D[key])])
        testing_df_x_test = pd.concat([testing_df_x_test, principalDf], axis=1)
      
    

    return testing_df_x_test



def get_test_input_PCA_all_3D(path_output, df_x_test, scaler, pca_3D, n_pca_variables_3D):
    '''
    PC_output (latxlon, number_pcs)
    '''
    testing_df_x_test = pd.DataFrame()
    
    variables_2D = { "Nd_max", "lwp"}  
    # variables_3D = { "pres", "ta", "hus"} 

        
    df = pd.DataFrame(scaler.transform(df_x_test), columns = df_x_test.columns) 
    
    print("=================== After scaler Dataframe testing =======================================")    
    name_file = 'after_scaler_inputs_statistics_testing'  
    df.describe().to_csv("{}/{}.csv".format(path_output, name_file))  
    
    
    for key in variables_2D: 
        testing_df_x_test=pd.concat([testing_df_x_test, df[key]], axis=1)

        
    X = df.drop(columns =["lwp","Nd_max"])
    principalDf = pd.DataFrame(data = pca_3D.transform(X.to_numpy()) #pca_3D [i] i de 1 
                               
         , columns = [f"3D_PCA_{i}" for i in range(n_pca_variables_3D)])
    testing_df_x_test = pd.concat([testing_df_x_test, principalDf], axis=1)

#     for key in n_pca_variables_3D:  
#         var_scaled = df.filter(like=key)   
#         principalDf = pd.DataFrame(data = pca_3D[key] .transform(var_scaled.to_numpy()) #pca_3D [i] i de 1 
#              , columns = [f"{key}_PCA_{i}" for i in range(n_pca_variables_3D[key])])
#         testing_df_x_test = pd.concat([testing_df_x_test, principalDf], axis=1)
        
    return testing_df_x_test


def get_test_output(df, path_output, scaler, pca):
    # y_test, rttov_bands = read_data_refl_emiss_rttov(path_rttov_test)
    # ###### flat ###########################
    # name_file = 'refl_emiss_test'
    # df = dataframe_csv(variable = y_test, colum = rttov_bands, path_output = path_output, name_file = name_file)
############################################333    
    df_normalized = preprocessing.normalize(df)
###########################################3
    # test_scaled= scaler.transform(df)
    test_scaled= scaler.transform(df_normalized)

    test_pca = pca.transform(test_scaled)
    
    print(np.shape(test_pca))
    test_df = pd.DataFrame(data = test_pca
             , columns = [f"PCA_{i}" for i in range(np.shape(test_pca)[1])])
    
    print("=================== After scaler and pca output testing Dataframe =======================================")   
    name_file = 'after_scaler_pca_output_statistics_testing'  
    test_df.describe().to_csv("{}/{}.csv".format(path_output, name_file)) 
    
        
    return test_df

def test_random_forest(train_x, train_y, test_x, test_y):
    # rf_model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
    #                   max_depth=None, max_features='auto', max_leaf_nodes=None,
    #                   max_samples=None, min_impurity_decrease=0.0,
    #                   min_impurity_split=None, min_samples_leaf=1,
    #                   min_samples_split=2, min_weight_fraction_leaf=0.0,
    #                   n_estimators=100, n_jobs=None, oob_score=False,
    #                   random_state=None, verbose=0, warm_start=False)
    # rf_model = RandomForestRegressor(bootstrap=False, random_state = 0, max_features='auto', n_estimators=200, min_samples_split= 100, min_samples_leaf=4,max_depth=1,  criterion='mse')
                                     
    rf_model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      #min_impurity_split=None, min_samples_leaf=1,
                      min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)                              
        
# rf_model = RandomForestRegressor( random_state=0, bootstrap=False, max_features='auto', **{'n_estimators': 200, 'min_samples_split': 100, 'min_samples_leaf': 4,  'max_depth': 1})
    
    rf_pcs = rf_model.fit(train_x, train_y)
    

        
    return rf_pcs

def from2to3d(x, lat_ds, lon_ds):
    x_3d = np.zeros((6, len(lat_ds) , len(lon_ds))) # 628,589)) 

    for i in range(6):
        x_3d[i,:,:] = x[:,i].reshape(-1, len(lon_ds))
        
    return x_3d  
  
def plot_target_prediction(target,lat_ds, lon_ds, prediction, path_output, name_plot):
    n_img = len(target)
    fig = plt.figure(figsize=(5 * n_img, 2 * 2 ),facecolor = 'white')  #WxH
    
    fig.suptitle("Comparation between target and prediction", fontsize = 24)

    # x, y = np.meshgrid(lat_ds.values.ravel(), lon_ds.values.ravel())
    x = lon_ds #.values  
    y = lat_ds #.values

    # x, y = np.meshgrid(x, y)
    
    for i in range(n_img):

        # Emulator
        axes = plt.subplot(2,n_img, i + 1)
        ctr = axes.pcolormesh(x, y,prediction[i],cmap = "cividis",shading='auto')
        # ctr = axes.pcolormesh(predicted_input)
        axes.set_title(f"Prediction PC_{i}",fontsize=14)
        # axes.axis('off')

        plt.colorbar(ctr)
        
        # Target
        axes0 = plt.subplot(2,n_img, i + 1 + n_img)
        ctr = axes0.pcolormesh(x, y,target[i],cmap = "cividis",shading='auto')
        # ctr = axes0.pcolormesh(target)
        axes0.set_title(f"Target PC_{i}",fontsize=14)
        # axes0.axis('off')

        plt.colorbar(ctr)
    plt.tight_layout() #para q no queden muchos borde blanco
    
    figure_name = '{}/{}.png'.format(path_output, name_plot) #aca pasarr con todo path
                   
    fig.savefig(figure_name) 
    plt.close() 
    
    
def metric_calculation(x_train, y_train, x_test, y_test, model, name_model):
    gt =  y_train 
    pred = model.predict(x_train)
    print("*************************** Results of the model {} *********************************".format(name_model))

    print("========================= Training metrics ==========================") 
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(gt, pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(gt, pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(gt, pred)))
    mape = np.mean(np.abs((gt - pred) / np.abs(gt)))
    print('Mean Absolute Percentage Error (MAPE): \n', round(mape * 100, 2))
    print('Accuracy: \n', round(100*(1 - mape), 2))

    
    print("========================= Testing metrics ==========================") 
    gt =  y_test 
    pred = model.predict(x_test)
 
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(gt, pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(gt, pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(gt, pred)))
    mape = np.mean(np.abs((gt - pred) / np.abs(gt)))
    print('Mean Absolute Percentage Error (MAPE): \n', round(mape * 100, 2))
    print('Accuracy: \n', round(100*(1 - mape), 2))


    score = model.score(x_train, y_train)
    print('score in training:', score)  
    score = model.score(x_test, y_test)
    print('score in testing:', score)    

    

def permutation_test(X_test, pr_truth, rf_pr):
    #%%time
    pr_result = permutation_importance(
        # rf_pr.model.model, X_test, pr_truth, n_repeats=10, random_state=42, n_jobs=1, scoring=make_scorer(get_rmse_array))
        rf_pr.model.model, X_test, pr_truth, n_repeats=10, random_state=42, n_jobs=1)

    importances = rf_pr.model.model.feature_importances_
    feature_names = list(X_test.columns)
    
    std = np.std([tree.feature_importances_ for tree in rf_pr.model.model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Feature importances")
    fig.tight_layout()
            
    figure_name = '{}/Feature importances.png'.format(path_output) #aca pasarr con todo path
                   
    fig.savefig(figure_name) 
    plt.close() 
    
    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-ICON', type=str, default='/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc', help='path of the dataset is the ICON simulations')
    arg('--k-fold', type=int, default=1, help='fold to be simulated: 1, 2, 3, 4, 5')
    arg('--name-PCA', type=str, default='PCA_1', help='PCA_0, 1, 2, 3, 4, 5')

    arg('--rttov-path-refl-emmis', type = str, default = '/home/b/b381362/output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc', help = 'Path of the dataset with only reflectances 1-19 and 26')
    arg('--rttov-path-rad', type = str, default = '/home/b/b381362/output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc', help = 'Path of the dataset with only radiances')
    
    arg('--rttov-path', type = str, default = "/work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc", help = 'Path of the dataset with only reflectances 1-19 and 26')

   
    
    arg('--path-rttov-test', type = str, default = '/home/b/b381362/output/output-rttov/rttov-131-data-icon-1to36-T09.nc', help = 'Path of the test-dataset ')
    arg('--path-ICON-test', type = str, default = '/work/bb1036/b381362/dataset/data_rttov_T09_dropupbottom_Reff.nc.nc', help = 'Path of the test-dataset ')
    arg('--path-output', type=str, default='/home/b/b381362/output/ML_output', help='path of the output data is')

    args = parser.parse_args()

    path_ICON = args.path_ICON
    path_output=args.path_output

    rttov_path_refl_emmis = args.rttov_path_refl_emmis
    rttov_path_rad = args.rttov_path_rad
    
    k_fold = args.k_fold
    name_PCA = args.name_PCA
    
    rttov_path = args.rttov_path
    # rttov_path_refl_emmis = rttov_path
    # rttov_path_rad = rttov_path

    path_rttov_test = args.path_rttov_test
    path_ICON_test = args.path_ICON_test
    
    # ds=xr.open_dataset(path_ICON)
    # file_name= os.path.splitext(os.path.basename(path_ICON))[0][:-5] 

#######################  3D #############33
    # x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds = get_split_data_xarray(path_ICON, rttov_path_rad, rttov_path_refl_emmis, path_output)
    # x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds = get_split_data_xarray(path_ICON, rttov_path, path_output, k_fold)
    x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds = get_split_data_xarray(path_ICON, path_output, k_fold, rttov_path_rad, rttov_path_refl_emmis)

    #######################  end 3D #############33

    # x_train_2D_all, x_train_3D_all,  df_x_train, df_x_test, df_y_train, df_y_test = get_split_data_xarray_alldata(path_ICON, rttov_path_rad, rttov_path_refl_emmis, path_output)

    #######################  scaler all 3D variables #############
    # training_df_x_train, scaler_x, pca_3D, n_pca_variables_3D = scaler_PCA_input_all_3D(df_x_train, path_output)
    ####################### end scaler all #############

    ### PCA of each input and scaler
    training_df_x_train, scaler_x, pca_3D, n_pca_variables_3D = scaler_PCA_input(df_x_train, path_output)

     #######################  scaler all 3D variables in test #############
    # testing_df_x_test = get_test_input_PCA_all_3D(path_output, df_x_test, scaler_x, pca_3D, n_pca_variables_3D)

    ####################### end scaler all 3D variables in test #############    
#     ##### scale and PCA each input Test 
    testing_df_x_test = get_test_input(path_output, df_x_test, scaler_x, pca_3D, n_pca_variables_3D)
    
    ### PCA of the output and scaler
    scaler_y, training_df_y_train, pca_y = PCA_read_input_target(df_y_train, path_output)

    
    ### scale and PCA Test output

    testing_df_y_test = get_test_output(df_y_test, path_output, scaler_y, pca_y)
    
    print("columns train", training_df_y_train.columns)
    print("columns test", testing_df_y_test.columns)

    x_train = training_df_x_train
    y_train = training_df_y_train[name_PCA] #['PCA_1']

    x_test = testing_df_x_test
    y_test = testing_df_y_test[name_PCA] #['PCA_1']
    
#===================================random forest ===================================
    rf_pcs = test_random_forest(train_x = x_train,
                                          train_y = y_train, 
                                          test_x = x_test, 
                                          test_y = y_test)
    
    metric_calculation(x_train, y_train, x_test, y_test, model = rf_pcs, name_model = "Random_forest")
  
# #=================================== MLPRegressor ===================================

#     clf = MLPRegressor(solver='lbfgs', 
#                    alpha=1e-5,     # used for regularization, ovoiding overfitting by penalizing large magnitudes
#                    hidden_layer_sizes=(5, 2), random_state=24)
#     clf.fit(x_train, y_train)
#     # res = clf.predict(train_data)
#     metric_calculation(x_train, y_train, x_test, y_test, model = clf, name_model = "MLRegressor")
#     model = clf

# #=================================== Gaussian Process ===================================

#     kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
#     gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#     gaussian_process.fit(x_train, y_train)
#     metric_calculation(x_train, y_train, x_test, y_test, model = clf, name_model = "GP")

# #===================================  ===================================


#     print("#############pickle #########################")
#     # save the model to disk
#     filename = 'finalized_model.sav'
#     pickle.dump(rf_pcs, open(filename, 'wb'))
#     # some time later...
#     # load the model from disk
#     loaded_model = pickle.load(open(filename, 'rb'))
#     #############pickle #########################
#     score = loaded_model.score(x_train, y_train)
#     print('score in training:', score)  
#     score = loaded_model.score(x_test, y_test)
#     print('score in testing:', score)

    
#     print("#############joblib #########################")
#     joblib.dump(rf_pcs, "./random_forest.joblib")
#     loaded_rf = joblib.load("./random_forest.joblib")
#     score = loaded_rf.score(x_train, y_train)
#     print('score in training:', score)  
#     score = loaded_rf.score(x_test, y_test)
#     print('score in testing:', score)

    #####
    # score = rf_pcs.score(train_x_df, train_y_df)
    # print('score in training:', score)  
    # score = rf_pcs.score(test_x_df, test_y)
    # print('score in testing:', score)
    
    # pred_pcs_train = rf_pcs.predict(train_x_df)
    # predicted_3D_train = from2to3d(pred_pcs_train)

    
    
    # xr_output = xr.Dataset(dict(refl_emis = m_out_pc1))
    # # save output to netcdf 
    # xr_output.to_netcdf(path_output/"output_predicted.nc",'w')

    
    # train_y_3D = from2to3d(train_y_df.to_numpy())
    # plot_target_prediction(target = train_y_3D, prediction = predicted_3D_train, path_output = path_output, name_plot = 'target_pred_training')
    

#     ######################  3D #############33
     
    # test_pred_pcs = model.predict(testing_df_x_test)
    # test_predicted_3D = from2to3d(test_pred_pcs,lat_test_ds, lon_test_ds)
    # test_target_3D = from2to3d(testing_df_y_test, lat_test_ds, lon_test_ds)
    # plot_target_prediction(test_target_3D, lat_test_ds, lon_test_ds, test_predicted_3D, path_output, name_plot = 'target_pred_testing')

# ######################  end 3D #############33

    # permutation_test(x = train_x_df, y = train_y_df, model = rf_pcs)




#     sys.stdout.close()
if __name__ == '__main__':
    main()