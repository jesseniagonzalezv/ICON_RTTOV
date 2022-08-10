import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import argparse
import pandas as pd
from eofs.xarray import Eof
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns

from lwp_nd import lwp_nd_input_ICON
from PCA_radiances import PCA_calculation, variance_sklearn_plot, dataframe_csv #, convert_3D

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, check_scoring, mean_squared_error

from sklearn import metrics

from skimage.metrics import structural_similarity as ssim

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os

import joblib
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

def read_data_refl_emiss_rttov( rttov_path_rad, rttov_path_refl_emmis): #rttov_path
  
    #input: path of the radiances and reflectances
    #output: refl_emmiss CHxHxW  
    
    rttov_ds_rad = xr.open_dataset(rttov_path_rad).compute()  # write read rttov in a function
    rttov_ds_refl_emmi = xr.open_dataset(rttov_path_refl_emmis).compute()  
    
    
    # rttov_ds = xr.open_dataset(rttov_path).compute()        
    # rttov_variable = np.zeros((np.shape(rttov_ds['Radiance_total'].values)))
    rttov_variable = np.zeros((np.shape(rttov_ds_rad['Y'].values)))

    rttov_variable[:19] = rttov_ds_refl_emmi['bt_refl_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
    rttov_variable[19:25] = rttov_ds_rad['Y'][19:25]
    rttov_variable[25] = rttov_ds_refl_emmi['bt_refl_total'][19] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    rttov_variable[26:36] = rttov_ds_rad['Y'][26:36]

#     rttov_variable[:19] = rttov_ds['BRF_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
#     rttov_variable[19:25] = rttov_ds['Radiance_total'][19:25]
#     rttov_variable[25] = rttov_ds['BRF_total'][25] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
#     rttov_variable[26:36] = rttov_ds['Radiance_total'][26:36]
    
    print("===================================== Training output ====================================== ")
    print("****************variables shape training_output", np.shape(rttov_variable))
    
    # rttov_ds.close()
    # rttov_variable_ds = xr.DataArray( rttov_variable, dims=['channel','lat','lon'], coords= [rttov_ds.Channel.data, rttov_ds.Latitude.data, rttov_ds.Longitude.data])      

    rttov_ds_rad.close()
    rttov_ds_refl_emmi.close()
    rttov_variable_ds = xr.DataArray( rttov_variable[:,9:,], dims=['channel','lat','lon'], coords= [rttov_ds_rad.chan.data, rttov_ds_refl_emmi.lat.data[9:], rttov_ds_refl_emmi.lon.data ])

    
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
    y_train, y_test (channel: 36, Latitude, Longitude)  = radiances/reflectances 
    df_x_train, df_x_test: dataframes of the ICON-LES inputs
                           dataframe 2D variables (row: lat x lon(H*W))
                           dataframe 3D variables (row: lat x lon(H*W), columns(height))
    df_y_train, df_y_test: dataframes of the reflectances/radiances
                           dataframe variables (row: Latitude x Longitude(H*W), columns(channel))

    '''
    ds = xr.open_dataset(path_ICON).compute()
    # rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path)
    rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path_rad, rttov_path_refl_emmis)

    #================================= X ========================================
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
 
    
    lat_test_ds = y_test.lat
    lon_test_ds = y_test.lon

    
        
    # df_x_train.drop(df_x_train[(df_x_train['Nd_max'] <400) & (df_x_train['lwp'] < 600)].index, inplace=True)
   
    #=============================Dataframe X  ====================================    
    # x_train_2D = { "Nd_max": np.log(ds_x_train.Nd_max.values+ 1.0e-16), "lwp": np.log(ds_x_train.lwp.values+ 1.0e-16)}  
    # x_train_3D = { "pres": ds_x_train.pres.values, "ta": ds_x_train.ta.values, "hus": ds_x_train.hus.values}
 
    x_train_2D = { "Nd_max": np.log(ds_x_train.Nd_max.values+ 1.0e-16), "lwp": np.log(ds_x_train.lwp.values+ 1.0e-16), "v_10m": ds_x_train.v_10m.values}  
    x_train_3D = { "clc": ds_x_train.cli.values, "cli": ds_x_train.clc.values, "clw": ds_x_train.clw.values, "hus": ds_x_train.hus.values}
    

    # x_test_2D = { "Nd_max": np.log(ds_x_test.Nd_max.values+ 1.0e-16), "lwp": np.log(ds_x_test.lwp.values+ 1.0e-16)}  
    # x_test_3D = { "pres": ds_x_test.pres.values, "ta": ds_x_test.ta.values, "hus": ds_x_test.hus.values}
    x_test_2D = { "Nd_max": np.log(ds_x_test.Nd_max.values+ 1.0e-16), "lwp": np.log(ds_x_test.lwp.values+ 1.0e-16), "v_10m": ds_x_test.v_10m.values}  
    x_test_3D = { "clc": ds_x_test.cli.values, "cli": ds_x_test.clc.values, "clw": ds_x_test.clw.values, "hus": ds_x_test.hus.values}

    
    
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
    
#     print("shape before checking col/row 0, train, test", df_x_train.shape, df_x_test.shape)

#     df_x_train = df_x_train.loc[:, (df_x_train != 0).any(axis=0)]
#     df_x_test = df_x_test.loc[:, (df_x_test != 0).any(axis=0)]
#     print("shape after checking col/row 0, train, test", df_x_train.shape, df_x_test.shape)


    
    return x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds


    
def get_data_xarray(path_ICON, rttov_path_rad, rttov_path_refl_emmis, path_output):  #rttov_path
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
    df_x: dataframes of the ICON-LES inputs
                           dataframe 2D variables (row: lat x lon(H*W))
                           dataframe 3D variables (row: lat x lon(H*W), columns(height))
    df_y: dataframes of the reflectances/radiances
                           dataframe variables (row: Latitude x Longitude(H*W), columns(channel))

    '''
    ds = xr.open_dataset(path_ICON).compute()
    # rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path)
    rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path_rad, rttov_path_refl_emmis)

     
    lat_ds = rttov_variable_ds.lat
    lon_ds = rttov_variable_ds.lon
    
        
    # df_x_train.drop(df_x_train[(df_x_train['Nd_max'] <400) & (df_x_train['lwp'] < 600)].index, inplace=True)
   
    #=============================Dataframe X  ====================================    
    # x_2D = { "Nd_max": np.log(ds.Nd_max.values+ 1.0e-16), "lwp": np.log(ds.lwp.values+ 1.0e-16)}     
    x_2D = { "Nd_max": np.log(ds.Nd_max.values+ 1.0e-16), "lwp": np.log(ds.lwp.values+ 1.0e-16), "v_10m": ds.v_10m.values}  
 
    # x_3D = { "pres": ds.pres.values, "ta": ds.ta.values, "hus": ds.hus.values}
    x_3D = { "clc": ds.cli.values, "cli": ds.clc.values, "clw": ds.clw.values, "hus": ds.hus.values}
    
    ds.close()
    #=============================Dataframe X  ====================================    
    df_x = pd.DataFrame()
    df_x = pd.DataFrame()
    
    for key, var in x_2D.items():  
        df = pd.DataFrame(data = var.flatten()
             , columns =  [key])
        df_x = pd.concat([df_x, df], axis=1)   
    for key, var in x_3D.items():  
        var = var.transpose(1,2,0) #lat,lot,heigh
        var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
        var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
        df_x=pd.concat([df_x, var_df], axis=1)
    
    print("=================== Dataframe training =======================================")    
    name_file = 'inputs_statistics_testing_img'  
    df_x.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    
        
    #================================= Y ========================================
    ###############################Dataframe Y_data##################################
    name_file = 'refl_emiss_statistics_testing_img'
    df_y = dataframe_csv(variable = rttov_variable_ds.transpose('lat', 'lon', 'channel').values, 
                  column = rttov_variable_ds.channel.values,  
                  path_output = path_output, 
                  name_file = name_file)
    rttov_variable_ds.close()
           
    return df_x, df_y, lat_ds, lon_ds


  
def scaler_PCA_input(df_x_train, path_output):
    '''
    input: dataframe input with 2D and 3D
    output: x_train_df dataframe with the variable 2D and 3D PCAs
    '''
    # variables_2D = { "Nd_max", "lwp"}  
    # n_pca_variables_3D = { "pres": 5, "ta": 10, "hus":24} #(height, lat, lon)
    variables_2D = { "Nd_max", "lwp", "v_10m"}  
    n_pca_variables_3D = { "clc": 22, "cli": 31, "clw": 27, "hus":15} #24(height, lat, lon)
    # n_pca_variables_3D = { "clc": 5, "cli": 5, "clw": 5, "hus":5} #24(height, lat, lon)
    
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


def PCA_read_input_target(df_y_train, path_output):
    '''
    PC_output (latxlon, number_pcs)
    '''
###########################################
    # df_y_train_normalized = preprocessing.normalize(df_y_train)
###########################################3

    scaler_y = preprocessing.StandardScaler().fit(df_y_train)  #Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler_y.transform(df_y_train)

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
    
    variables_2D = { "Nd_max", "lwp", "v_10m"}  
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


def get_test_output(df, path_output, scaler, pca):

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
                                     
    rf_model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      #min_impurity_split=None, min_samples_leaf=1,
                      min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)                                      
    rf_pcs = rf_model.fit(train_x, train_y)
    
    return rf_pcs

def from2to3d(x, lat_ds, lon_ds):
    print(np.shape(x))
    columns = np.shape(x)[1]
    x_3d = np.zeros((columns, len(lat_ds) , len(lon_ds))) # 628,589)) 

    for i in range(columns):
        x_3d[i,:,:] = x[:,i].reshape(-1, len(lon_ds))
        
    return x_3d  
 
def from1to2d(x, lat_ds, lon_ds):
    print(np.shape(x))

    x_2d = np.zeros((len(lat_ds) , len(lon_ds))) # 628,589)) 

    x_2d[:,:] = x.reshape(len(lat_ds) , len(lon_ds))
        
            
    return x_2d 

# def plot_target_prediction(target,lat_ds, lon_ds, prediction, path_output, name_plot):
#     n_img = len(target)
#     fig = plt.figure(figsize=(5 * n_img, 2 * 2 ),facecolor = 'white')  #WxH
    
#     fig.suptitle("Comparation between target and prediction", fontsize = 24)

#     x = lon_ds #.values  
#     y = lat_ds #.values
    
#     for i in range(n_img):

#         axes = plt.subplot(2,n_img, i + 1)
#         # ctr = axes.pcolormesh(x, y,prediction[i],cmap = "cividis",shading='auto')
#         ctr = axes.pcolormesh(x, y,prediction,cmap = "cividis",shading='auto')

#         axes.set_title(f"Prediction PC_{i}",fontsize=14)
#         plt.colorbar(ctr)
        
#         # Target
#         axes0 = plt.subplot(2,n_img, i + 1 + n_img)
#         # ctr = axes0.pcolormesh(x, y,target[i],cmap = "cividis",shading='auto')
#         ctr = axes0.pcolormesh(x, y,target,cm
#         # Emulatorap = "cividis",shading='auto')
#         axes0.set_title(f"Target PC_{i}",fontsize=14)

#         plt.colorbar(ctr)
#     plt.tight_layout() #para q no queden muchos borde blanco

def plot_target_prediction(ds_out, path_output, name_plot):

    # n_channels = len(output_ds)
    # n_channels
    f, axes = plt.subplots(1, 2, figsize=(4*2, 1*4*1.066)) #width of 15 inches and 7 inches in height.
    f.subplots_adjust(wspace=0.4, hspace=0.4)

    axli = axes.ravel()

    # for i in n_channels:
    ds_out['target'].plot(ax = axli[0] )#, vmin=0, vmax=0.12, cmap='jet')
    ds_out['prediction'].plot(ax = axli[1])#, vmin=0, vmax=0.12, cmap='jet')# cmap='cividis', 

    plt.suptitle("ssim: " + str(ssim(ds_out['target'], ds_out['prediction'])) + " rmse: " + str(mean_squared_error(ds_out['target'], ds_out['prediction'])))

    figure_name = '{}/{}.png'.format(path_output, name_plot) #aca pasarr con todo path
     
    f.tight_layout()

    f.savefig(figure_name) 
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
    score = model.score(x_train, y_train)
    print('score in training:', score)
    
    print("========================= Testing metrics ==========================") 
    gt =  y_test 
    pred = model.predict(x_test)
 
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(gt, pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(gt, pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(gt, pred)))
    mape = np.mean(np.abs((gt - pred) / np.abs(gt)))
    print('Mean Absolute Percentage Error (MAPE): \n', round(mape * 100, 2))
    print('Accuracy: \n', round(100*(1 - mape), 2))


  
    score = model.score(x_test, y_test)
    print('score in testing:', score)    
    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-ICON', type=str, default='/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc', help='path of the dataset is the ICON simulations')
    arg('--k-fold', type=int, default=2, help='fold to be simulated: 1, 2, 3, 4, 5')  
    arg('--name-PCA', type=str, default='PCA_0', help='PCA_0, 1, 2, 3, 4, 5')

        
    arg('--rttov-path', type = str, default = "/work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc", help = 'Path of the dataset with only reflectances 1-19 and 26')   
    
    arg('--rttov-path-refl-emmis', type = str, default = '/home/b/b381362/output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc', help = 'Path of the dataset with only reflectances 1-19 and 26')
    arg('--rttov-path-rad', type = str, default = '/home/b/b381362/output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc', help = 'Path of the dataset with only radiances')
    
    
    arg('--path-rttov-test', type = str, default = '/home/b/b381362/output/output-rttov/rttov-131-data-icon-1to36-T09.nc', help = 'Path of the test-dataset ')
    arg('--path-ICON-test', type = str, default = '/work/bb1036/b381362/dataset/data_rttov_T09_dropupbottom_Reff.nc.nc', help = 'Path of the test-dataset ')

    arg('--path-output', type=str, default='/home/b/b381362/output/ML_output', help='path of the output data is')

    args = parser.parse_args()

    path_ICON = args.path_ICON
    path_output=args.path_output

    k_fold = args.k_fold
    name_PCA = args.name_PCA

        
    rttov_path = args.rttov_path
    rttov_path_refl_emmis = args.rttov_path_refl_emmis
    rttov_path_rad = args.rttov_path_rad

    path_rttov_test = args.path_rttov_test
    path_ICON_test = args.path_ICON_test
    
    # ds=xr.open_dataset(path_ICON)
    # file_name= os.path.splitext(os.path.basename(path_ICON))[0][:-5] 

    #######################  3D #############
    # x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds = get_split_data_xarray(path_ICON, rttov_path, path_output, k_fold)
    
    x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds = get_split_data_xarray(path_ICON, path_output, k_fold, rttov_path_rad, rttov_path_refl_emmis)

    #######################  testing #############
    # df_x_img, df_y_img, lat_ds_img, lon_ds_img = get_data_xarray(path_ICON, rttov_path, path_output)
    df_x_img, df_y_img, lat_ds_img, lon_ds_img = get_data_xarray(path_ICON, rttov_path_rad, rttov_path_refl_emmis, path_output)

    #######################  endtesting #############

    #######################  end 3D #############
    
    ### PCA of each input and scaler
    training_df_x_train, scaler_x, pca_3D, n_pca_variables_3D = scaler_PCA_input(df_x_train, path_output)

        
    ##### scale and PCA each input Test 
    testing_df_x_test = get_test_input(path_output, df_x_test, scaler_x, pca_3D, n_pca_variables_3D)
    
    #######################  testing #############
    df_x_img = get_test_input(path_output, df_x_img, scaler_x, pca_3D, n_pca_variables_3D)
    #######################  endtesting #############
    
    ### PCA of the output and scaler
    scaler_y, training_df_y_train, pca_y = PCA_read_input_target(df_y_train, path_output)

    
    ### scale and PCA Test output
    testing_df_y_test = get_test_output(df_y_test, path_output, scaler_y, pca_y)
    
    #######################  testing #############
    df_y_img = get_test_output(df_y_img, path_output, scaler_y, pca_y)
    #######################  endtesting #############
    
    
    print("columns train", training_df_y_train.columns)
    print("columns test", testing_df_y_test.columns)

    x_train = training_df_x_train
    y_train = training_df_y_train[name_PCA] #['PCA_0']

    x_test = testing_df_x_test
    y_test = testing_df_y_test[name_PCA] #['PCA_0']
    
#===================================random forest ===================================
#     rf_pcs = test_random_forest(train_x = x_train,
#                                           train_y = y_train, 
#                                           test_x = x_test, 
#                                           test_y = y_test)
    
#     metric_calculation(x_train, y_train, x_test, y_test, model = rf_pcs, name_model = "Random_forest")
#     model = rf_pcs
    
#         # view the feature scores

#     print("$$$$$$$$$$$$$$$$$ testing model in the timestep T09 $$$$$$$$$$$$$$$$$")
#     metric_calculation(x_train, y_train, df_x_img, df_y_img[name_PCA], model = model, name_model = "RF_T09")

    
# #=================================== MLPRegressor ===================================

    # clf = MLPRegressor(solver='lbfgs', 
    #                alpha=1e-5,     # used for regularization, ovoiding overfitting by penalizing large magnitudes
    #                hidden_layer_sizes=(5, 2), random_state=24)
    # clf.fit(x_train, y_train)
    # # res = clf.predict(train_data)
    # metric_calculation(x_train, y_train, x_test, y_test, model = clf, name_model = "MLRegressor")
    # model = clf
    
    
    # #=================================== MLPRegressor ===================================
    mlp_reg = MLPRegressor(hidden_layer_sizes=(150,100,50),
                       max_iter = 50,activation = 'relu',
                       solver = 'adam')
    mlp_reg.fit(x_train, y_train)
    metric_calculation(x_train, y_train, x_test, y_test, model = mlp_reg, name_model = "MLRegressor_2")
    model = mlp_reg

#     param_grid = {
#     'hidden_layer_sizes': [(250,100,50), (150,80,40), (100,50,30)],
#     'max_iter': [50, 150],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }
    
#     grid = GridSearchCV(model, param_grid, n_jobs= -1, cv=5)
#     grid.fit(x_train, y_train)

#     print("=========== best parameters $$$$$$$$$$$$$$$$$")

#     print(grid.best_params_) 
    
    print("$$$$$$$$$$$$$$$$$ testing model in the timestep T09 $$$$$$$$$$$$$$$$$")
    metric_calculation(x_train, y_train, df_x_img, df_y_img[name_PCA], model = model, name_model = "MLRegressor_T09")

      # #===================================================================================


    pred_pcs_img = model.predict(df_x_img)
       

    # save output to netcdf 

    # predicted_3D_img = from2to3d(pred_pcs_img, lat_ds_img, lon_ds_img)
    # target_3D_img = from2to3d(df_y_img.to_numpy())

    # plot_target_prediction(target = target_3D_img, prediction = predicted_3D_img, path_output = path_output, name_plot = k_fold +'target_pred_img_ML')


            
    predicted_2D_img = from1to2d(pred_pcs_img, lat_ds_img, lon_ds_img)
    target_2D_img = from1to2d(df_y_img[name_PCA].to_numpy(), lat_ds_img, lon_ds_img)




    

    
    print(np.shape(predicted_2D_img), np.shape(target_2D_img), np.shape(lat_ds_img), np.shape(lon_ds_img))
  

    
#     m_pred = xr.DataArray(predicted_3D_img, dims=['channel','lat','lon'], coords= [range(1,np.shape(predicted_3D_img)[0]+1),lat_ds_img, lon_ds_img])
#     xr_output.to_netcdf(path_output/"output_predicted.nc",'w')
    
#     m_target = xr.DataArray(target_3D_img, dims=['channel','lat','lon'], coords= [range(1,np.shape(target_3D_img)[0]+1), lat_ds_img, lon_ds_img])
    
    m_pred = xr.DataArray(predicted_2D_img, dims=['lat','lon'], coords= [lat_ds_img, lon_ds_img])
    m_target = xr.DataArray(target_2D_img, dims=['lat','lon'], coords= [lat_ds_img, lon_ds_img])
    
    xr_output = xr.Dataset(dict(prediction = m_pred, target = m_target))
    xr_output.to_netcdf(path_output +"/output_pred_target.nc",'w')
    
    
    # plot_target_prediction(target = target_2D_img, lat_ds = lat_ds_img, lon_ds = lon_ds_img, prediction = predicted_2D_img, path_output = path_output, name_plot = 'target_pred_img_ML_k_fold_' + str(k_fold))
    plot_target_prediction(xr_output, path_output = path_output, name_plot = 'target_pred_img_ML_k_fold_' + str(k_fold))


#     ##===================================
#     feature_scores = pd.Series(model.feature_importances_, index = x_train.columns).sort_values(ascending=False)

#     print(feature_scores)

#     f, ax = plt.subplots(figsize=(30, 24))
#     ax = sns.barplot(x=feature_scores, y=feature_scores.index, data = x_train)
#     ax.set_title("Visualize feature scores of the features")
#     ax.set_yticklabels(feature_scores.index)
#     ax.set_xlabel("Feature importance score")
#     ax.set_ylabel("Features")
#     plt.show()
#     figure_name = '{}/feature importances.png'.format(path_output) #aca pasarr con todo path
     
#     f.tight_layout()

#     f.savefig(figure_name) 
##===================================


    xr_output.close()

        
##=================================== Gaussian Process ===================================

#     kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
#     gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#     gaussian_process.fit(x_train, y_train)
#     metric_calculation(x_train, y_train, x_test, y_test, model = clf, name_model = "GP")

##======================================================================
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

##======================================================================

    # permutation_test(x = train_x_df, y = train_y_df, model = rf_pcs)




#     sys.stdout.close()
if __name__ == '__main__':
    main()