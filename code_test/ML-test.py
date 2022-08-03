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


def read_data_refl_emiss_rttov(rttov_path): #rttov_path_rad, rttov_path_refl_emmis):
 
    rttov_ds = xr.open_dataset(rttov_path).compute()        
    rttov_variable = np.zeros((np.shape(rttov_ds['Radiance_total'].values)))

    rttov_variable[:19] = rttov_ds['BRF_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
    rttov_variable[19:25] = rttov_ds['Radiance_total'][19:25]
    rttov_variable[25] = rttov_ds['BRF_total'][25] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    rttov_variable[26:36] = rttov_ds['Radiance_total'][26:36]
    
    print("===================================== Training output ====================================== ")
    print("****************variables shape training_output", np.shape(rttov_variable))
    
    rttov_ds.close()
    rttov_variable_ds = xr.DataArray( rttov_variable, dims=['channel','lat','lon'], coords= [rttov_ds.Channel.data, rttov_ds.Latitude.data, rttov_ds.Longitude.data])      

    return  rttov_variable_ds  
   
 
def get_split_data_xarray(path_ICON, rttov_path, path_output, k_fold): #rttov_path_rad, rttov_path_refl_emmis
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
    rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path)

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
    x_train_2D = { "Nd_max": np.log(ds_x_train.Nd_max.values+ 1.0e-16), "lwp": np.log(ds_x_train.lwp.values+ 1.0e-16)}  
    x_train_3D = { "pres": ds_x_train.pres.values, "ta": ds_x_train.ta.values, "hus": ds_x_train.hus.values}
    

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


    
def get_data_xarray(path_ICON, rttov_path, path_output): 
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
    rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path)
     
    lat_ds = rttov_variable_ds.lat
    lon_ds = rttov_variable_ds.lon
    
        
    # df_x_train.drop(df_x_train[(df_x_train['Nd_max'] <400) & (df_x_train['lwp'] < 600)].index, inplace=True)
   
    #=============================Dataframe X  ====================================    
    x_2D = { "Nd_max": np.log(ds.Nd_max.values+ 1.0e-16), "lwp": np.log(ds.lwp.values+ 1.0e-16)}  
    x_3D = { "pres": ds.pres.values, "ta": ds.ta.values, "hus": ds.hus.values}
    
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
    variables_2D = { "Nd_max", "lwp"}  
    n_pca_variables_3D = { "pres": 5, "ta": 10, "hus":24} #(height, lat, lon)
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
    df_y_train_normalized = preprocessing.normalize(df_y_train)
###########################################3

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
    x_3d = np.zeros((6, len(lat_ds) , len(lon_ds))) # 628,589)) 

    for i in range(6):
        x_3d[i,:,:] = x[:,i].reshape(-1, len(lon_ds))
        
    return x_3d  
  
def plot_target_prediction(target,lat_ds, lon_ds, prediction, path_output, name_plot):
    n_img = len(target)
    fig = plt.figure(figsize=(5 * n_img, 2 * 2 ),facecolor = 'white')  #WxH
    
    fig.suptitle("Comparation between target and prediction", fontsize = 24)

    x = lon_ds #.values  
    y = lat_ds #.values
    
    for i in range(n_img):

        # Emulator
        axes = plt.subplot(2,n_img, i + 1)
        ctr = axes.pcolormesh(x, y,prediction[i],cmap = "cividis",shading='auto')
        axes.set_title(f"Prediction PC_{i}",fontsize=14)
        plt.colorbar(ctr)
        
        # Target
        axes0 = plt.subplot(2,n_img, i + 1 + n_img)
        ctr = axes0.pcolormesh(x, y,target[i],cmap = "cividis",shading='auto')
        axes0.set_title(f"Target PC_{i}",fontsize=14)

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
    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-ICON', type=str, default='/home/jvillarreal/Documents/phd/dataset/data_rttov_T12.nc', help='path of the dataset is the ICON simulations')
    arg('--k-fold', type=int, default=1, help='fold to be simulated: 1, 2, 3, 4, 5')    
    arg('--rttov-path', type = str, default = "/work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc", help = 'Path of the dataset with only reflectances 1-19 and 26')   
    
    arg('--path-rttov-test', type = str, default = '/home/jvillarreal/Documents/phd/output/output-rttov/rttov-131-data-icon-1to36-T09.nc', help = 'Path of the test-dataset ')
    arg('--path-ICON-test', type = str, default = '/home/jvillarreal/Documents/phd/dataset/data_rttov_T09.nc', help = 'Path of the test-dataset ')

    arg('--path-output', type=str, default='/home/jvillarreal/Documents/phd/output/ML_output', help='path of the output data is')

    args = parser.parse_args()

    path_ICON = args.path_ICON
    path_output=args.path_output

    k_fold = args.k_fold
    rttov_path = args.rttov_path


    path_rttov_test = args.path_rttov_test
    path_ICON_test = args.path_ICON_test
    
    # ds=xr.open_dataset(path_ICON)
    # file_name= os.path.splitext(os.path.basename(path_ICON))[0][:-5] 

    #######################  3D #############
    x_train_2D, x_train_3D, x_test_2D, x_test_3D, y_train, y_test,  df_x_train, df_x_test, df_y_train, df_y_test, lat_test_ds, lon_test_ds = get_split_data_xarray(path_ICON, rttov_path, path_output, k_fold)
    
    #######################  testing #############
    df_x_img, df_y_img, lat_ds_img, lon_ds_img = get_data_xarray(path_ICON, rttov_path, path_output)
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
    y_train = training_df_y_train #['PCA_1']

    x_test = testing_df_x_test
    y_test = testing_df_y_test #['PCA_1']
    
#===================================random forest ===================================
#     rf_pcs = test_random_forest(train_x = x_train,
#                                           train_y = y_train, 
#                                           test_x = x_test, 
#                                           test_y = y_test)
    
#     metric_calculation(x_train, y_train, x_test, y_test, model = rf_pcs, name_model = "Random_forest")
  
# #=================================== MLPRegressor ===================================

    clf = MLPRegressor(solver='lbfgs', 
                   alpha=1e-5,     # used for regularization, ovoiding overfitting by penalizing large magnitudes
                   hidden_layer_sizes=(5, 2), random_state=24)
    clf.fit(x_train, y_train)
    # res = clf.predict(train_data)
    metric_calculation(x_train, y_train, x_test, y_test, model = clf, name_model = "MLRegressor")
    model = clf

    print("$$$$$$$$$$$$$$$$$ testing model in the timestep T09 $$$$$$$$$$$$$$$$$")
    metric_calculation(x_train, y_train, df_x_img, df_y_img, model = clf, name_model = "MLRegressor")

  

    pred_pcs_img = model.predict(df_x_img)
    xr_output = xr.Dataset(dict(refl_emis = pred_pcs_img))
    # save output to netcdf 
    xr_output.to_netcdf(path_output/k_fold +"_output_predicted.nc",'w')

    predicted_3D_img = from2to3d(pred_pcs_img)
    
    target_3D_img = from2to3d(df_y_img.to_numpy())
    plot_target_prediction(target = target_3D_img, prediction = predicted_3D_img, path_output = path_output, name_plot = k_fold +'target_pred_training_ML')
    
    
        
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