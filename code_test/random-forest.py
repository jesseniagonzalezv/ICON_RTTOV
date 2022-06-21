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


def get_rmse_array(truth, pred):
    print(truth.shape, pred.shape)
    weights = np.cos(np.deg2rad(truth.lat))

    return np.sqrt(((truth - pred.reshape(-1, 96,144))**2).weighted(weights).mean(['lat', 'lon'])).data.mean()


def get_training_inputs(path_output, path_ICON):
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
    
    train_inputs_variables_2D = ['Nd_max', 'lwp'] #(lat, lon)
    train_inputs_variables_3D = { "pres": p_2013, "ta": T_2013, "hus":q_2013} #(height, lat, lon)
    n_pca_variables_3D = { "pres": 5, "ta": 10, "hus":24} #(height, lat, lon)
    ############ variables 2D ######################################
    df = pd.DataFrame({  #falta normalizar
    "Nd_max" : ds.Nd_max.values.flatten(),
    "LWP" : ds["lwp"].values.flatten() }) 

    print("============== 2D variables statistics ==================== ")
    print("LWP shape: {}".format(np.shape(ds["lwp"].values)))
    print(df.describe())#if i mask i will have nan instead of Nd<2 and LWP<o
    print("LWP min: {}, max: {}".format(ds["lwp"].values.min(), np.max(ds["lwp"].values)))
    print("Nd_max min: {}, max: {}".format(ds["Nd_max"].values.min(), np.max(ds["Nd_max"].values)))
    count_nan_in_df = pd.DataFrame(df).isnull().sum()
    print ("=========print NaNs: \n", count_nan_in_df)  #can i used 0 in the lwp and nd?
    # df = df.fillna(0) #there is not nan

    print("============== 3D variables statistics ==================== ")
    print("pres min: {}, max: {}, mean: {}".format(np.min(p_2013), np.max(p_2013), np.mean(p_2013)))
    print("ta min: {}, max: {}, mean: {}".format(ds["ta"].values.min(), np.max(ds["ta"].values),ds["ta"].values.mean()))
    print("hus min: {}, max: {}, mean: {}".format(ds["hus"].values.min(), np.max(ds["hus"].values), ds["hus"].values.mean()))

        
        
    scaler_2D = preprocessing.StandardScaler()# Fit on training set only.
    scaler_2D.fit(df)

    training_input_variables_df = pd.DataFrame(scaler_2D.transform(df), columns = ["Nd_max", "lwp"]) 

    # Get PC of the variables 3D
    scaler_3D = []
    pca_3D = []

    for key, var in train_inputs_variables_3D.items():  
        var = var.transpose(1,2,0) #lat,lot,heigh
        var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
        var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
        # scaler = preprocessing.StandardScaler().fit(var_df) 
        scaler = preprocessing.StandardScaler()# Fit on training set only. 
        scaler_3D.append(scaler)
        
        scaler.fit(var_df)
        var_scaled = scaler.transform(var_df)
        
        
        n_pca = n_pca_variables_3D[key] #150
        name_plot= "{}_PCA_variance_{}_variable".format(n_pca, key)
        
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
    training_input_variables_df.describe().to_csv(path_output + "/inputs_description.csv")    
    # count_nan 
    count_nan_in_df = training_input_variables_df.isnull().sum()
    print (count_nan_in_df)  #can i used 0 in the lwp and nd?
    
    #X_train_np = np.array([X_train_reshaped['pres'].data, X_train_reshaped['ta'].data, X_train_reshaped['hus'].data, X_train_reshaped['qnc'].data])

    # pd.set_option('display.float_format', lambda x: '%.1f' % x)
    # df.columns= colum
    

    return training_input_variables_df, scaler_2D, scaler_3D, pca_3D, n_pca_variables_3D


def from2to3d(x):
    x_3d = np.zeros((6, 628,589)) 

    for i in range(6):
        x_3d[i,:,:] = x[:,i].reshape(-1,589)
        
    return x_3d
  

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
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)                              
        
# rf_model = RandomForestRegressor( random_state=0, bootstrap=False, max_features='auto', **{'n_estimators': 200, 'min_samples_split': 100, 'min_samples_leaf': 4,  'max_depth': 1})
    
    rf_pcs = rf_model.fit(train_x, train_y)
    

        
    return rf_pcs


def read_data_refl_emiss_rttov_old(rttov_path_rad, rttov_path_refl_emmis):
    '''
    
    output: refl_emmiss (HxWxCH)
    '''
    rttov_ds_rad = xr.open_dataset(rttov_path_rad).compute()  # write read rttov in a function
    rttov_ds_refl_emmi = xr.open_dataset(rttov_path_refl_emmis).compute()
    rttov_variable = np.zeros((np.shape(rttov_ds_rad['Y'].values)))
    
    
    print("****************variables shape ", np.shape(rttov_ds_refl_emmi['bt_refl_total'].values), np.shape(rttov_variable), np.shape(rttov_ds_rad['Y'].values))

    rttov_variable[:19] = rttov_ds_refl_emmi['bt_refl_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
    rttov_variable[19:25] = rttov_ds_rad['Y'][19:25]
    rttov_variable[25] = rttov_ds_refl_emmi['bt_refl_total'][19] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    rttov_variable[26:36] = rttov_ds_rad['Y'][26:36]

    rttov_bands =rttov_ds_rad['chan'].values
    print('rttov_variable',np.shape(rttov_variable))
    refl_emmiss =  rttov_variable[:,9:,].transpose(1,2,0)
    
    rttov_ds_rad.close()
    rttov_ds_refl_emmi.close()
    return  refl_emmiss, rttov_bands

### the next should be the last version 
def read_data_refl_emiss_rttov(path_rttov_test):
    '''
    
    output: refl_emmiss (HxWxCH)
    '''
    rttov_ds_refl_emmi = xr.open_dataset(path_rttov_test).compute()
    rttov_variable = np.zeros((np.shape(rttov_ds_refl_emmi['radiances_total'].values)))
    
    
    print("****************variables shape ", np.shape(rttov_ds_refl_emmi['bt_refl_total'].values), np.shape(rttov_variable), np.shape(rttov_ds_refl_emmi['radiances_total'].values))

    rttov_variable[:19] = rttov_ds_refl_emmi['bt_refl_total'][:19] #refl 1-19, 26 rad 20-25 and 27-36
    rttov_variable[19:25] = rttov_ds_refl_emmi['radiances_total'][19:25]
    rttov_variable[25] = rttov_ds_refl_emmi['bt_refl_total'][25] #solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    rttov_variable[26:36] = rttov_ds_refl_emmi['radiances_total'][26:36]

    rttov_bands =rttov_ds_refl_emmi['chan'].values
    print('rttov_variable',np.shape(rttov_variable))
    refl_emmiss =  rttov_variable[:,9:,].transpose(1,2,0)
    
    rttov_ds_refl_emmi.close()
    return  refl_emmiss, rttov_bands
    
    

def read_input_target(rttov_path_rad, rttov_path_refl_emmis, path_output):
    '''
    PC_output (latxlon, number_pcs)
    '''

    refl_emmiss, rttov_bands = read_data_refl_emiss_rttov_old(rttov_path_rad, rttov_path_refl_emmis)

    ###### flat ###########################
    name_file = 'refl_emiss'
    df = dataframe_csv(variable = refl_emmiss, colum = rttov_bands, path_output = path_output, name_file = name_file)
    ###### standard scaler ###########################\
    scaler = preprocessing.StandardScaler().fit(df)  #Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler.transform(df)
    ###### analysis  PCA###########################
    name_plot= "PCA_variance_refl_emiss"
    n_pca = 6 #2 #test JQ 
    n_bands = len(rttov_bands) #2 #test JQ     
    X_reduced_output, pca = PCA_calculation(X_scaled,name_plot,n_pca, path_output) #PC_output_all

    principalDf = pd.DataFrame(data = X_reduced_output
             , columns = [f"PCA_{i}" for i in range(np.shape(X_reduced_output)[1])])
        
        
    return scaler, principalDf, pca
    
##################### no
# def get_test_input(path_output, path_ICON_test, scaler_2D, scaler_3D, pca_3D, n_pca_variables_3D):
#     '''
#     PC_output (latxlon, number_pcs)
#     '''
#     ds,p_2013, T_2013, q_2013, max_cdnc_2013_cm, lwp_2013, lat, lon, height= lwp_nd_input_ICON(path_output = path_output, path_ICON = path_ICON_test) #obtain more outputs of it
#     ds.close()
#     train_inputs_variables_3D = { "pres": p_2013, "ta": T_2013, "hus":q_2013} #(height, lat, lon)

# #     whereAreNaNs = np.isnan(max_cdnc_2013_cm)
# #     max_cdnc_2013_cm[whereAreNaNs] = 0
# #     whereAreNaNs = np.isnan(lwp_2013)
# #     lwp_2013[whereAreNaNs] = 0
    

#     df = pd.DataFrame({  #falta normalizar
#         "qnc_max" : max_cdnc_2013_cm.flatten(),
#         "lwp": lwp_2013.flatten()
#     }) #0,1,2,3,....row,row,row,

#     df = df.fillna(0) #instead of doing it he told me that i should eliminate it 

        
#     test_df= scaler_2D.transform(df)        

#     testing_input_variables_df = pd.DataFrame(test_df, columns = ["qnc_max", "lwp"]) 

#     # Combine with aerosol EOFs
#     i = 0
#     for key, var in train_inputs_variables_3D.items():  
#         var = var.transpose(1,2,0)
#         var_df = pd.DataFrame(var.reshape(-1, var.shape[2]))
#         var_df.columns = [f"{key}_{i}" for i in range(var.shape[2])]
        
#         test_df= scaler_3D[i].transform(var_df)        
#         principalDf = pd.DataFrame(data = pca_3D[i].transform(test_df)
#              , columns = [f"{key}_PCA_{i}" for i in range(n_pca_variables_3D[key])])
#         testing_input_variables_df=pd.concat([testing_input_variables_df, principalDf], axis=1)
 


#         i+=1
        
#     return testing_input_variables_df

# def get_test_output(path_rttov_test, path_output, scaler, pca):
    
#     y_test, rttov_bands = read_data_refl_emiss_rttov(path_rttov_test)
#     ###### flat ###########################
#     name_file = 'refl_emiss_test'
#     df = dataframe_csv(variable = y_test, colum = rttov_bands, path_output = path_output, name_file = name_file)
    
    
#     test_df= scaler.transform(df)
#     test_df = pca.transform(test_df)
    
#     return test_df
    
def plot_target_prediction(target, prediction, path_output, name_plot):
    n_img = len(target)
    fig = plt.figure(figsize=(5 * n_img, 5 * 2 ))  #WxH
    
    fig.suptitle("Comparation between target and prediction")

    for i in range(n_img):

        # Emulator
        axes = plt.subplot(2,n_img, i + 1)
        ctr = axes.pcolormesh(prediction[i])
        # ctr = axes.pcolormesh(predicted_input)
        axes.set_title(f"Prediction PC_{i}")
        plt.colorbar(ctr)
        
        # Target
        axes0 = plt.subplot(2,n_img, i + 1 + n_img)
        ctr = axes0.pcolormesh(target[i])
        # ctr = axes0.pcolormesh(target)
        axes0.set_title(f"Target PC_{i}")
        plt.colorbar(ctr)
    plt.tight_layout()
        
        
    figure_name = '{}/{}.png'.format(path_output, name_plot) #aca pasarr con todo path
                   
    fig.savefig(figure_name) 
    plt.close()   
    

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
    ax.set_title("PC0")
    ax.set_ylabel("Feature importances")
    fig.tight_layout()
            
    figure_name = '{}/Feature importances.png'.format(path_output) #aca pasarr con todo path
                   
    fig.savefig(figure_name) 
    plt.close()   
        
   
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-ICON', type=str, default='/home/jvillarreal/Documents/phd/dataset/data_rttov_T12.nc', help='path of the dataset is the ICON simulations')
#     arg('--path-OUTPUT-RTTOV', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/VF-output-test-modis-T12.nc', help='path of the dataset the output of RTTOV')
    # arg('--path-OUTPUT-RTTOV', type=str, default='/home/jvillarreal/Documents/phd/output/output-rttov/output-test-2-modis.nc', help='path of the dataset the output of RTTOV')
    arg('--rttov-path-refl-emmis', type = str, default = '/home/jvillarreal/Documents/phd/output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc', help = 'Path of the dataset with only reflectances 1-19 and 26')
    arg('--rttov-path-rad', type = str, default = '/home/jvillarreal/Documents/phd/output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc', help = 'Path of the dataset with only radiances')
    
    arg('--path_rttov_test', type = str, default = '/home/jvillarreal/Documents/phd/output/output-rttov/rttov-131-data-icon-1to36-T09.nc', help = 'Path of the test-dataset ')
    arg('--path_ICON_test', type = str, default = '/home/jvillarreal/Documents/phd/dataset/data_rttov_T09.nc', help = 'Path of the test-dataset ')

    arg('--path-output', type=str, default='/home/jvillarreal/Documents/phd/output/ML_output', help='path of the output data is')

    args = parser.parse_args()

    path_ICON = args.path_ICON
    path_output=args.path_output
    # path_OUTPUT_RTTOV = args.path_OUTPUT_RTTOV 
    rttov_path_refl_emmis = args.rttov_path_refl_emmis
    rttov_path_rad = args.rttov_path_rad
    path_rttov_test = args.path_rttov_test
    path_ICON_test = args.path_ICON_test
    
    # ds=xr.open_dataset(path_ICON)
    # file_name= os.path.splitext(os.path.basename(path_ICON))[0][:-5] 
########### Reading the input and output for the training #############
    train_x_df, scaler_2D, scaler_3D, pca_3D, n_pca_variables_3D = get_training_inputs(path_output,path_ICON)
    
    scaler_Y, train_y_df, PCA_Y = read_input_target(rttov_path_rad, rttov_path_refl_emmis, path_output)


    # test_x_df = get_test_input(path_output, path_ICON_test, scaler_2D, scaler_3D, pca_3D, n_pca_variables_3D)
    # test_y = get_test_output(path_rttov_test, path_output, scaler_Y, PCA_Y)

    count_nan_in_df = train_x_df.isnull().sum()
    print ("--------training_input_variables_df nan-------",count_nan_in_df)  

    x_train, x_test, y_train, y_test = train_test_split(train_x_df, train_y_df, test_size=0.33, random_state=42)


    # rf_pcs = test_random_forest(train_x = train_x_df,
    #                                           train_y = train_y_df, 
    #                                           test_x = test_x_df, 
    #                                           test_y = test_y)


    rf_pcs = test_random_forest(train_x = x_train,
                                          train_y = y_train, 
                                          test_x = x_test, 
                                          test_y = y_test)

    
    
    gt =  y_train 
    pred = rf_pcs.predict(x_train)
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(gt, pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(gt, pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(gt, pred)))
    mape = np.mean(np.abs((gt - pred) / np.abs(gt)))
    print('Mean Absolute Percentage Error (MAPE): \n', round(mape * 100, 2))
    print('Accuracy: \n', round(100*(1 - mape), 2))

    gt =  y_test 
    pred = rf_pcs.predict(x_test)
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(gt, pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(gt, pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(gt, pred)))
    mape = np.mean(np.abs((gt - pred) / np.abs(gt)))
    print('Mean Absolute Percentage Error (MAPE): \n', round(mape * 100, 2))
    print('Accuracy: \n', round(100*(1 - mape), 2))

    score = rf_pcs.score(x_train, y_train)
    print('score in training:', score)  
    score = rf_pcs.score(x_test, y_test)
    print('score in testing:', score)

    #####
    # score = rf_pcs.score(train_x_df, train_y_df)
    # print('score in training:', score)  
    # score = rf_pcs.score(test_x_df, test_y)
    # print('score in testing:', score)
    
    # pred_pcs_train = rf_pcs.predict(train_x_df)
    # predicted_3D_train = from2to3d(pred_pcs_train)
    # test_pred_pcs = rf_pcs.predict(test_x_df)
    # predicted_3D = from2to3d(test_pred_pcs)
    
    
    # xr_output = xr.Dataset(dict(refl_emis = m_out_pc1))
    # # save output to netcdf 
    # xr_output.to_netcdf(path_output/"output_predicted.nc",'w')

    
    # train_y_3D = from2to3d(train_y_df.to_numpy())
    # plot_target_prediction(target = train_y_3D, prediction = predicted_3D_train, path_output = path_output, name_plot = 'target_pred_training')


    # target_3D = from2to3d(test_y)
    # plot_target_prediction(target_3D, predicted_3D, path_output, name_plot = 'target_pred_testing')


    # permutation_test(x = train_x_df, y = train_y_df, model = rf_pcs)




#     sys.stdout.close()
if __name__ == '__main__':
    main()