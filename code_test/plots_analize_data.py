import xarray as xr
import argparse
from matplotlib import pyplot as plt
import os
import seaborn as sns
import pandas as pd
# import plotly.express as px

def get_outliers(df, path_results, name_data):
    # fig.update_traces(orientation='h')
    n_var = len(df.columns)
    nrows = 3
    ncols = n_var // nrows + (n_var % nrows > 0)

    #fig,axes = plt.subplots(nrows,ncols) #,figsize = (32,20))

    fig = plt.figure(figsize=(5*ncols, 5*nrows))   #(4*ncols, 4*nrows)) #for the subset
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(n_var):
        ax = plt.subplot(nrows, ncols, i + 1)
        name_column = df.columns[i]
        x = df[name_column]
        ax = sns.boxplot(x = x)
        # ax = px.box(df_x_train, y='Nd_max')
        ax.set_title(("Visualize outliers in {} variable").format(name_column))
        # plt.show()

    plt.tight_layout()
    
    title = "Outliers"

    figure_name = '{}{}_{}'.format(path_results, title,name_data) #aca pasarr con todo path
    fig.savefig(figure_name, dpi=100)
    print("figure saved in ", figure_name)

    
def get_distribution(df, path_results, name_data):   
    n_var = len(df.columns)
    nrows = 3
    ncols = n_var // nrows + (n_var % nrows > 0)

    #fig,axes = plt.subplots(nrows,ncols) #,figsize = (32,20))
    fig = plt.figure(figsize=(5*ncols, 5*nrows))   #(4*ncols, 4*nrows)) #for the subset
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(n_var):
        ax = plt.subplot(nrows, ncols, i + 1)
        name_column = df.columns[i]
        x = df[name_column]
        ax = sns.kdeplot(x, shade=True, color='gray') #
        ax.set_title(("Distribution of {} variable").format(name_column))
        # plt.show()

    plt.tight_layout()

    title = "Distribution" #
    figure_name = '{}{}_{}'.format(path_results, title,name_data) #aca pasarr con todo path
    fig.savefig(figure_name, dpi=100)
    print("figure saved in ", figure_name)
    
    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-ICON', type=str, default='/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc', help='path of the initial data is')
    arg('--path-results', type=str, default='/home/b/b381362/output/input_output_plots/', help='path of the output file to save the results' ) 

    args = parser.parse_args()

    # path_dataset="/work/bb1036/b381362/dataset" 
    # path_results = "/home/b/b381362/output/input_output_plots/" 
    # # path_ICON = path_dataset + "/data_rttov_T12_dropupbottom_Reff.nc"
    # path_ICON = path_dataset + "/data_rttov_T09_dropupbottom_Reff.nc"
    # path_ICON_test = "/home/jvillarreal/Documents/phd/dataset/data_rttov_T09_dropupbottom_Reff.nc"
    # k_fold = 2
    # name_PCA = 'PCA_0'
    # rttov_path_refl_emmis = path_out+ "/output-rttov/rttov-131-data-icon-1to19-26-T12.nc"
    # rttov_path_rad = path_out + "/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc"
    # rttov_path = "/work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc"
    # path_rttov_test = path_out + "/output-rttov/rttov-131-data-icon-1to36-T09.nc"
    # path_out="/home/b/b381362/output/ML_output"       

    path_ICON = args.path_ICON
    path_results = args.path_results
    
    ds = xr.open_dataset(path_ICON)

    # obtain the name of the input file data_rttov_T12_dropupbottom 
    file_name= os.path.splitext(os.path.basename(path_ICON))[0][:-5] #os.path.splitext(fname)[0][:-5] with path
    print(" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dataset {} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ".format(file_name))

    print(ds)

    print(" ========================= Checking data 2D ========================= ")

    df_2d = ds.drop_dims("height").to_dataframe()

    print(df_2d.info())
    print(df_2d.dtypes)
    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    name_file = "Statistics input in 2D variables"
    df_2d.describe().T.applymap("{0:.4f}".format).to_csv("{}/{}.csv".format(path_results, name_file))     

    print("=== Values Null =====\n", df_2d.isnull().sum())

    ##assert that there are no missing values in the dataframe
    # assert pd.notnull(df_2d).all().all()


    print(" ----------- Checking outliers in 2D ----------- ")

    get_outliers(df = df_2d, 
                     path_results = path_results, 
                     name_data = "2d_" + file_name )
    
    
    get_distribution(df = df_2d, 
                     path_results = path_results, 
                     name_data = "2d_" + file_name ) 
        
if __name__ == '__main__':
    main()
