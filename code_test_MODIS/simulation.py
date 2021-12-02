# import os
# from pathlib import Path
# from pyhdf.SD import SD, SDC
# import pprint
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import argparse
import pandas as pd
import netCDF4 #https://www.earthinversion.com/utilities/reading-NetCDF4-data-in-python/


def plot_input_ICON(out_file,variable,path_ICON,dimension_variable,input_data="my_data"):
    '''
    input:
    variable: name of the variable of 2 or 3 dimensions
    '''
    ds = xr.open_dataset(path_ICON).compute()
    
    nlat = len(ds['lat'])
    nlon = len(ds['lon'])
    
    if(dimension_variable == 3):
        if(input_data == "example"):
            n_heigh =len(ds['level']) ### consider also a exception si no es eso q sea level
            variable_values= ds[variable].transpose('lat', 'lon','level').values
        else:
            n_heigh =len(ds['height']) ### consider also a exception si no es eso q sea level
            variable_values= ds[variable].transpose('lat', 'lon','height').values
            
        variable_flated = np.reshape(variable_values, (nlat*nlon,-1))
   
    else:
        #qv_values=ds['qv']      #(level, lat, lon) select top of the atmospheric Nd
        #reff_values= ds['Reff'] 
        variable_values = ds[variable].values  #(lat, lon)
        variable_flated = variable_values.flatten()  # covert 2d to 1d array 
        
        fig,ax = plt.subplots(1,1,figsize = (14,8))

        pcm=ax.imshow(np.fliplr(variable_values),origin='lower')#,vmin=100,vmax=1500)
              #  pcm = axes[i].imshow(np.fliplr(sds_list[x][band]),  origin='lower') #interpolation='nearest'
        cbar=fig.colorbar(pcm,ax=ax)
        #plt.show()
        fig.savefig(out_file+'/'+variable+".png") 
        plt.close()

    variable_df = pd.DataFrame(variable_flated)
    variable_df.describe().to_csv(out_file+ "/"+ input_data+ "_"+variable+"_description.csv")        




def plot_variable_RTTOV(ds_array, variable,out_file,input_data="my_data"):
    fig,axes = plt.subplots(5,8,figsize = (32,20))
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    channels=np.shape(ds_array[variable])[0]
    print(channels)
    

        
    for col in range(channels):
            ax = axes[col]
            
            
            if(variable=="brdf"):
                label='BRDF/unitless'
                pcm=ds_array[variable].sel(chan=col).plot(ax=ax, cbar_kwargs={"label":label})
            elif(variable=="Y"):
                label= '$Radiances/(W\,m^{-2}\,\mu m^{-1}\,sr^{-1})$'
                pcm=ds_array[variable].sel(chan=col).plot(ax=ax, cbar_kwargs={"label":label}) #vmin=0, ,  vmax=10 vmax=600,
      
            ax.set_title('Channel %d'% (col+1))
    
    #plt.show()

    fig.delaxes(axes[-1])
    fig.delaxes(axes[-2])
    fig.delaxes(axes[-3])
    fig.delaxes(axes[-4])

    plt.tight_layout()
   
    fig.savefig(out_file+ "/"+ input_data+ "_"+variable+".png") 
    plt.close()



def output_RTTOV(out_file,variable,path_OUTPUT_RTTOV,input_data="my_data"):
    '''
    input:
    variable: brdf/Y/f name of the variable of 3 dimensions (chan, lat, lon)
    '''
    ds = xr.open_dataset(path_OUTPUT_RTTOV).compute()
    print(ds)
    #BRDF_flated = .reshape(-1,)
    nlat = len(ds['lat'])
    nlon = len(ds['lon'])
    n_bands =len(ds['chan'])
    print("number of bands",n_bands)

    
#     pd.set_option('display.float_format', lambda x: '%.4f' % x)
#     pd.set_option('display.max_columns', None)



    # convert to Dataframe 
    variable_values= ds[variable].transpose('lat', 'lon','chan').values
    variable_flated = np.reshape(variable_values, (nlat*nlon,-1))
    variable_df = pd.DataFrame(variable_flated)
    #variable_df.describe()

    for i in range((n_bands)-1):
        count_nan = variable_df[i].isnull().sum()
        print (i,'Count of NaN: ' + str(count_nan))    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    

    variable_df.describe().to_csv(out_file+ "/"+ input_data+ "_"+variable+"_description.csv")       
    
    print("ok dataframe")
    #plot figure 
    plot_variable_RTTOV(ds_array=ds, variable=variable,out_file=out_file,input_data=input_data)
        
     



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-ICON', type=str, default='/home/jvillarreal/Documents/phd/dataset/test-2.nc', help='path of the dataset is the ICON simulations')
#     arg('--path-OUTPUT-RTTOV', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/VF-output-test-modis-T12.nc', help='path of the dataset the output of RTTOV')
    arg('--path-OUTPUT-RTTOV', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/output-test-2-modis.nc', help='path of the dataset the output of RTTOV')

    arg('--path-output', type=str, default='/home/jvillarreal/Documents/phd/output', help='path of the output data is')
    arg('--input-data', type=str, default='test-2', help='name of the input to plot test-2')
    arg('--variable-input', type=str, default='qnc', help='name of the input thermo variable ')
    arg('--dimension-variable', type=int, default=3, help='dimension of the input variable ')
    arg('--type-simulation', type=str, default='ICON', help='radiances or ICON')


    args = parser.parse_args()

    path_ICON = args.path_ICON
    input_data = args.input_data
    path_OUTPUT_RTTOV = args.path_OUTPUT_RTTOV 
    path_output=args.path_output
    variable_input=args.variable_input
    dimension_variable=args.dimension_variable
    type_simulation=args.type_simulation
    
    #plot_input_ICON(out_file=path_output,variable="lwp",path_ICON=path_ICON)
#     output_RTTOV(out_file=path_output,variable='brdf',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV)
#     output_RTTOV(out_file=path_output,variable='Y',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV)

#     output_RTTOV(out_file=path_output,variable='brdf',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV,input_data=input_data)
#     output_RTTOV(out_file=path_output,variable='Y',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV,input_data=input_data)
    
    if(type_simulation=='ICON'):

        plot_input_ICON(out_file=path_output,variable=variable_input, path_ICON=path_ICON, dimension_variable=dimension_variable, input_data=input_data)
        
    else:
        output_RTTOV(out_file=path_output,variable='Y',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV,input_data=input_data)
        #output_RTTOV(out_file=path_output,variable='brdf',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV,input_data=input_data)


        

    
    
#     sys.stdout = open(out_file+'/log_PCA_radiances.txt','wt')
    
    
#     sys.stdout.close()
  
if __name__ == '__main__':
    main()