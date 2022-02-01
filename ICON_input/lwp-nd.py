import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import argparse
import pandas as pd
import netCDF4 

def lwp_nd_input_ICON(out_file,variable,path_ICON):
    '''
    input:
    variable: name of the variable of 2 dimensions
    '''
    ds = xr.open_dataset(path_ICON).compute()
    lwp_2013_a = ds['clwvi']   #2d los demas 3D
    qnc_2013 = ds['qnc']  
    T_2013 = ds['ta'] 
    q_2013 = ds['hus'] 
    p_2013 = ds['pres'] 
    clw_2013 = ds['clw']  #kg/kg

    lwp_2013 = lwp_2013_a*1000 # g/m^2

    ####convert cdnc in m^-3####################
    T_c =  T_2013 - 273.15

    esat_2013 = 0.611*exp((17.3*T_c)/(T_c+237.3))
    esat_2013 = esat_2013*1000.0
    qs_2013 =  0.622*(esat_2013/p_2013)

    r_2013 = q_2013/(1-q_2013)
    RH_2013 = 100*(r_2013/qs_2013)

    pv_2013 = (esat_2013*RH_2013)/100.0
    pd_2013 = p_2013 - pv_2013

    rho_2013 = (pd_2013/(287.058*T_2013)) + (pv_2013/(461.495*T_2013))

    cdnc_2013_cm = (rho_2013*qnc_2013)/1000000.0 # convert to cm^-3

    N = rho_2013*qnc_2013 # im m^-3
    L  = rho_2013*clw_2013 # in kgm^-3
    
    dm  <- dim(qnc_2013) # no se considera time
    #dm1 <- dm[1]
    dm=ds.shape()
    dm1 <- dm[0]
    dm2 <- dm[1]
    dm3 <- dm[2] 


    cdnc_top_2013 = array((dm[1],dm[2],dm[4]) )
#     cot_lyr_2013 = array(NA,dim=c(dm[1],dm[2],dm[3],dm[4]) )



    
    #qv_values=ds['qv']      #(level, lat, lon) select top of the atmospheric Nd
    #reff_values= ds['Reff'] 
    variable_values = ds[variable].values  #(lat, lon)
    variable_values_array = variable_values.flatten()  # covert 2d to 1d array 
    fig,ax = plt.subplots(1,1,figsize = (14,8))

    pcm=ax.imshow(np.fliplr(variable_values),origin='lower')#,vmin=100,vmax=1500)
          #  pcm = axes[i].imshow(np.fliplr(sds_list[x][band]),  origin='lower') #interpolation='nearest'

    cbar=fig.colorbar(pcm,ax=ax)
    #plt.show()
    fig.savefig(out_file+'/'+variable+".png") 
    plt.close()
    
    
    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-ICON', type=str, default='/home/jvillarreal/Documents/phd/dataset/test-2.nc', help='path of the dataset is the ICON simulations')
#     arg('--path-OUTPUT-RTTOV', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/VF-output-test-modis-T12.nc', help='path of the dataset the output of RTTOV')
    arg('--path-OUTPUT-RTTOV', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/output-test-2-modis.nc', help='path of the dataset the output of RTTOV')

    arg('--path-output', type=str, default='/home/jvillarreal/Documents/phd/output', help='path of the output data is')

    args = parser.parse_args()

    path_ICON = args.path_ICON
    path_OUTPUT_RTTOV = args.path_OUTPUT_RTTOV 
    path_output=args.path_output
    
    #plot_input_ICON(out_file=path_output,variable="lwp",path_ICON=path_ICON)
#     output_RTTOV(out_file=path_output,variable='brdf',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV)
#     output_RTTOV(out_file=path_output,variable='Y',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV)

    output_RTTOV(out_file=path_output,variable='brdf',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV,input_data="ex_data")
    output_RTTOV(out_file=path_output,variable='Y',path_OUTPUT_RTTOV=path_OUTPUT_RTTOV,input_data="ex_data")
    
    
    
#     sys.stdout = open(out_file+'/log_PCA_radiances.txt','wt')
    
    
#     sys.stdout.close()
  
if __name__ == '__main__':
    main()