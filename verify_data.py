import numpy as np
import xarray as xr

import argparse

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-data-ini', type=str, default='dataset', help='path of the data initial is')
    arg('--path-data-copied', type=str, default='dataset', help='path of the data copied is' )
    arg('--type-data', type=str, default='2D', help='2D or 3D' )


    args = parser.parse_args()
    #path_data_ini='/poorgafile1/climate/hdcp2/2013' #/work/bb1036/b381362/dataset
    #path_data_copied='/home/jvillarreal'

    fname = args.path_data_ini 
    fname_1timestep = args.path_data_copied 
 

    ds = xr.open_dataset(fname)
    ds_1timestep = xr.open_dataset(fname_1timestep)

    print(ds_1timestep.data_vars)

    print('timestep',ds_1timestep.data_vars['time'].values)
    if(args.type_data == '3D'):
      print('Verifing copied data_3D')
      for v in ds_1timestep.data_vars:
        if (v != ('time') and v !=('height_bnds')):         	 
          print(v,':copied correctly',np.array_equal(ds[v].values[5,:,:,:],ds_1timestep[v].values[:,:,:]))
   		
    elif(args.type_data == '2D'):
      print('Verifing copied data_2D')
      for v in ds_1timestep.data_vars:
        if (v != ('ps') and v !=('t_s') and v != ('time') and v !=('height_2')):
          print(v,':copied correctly',np.array_equal(ds[v].values[25,0,:,:],ds_1timestep[v].values[:,:]))
          	    		
        elif(v != ('time') and v !=('height_2')):
          print(v,':copied correctly',np.array_equal(ds[v].values[25,:,:],ds_1timestep[v].values[:,:]))
         	    		    
            
if __name__ == '__main__':
    main()