import numpy as np
import xarray as xr
import argparse

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-data-ini', type=str, default='input.nc', help='path of the initial data is')
    arg('--path-data-copied', type=str, default='input_copied.nc', help='path of the copied data is' )
    arg('--type-data', type=str, default='2D', help='2D or 3D or surface' )


    args = parser.parse_args()

    fname = args.path_data_ini 
    fname_1timestep = args.path_data_copied 
 

    ds = xr.open_dataset(fname)
    ds_1timestep = xr.open_dataset(fname_1timestep)

    print(ds_1timestep.data_vars)

    if(args.type_data == '3D'):
      print('Verifing copied data_3D')
      print('timestep',ds_1timestep.data_vars['time'].values)
      for v in ds_1timestep.data_vars:
        if (v != ('time') and v !=('height_bnds')):         	 
          print(v,':copied correctly',np.array_equal(ds[v].values[5,:,:,:],ds_1timestep[v].values[:,:,:]))
   		
    elif(args.type_data == '2D'):
      print('Verifing copied data_2D')
      print('timestep',ds_1timestep.data_vars['time'].values)
      for v in ds_1timestep.data_vars:
        if (v != ('ps') and v !=('t_s') and v != ('time') and v !=('height_2')):
          print(v,':copied correctly',np.array_equal(ds[v].values[25,0,:,:],ds_1timestep[v].values[:,:]))
          	    		
        elif(v != ('time') and v !=('height_2')):
          print(v,':copied correctly',np.array_equal(ds[v].values[25,:,:],ds_1timestep[v].values[:,:]))
         	    		    
    elif(args.type_data == 'surface'):
      print('Verifing copied data_surface')
      for v in ds_1timestep.data_vars:
        if (v == ('z_ifc') and v != ('height_bnds') ):
          print(v,':copied correctly',np.array_equal(ds[v].values[1:,:,:],ds_1timestep[v].values[:,:,:]))
        elif (v != ('topography_c') and v != ('height_bnds') ):
          print(v,':copied correctly',np.array_equal(ds[v].values[:,:,:],ds_1timestep[v].values[:,:,:]))
                                           
        elif( v !=('height_bnds')):
            print(v,':copied correctly',np.array_equal(ds[v].values[:,:],ds_1timestep[v].values[:,:]))
                        


if __name__ == '__main__':
    main()