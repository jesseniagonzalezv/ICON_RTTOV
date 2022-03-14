import numpy as np
import xarray as xr
import argparse

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-data-ini', type=str, default='input.nc', help='path of the initial data is')
    arg('--path-data-copied', type=str, default='input_copied.nc', help='path of the copied data is' )
    arg('--type-data', type=str, default='2D', help='2D or 3D or surface' )
    arg('--n-timestep', type=int, default=7, help='timestep 2=13 or 3D=7')
    #arg('--flip-div100', type=str, default='y', help='y= it was flip and divided between 100, n=not flip, not divided' )

    file_lwp= xr.open_dataset("/work/bb1036/b381362/dataset/2d_cloud_day_DOM03_ML_20130502T120000Z_grid.nc")

    args = parser.parse_args()


    fname = args.path_data_ini 
    fname_1timestep = args.path_data_copied 
 

    ds = xr.open_dataset(fname)
    ds_1timestep = xr.open_dataset(fname_1timestep)

    ds_variables=ds_1timestep.data_vars
    print(ds_variables)

    if 'time' in ds_variables:
      print('timestep',ds_variables['time'].values)

    if(args.type_data == '3D'):
      print('Verifing copied data_3D')
      for v in ds_variables:
        #if (v != ('time') and v !=('height_bnds')): 
        if(v== ('qnc') or v == ('cli') or v == ('clw') or v == ('hus') or v == ('qr') or v == ('qs') or v == ('pres') or v == ('ta') ):       	 
          #print(v,':copied correctly',np.array_equal(ds[v].values[(args.n_timestep-1),::-1,:,:],ds_1timestep[v].values[:,:,:]))
          print(v,':copied correctly',np.array_equal(ds[v].values[(args.n_timestep-1),:,:,:],ds_1timestep[v].values[:,:,:]))
          
        elif(v == ('clc')):
          #print(v,':copied correctly',np.array_equal((ds[v].values[(args.n_timestep-1),::-1,:,:]/100),ds_1timestep[v].values[:,:,:]))
          print(v,':copied correctly',np.array_equal((ds[v].values[(args.n_timestep-1),:,:,:]/100),ds_1timestep[v].values[:,:,:]))
    elif(args.type_data == '2D'):
      print('Verifing copied data_2D')
      for v in ds_variables:        
        if (v == ('u_10m') or v ==('v_10m')): 
          print(v,':copied correctly',np.array_equal(ds[v].values[(args.n_timestep-1),0,:,:],ds_1timestep[v].values[:,:]))
          	    		
        elif(v == ('ps') or v ==('t_s') ):
          print(v,':copied correctly',np.array_equal(ds[v].values[(args.n_timestep-1),:,:],ds_1timestep[v].values[:,:]))
        elif(v == ('clwvi')):
          print(v, ':copied correctly', np.array_equal(file_lwp[v].values[(args.n_timestep-1),:,:],ds_1timestep[v].values[:,:]))

    elif(args.type_data == 'surface'):
      print('Verifing copied data_surface')
      for v in ds_variables:
        if (v == ('z_ifc')):
          #print(v,':copied correctly',np.array_equal(ds[v].values[1:,:,:],ds_1timestep[v].values[:,:,:]))
          #print(v,':copied correctly',np.array_equal(ds[v].values[:0:-1,:,:],ds_1timestep[v].values[:,:,:]))
          print(v,':copied correctly',np.array_equal(ds[v].values[1:,:,:],ds_1timestep[v].values[:,:,:]))
        elif (v == ('z_mc')): 
          #print(v,':copied correctly',np.array_equal(ds[v].values[::-1,:,:],ds_1timestep[v].values[:,:,:]))
          print(v,':copied correctly',np.array_equal(ds[v].values[:,:,:],ds_1timestep[v].values[:,:,:]))
                                           
        elif( v ==('topography_c')):
          print(v,':copied correctly',np.array_equal(ds[v].values[:,:],ds_1timestep[v].values[:,:]))
	
    ds.close()
    ds_1timestep.close()
    file_lwp.close()                        


if __name__ == '__main__':
    main()
