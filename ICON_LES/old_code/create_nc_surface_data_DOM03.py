from netCDF4 import Dataset    # Note: python is case-sensitive!
import xarray as xr
import numpy as np
import argparse




def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-in', type=str, default='dom03.nc', help='path of the initial data is')
    arg('--path-out', type=str, default='dom03_copied.nc', help='path of the copied data is' )

    args = parser.parse_args()

    fname = args.path_in 
    ds = xr.open_dataset(fname)

    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    #ncfile = Dataset('/home/jvillarreal/GRID_DOM3_new.nc',mode='w',format='NETCDF4') 
    ncfile = Dataset(args.path_out,mode='w',format='NETCDF4') 

    #print(ncfile)

    lat_dim = ncfile.createDimension('lat', 637)     # latitude axis
    lon_dim = ncfile.createDimension('lon', 589)    # longitude axis
    height_dim = ncfile.createDimension('height', 150) # unlimited axis (can be appended to).
    bnds_dim = ncfile.createDimension('bnds', 2)
    for dim in ncfile.dimensions.items():
        print(dim)

    ncfile.title='Modify zifc and zmc'
    #print(ncfile.title)

    lat = ncfile.createVariable('lat', np.float64, ('lat',))
    lat.units = 'degrees_north'
    lat.standard_name = "latitude"
    lat.long_name = 'latitude'
    lat.axis = "Y" 

    lon = ncfile.createVariable('lon', np.float64, ('lon',))
    lon.units = 'degrees_east'
    lon.standard_name = "longitude" 
    lon.long_name = 'longitude'
    lon.axis = "X"
    
    #######why is it needed? 

    height = ncfile.createVariable('height', np.float64, ('height'))
    height.standard_name = "height"
    height.long_name = 'generalized_height'
    height.axis = "Z" ;
    height.bounds = "height_bnds" ;


    height_bnds = ncfile.createVariable('height_bnds', np.float64, ('height','bnds'))

                
    Z_ifc= ncfile.createVariable('z_ifc',np.float32,('height','lat','lon'))
    Z_ifc.units = 'm' 
    Z_mc= ncfile.createVariable('z_mc',np.float32,('height','lat','lon'))
    Z_mc.units = 'm' 

    Topography_c=ncfile.createVariable('topography_c',np.float32,('lat','lon'))

    Z_ifc.standard_name = "geometric_height_at_half_level_center" ;
    Z_ifc.long_name = "geometric height at half level center" ;
    Z_ifc.units = "m" ;
    Z_ifc.param = "6.3.0" ;

    Z_mc.standard_name = "geometric_height_at_full_level_center" ;
    Z_mc.long_name = "geometric height at full level center" ;
    Z_mc.units = "m" ;
    Z_mc.param = "6.3.0" ;

    Topography_c.standard_name = "surface_height" ;
    Topography_c.long_name = "geometric height of the earths surface above sea level" ;
    Topography_c.units = "m" ;
    Topography_c.param = "6.3.0" ;

    lat[:]=ds.lat
    lon[:]=ds.lon
    height[:]=ds.height_2
    bnds=ds.bnds

    #print(height_bnds.shape,ds['height_bnds'].shape )

    height_bnds[:,:]=ds['height_bnds'][1:,:]


    #Z_ifc[:,:,:]=ds['z_ifc'][:0:-1,:,:]
    Z_ifc[:,:,:]=ds['z_ifc'][1:,:,:]

    #Z_mc[:,:,:]=ds['z_mc'][::-1,:,:]
    Z_mc[:,:,:]=ds['z_mc'][:,:,:]

    Topography_c[:,:]=ds['topography_c']


    #print(ncfile)
    # close the Dataset.
    ncfile.close(); 
    print('Dataset was created!')



if __name__ == '__main__':
    main()
