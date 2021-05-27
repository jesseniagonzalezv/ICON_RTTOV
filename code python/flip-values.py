
from netCDF4 import Dataset    # Note: python is case-sensitive!
import xarray as xr
import numpy as np
import argparse

#def flip_values_3D_coarse(ds, fname_out):
#	ds.to_netcdf(fname_out) # rewrite to netcdf


def main():
		parser = argparse.ArgumentParser()
		arg = parser.add_argument
		arg('--path-in', type=str, default='dom03.nc', help='path of the initial data is')
		arg('--path-out', type=str, default='dom03_copied.nc', help='path of the copied data is' )
		args = parser.parse_args()
		fname = args.path_in 

		ds=xr.open_dataset(fname)
		ds['cli'][:,:,:]=np.flip(ds['cli'],0).values
		ds['clw'][:,:,:]=np.flip(ds['clw'],0).values

		#print(ds['clc'][:,149,149])
		ds['clc'][:,:,:]=(np.flip(ds['clc'],0).values)/100
		#print(ds['clc'][:,149,149])

		ds['hus'][:,:,:]=np.flip(ds['hus'],0).values
		ds['qr'][:,:,:]=np.flip(ds['qr'],0).values
		ds['qs'][:,:,:]=np.flip(ds['qs'],0).values
		ds['pres'][:,:,:]=np.flip(ds['pres'],0).values
		ds['ta'][:,:,:]=np.flip(ds['ta'],0).values

		ds.to_netcdf( args.path_out)  #'/work/bb1036/b381362/dataset/3D_mod.nc') # rewrite to netcdf

if __name__ == '__main__':
    main()
