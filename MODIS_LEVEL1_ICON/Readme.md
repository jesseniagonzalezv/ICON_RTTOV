
# MODIS LEVEL 1B 

----------
1.  Creation of the netcdf for the area of Germany using MODIS Level 1B 
    - task = spectral will create a xarray with variables of reflectance and radiances obtained from MODIS (ref_total, rad_total)
```
	$ python modis_l1b_read_plot.py --path-dataset "/work/bb1036/b381362/dataset" --path-output "/work/bb1036/b381362/output" --task "spectral" 
```
2. Dataset used: 
    - data_file = 'MYD021KM.A2013122.1140.061.2018046032403.hdf'
    - geolocation_file = 'MYD03.A2013122.1140.061.2018046005026.hdf'
