
# MODIS LEVEL 1B 

----------
Netcdf with radiances/ reflectances
1.  Creation of the netcdf for the area of Germany using MODIS Level 1B 
    - task = spectral will create a xarray with variables of reflectance and radiances obtained from MODIS (ref_total, rad_total)
```
	$ python modis_l1b_read_plot.py --path-dataset "/work/bb1036/b381362/dataset" --path-output "/work/bb1036/b381362/output" --task "spectral" 
    or 
    bash run_MODIS.sh
    sbatch run_MODIS.sh
```
2. Dataset used: 
    - data_file = 'MYD021KM.A2013122.1140.061.2018046032403.hdf'
    - geolocation_file = 'MYD03.A2013122.1140.061.2018046005026.hdf'
    
3. Output:
    -  MODIS_T1140_Germany_MYD021KM.A2013122.1140.061.2018046032403.nc (Netcdf with the area of Germany including 13.5 and 14.5 channel, channels are not in orden consequtive)
    - organized_chann_MODIS_T1140_Germany_MYD021KM.A2013122.1140.061.2018046032403.nc (channels are in orden from 1-36)


RGB MODIS
