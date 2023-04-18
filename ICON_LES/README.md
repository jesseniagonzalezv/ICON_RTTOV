# Creation of the variables needed to run RTTOV using ICON-LES simulations
Check folder ICON_input

----------
1. To obtain the input for the rttov with the variables correspondients firs we use the next code: 
create_icon_input.sh
It will generate a netcdf file

2. Creation de variables lwp, Nd_max, Reff  
- create_icon_input.sh
- It will generate a netcdf with the variables: "tas", "t_s", "ps", "huss", "u_10m", "v_10m", "topography_c", "FR_LAND", "pres", "ta", "hus",  "clc", "clw", "cli", "Reff", "lwp", "Nd_max"

'''
  $ bash create_icon_input.sh
  $ bash create_lwp_Nd_Reff.sh  
'''

