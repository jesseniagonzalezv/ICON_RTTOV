#!/bin/bash

#SBATCH --job-name=create_dataset
#SBATCH --partition=compute
#SBATCH --account=bb1036
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=testing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications

#SBATCH -o create_icon.o%j
#SBATCH --error=create_icon.log


#Set environment
module purge && module load cdo nco

#Input data
path_data_in=/work/bb1036/b381362/dataset_icon
#Output data
path_data_out=/work/bb1036/rttov_share/dataset_ICON

##############################################################################################################################
# nlayer = nlevel -1  (height 150)
# rttov require p, q, t by level (height_2:151)  =====> pres_ifc, hum_ifc, temp_ifc
# other variables 3D by level (heigh 150)
##############################################################################################################################

for hr in 08 #08 09 10 11 12 13 14 15 16 17 18 19 20 

do
   date >> create_icon.log

   #3D variables
   fname3d_in="3d_coarse_day_ll_DOM03_ML_20130502T${hr}0000Z.nc"
   fout3d_name=`echo $fname3d_in | cut -f 1 -d '.'`
   fout3d_1=`echo $fout3d_name\_1timestep.nc`
   fout3d_2=`echo $fout3d_name\_1timestep_2.nc`
   fout3d_3=`echo $fout3d_name\_1timestep_3.nc`
   fout3d_rttov=`echo $fout3d_name\_1timestep_rttov.nc`
   
   #2D variables
   fname2d_in="2d_surface_day_DOM03_ML_20130502T${hr}0000Z.nc"
   fout2d_name=`echo $fname2d_in | cut -f 1 -d '.'`
   fout2d_1=`echo $fout2d_name\_1timestep.nc`
   fout2d_2=`echo $fout2d_name\_1timestep_2.nc`
   fout2d_3=`echo $fout2d_name\_1timestep_3.nc`
   fout2d_rttov=`echo $fout2d_name\_1timestep_rttov.nc`
   
   #2D variables - LWP
   fname2d_lwp="2d_cloud_day_DOM03_ML_20130502T${hr}0000Z.nc"
   fout2d_lwp_name=`echo $fname2d_lwp | cut -f 1 -d '.'`
   fout2d_lwp=`echo $fout2d_lwp_name\_grid.nc`

   grid_file=$path_data_in/hdcp2_de_default_nest_R0156m.nc
   fname_FR_LAND=$path_data_in/extpar_hdcp2_de_default_nest_R0156m.nc
   fout_landmask_rttov=$path_data_out/landmask_R0156m_rttov.nc
   
   #1D variables
   fname_surface=$path_data_in/GRID_default_3d_fine_DOM03_ML.nc
   fout_surface=$path_data_out/surface_DOM03_${hr}.nc
   fout_surface_rttov=$path_data_out/surface_DOM03_rttov.nc
   
   #T=13:40pm
   #seltimestep_3D=9
   #seltimestep_2D=41

   if [ ${hr} -eq 13 ]
   then
    seltimestep_3D=9
    seltimestep_2D=41

   elif [ ${hr} -eq 08 ]
   then
    seltimestep_3D=1
    seltimestep_2D=5
   else                        # T09, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20
    seltimestep_3D=1
    seltimestep_2D=1
   fi

   ##############################################################################################################################
   echo "--------- Interpolation of pressure, temperature and humidity variables on half-levels ---------"
   
   #Gridding of height variables
   #old: z_ifc(height 151, ncells) #new: z_ifc(height_2 151, lat,lon)
   cdo -P 8 remapnn,myGridDef -setgrid,$grid_file -selname,z_ifc $fname_surface $path_data_out/tgtcoordinate.nc
   ncrename -d height,height_2 $path_data_out/tgtcoordinate.nc
   
   #old: z_mc(height_2 150, ncells) #new: z_mc(height 150, lat,lon)
   cdo -P 8 remapnn,myGridDef -setgrid,$grid_file -selname,z_mc $fname_surface $path_data_out/infile2.nc
   ncrename -d height_2,height $path_data_out/infile2.nc
   
   #============================================================================================================================#
   echo "--------- Interpolation of pressure in progress ---------"
   #Interpolation of pressure in layers to pressure on levels
   #https://code.mpimet.mpg.de/projects/cdo/embedded/index.html#x1-7200002.12.12
   #infile2 contains a single 3D variable, which represents the source 3D vertical coordinate -- z_mc height 150
   #infile1 contains the source data, which the vertical coordinate from infile2 belongs to -- pres(height 150)
   #tgtcoordinate only contains the target 3D height levels -- z_ifc height_2 151

   #Selection of pressure variable
   cdo -seltimestep,$seltimestep_3D -selvar,pres $path_data_in/$fname3d_in $path_data_out/infile1.nc
   
   #Interpolation of pressure in layers to pressure on levels
   cdo intlevelx3d,$path_data_out/tgtcoordinate.nc $path_data_out/infile1.nc $path_data_out/infile2.nc $path_data_out/pres_ifc.nc
   
   #Renaming pressure variable
   ncrename -v pres,pres_ifc $path_data_out/pres_ifc.nc
   
   #Renaming level dimension
   ncrename -d lev,height_2 $path_data_out/pres_ifc.nc
   
   #Removing unnecessary variables
   ncks -C -O -x -v lev,z_ifc $path_data_out/pres_ifc.nc $path_data_out/pres_ifc.nc
   
   #Setting explicit attribute name
   cdo -setattribute,pres_ifc@standard_name="air_pressure_at_half_level_center" $path_data_out/pres_ifc.nc \
   $path_data_out/pres_ifc_renamed.nc
      
   echo "--------- Interpolation of pressure done ---------"
   #============================================================================================================================#
   
   #============================================================================================================================#
   echo "--------- Interpolation of temperature in progress ---------"
   
   #Selection of temperature variable
   cdo -seltimestep,$seltimestep_3D -selvar,ta $path_data_in/$fname3d_in $path_data_out/infile1.nc
   
   #Interpolation of temperature in layers to pressure on levels
   cdo intlevelx3d,$path_data_out/tgtcoordinate.nc $path_data_out/infile1.nc $path_data_out/infile2.nc $path_data_out/temp_ifc.nc
   
   #Renaming temperature variable
   ncrename -v ta,ta_ifc $path_data_out/temp_ifc.nc
   
   #Renaming level dimension
   ncrename -d lev,height_2 $path_data_out/temp_ifc.nc
   
   #Removing unnecessary variables
   ncks -C -O -x -v lev,z_ifc $path_data_out/temp_ifc.nc $path_data_out/temp_ifc.nc
   
   #Setting explicit attribute name
   cdo -setattribute,ta_ifc@standard_name="air_temperature_at_half_level_center" $path_data_out/temp_ifc.nc \
   $path_data_out/temp_ifc_renamed.nc
      
   echo "--------- Interpolation of temperature done ---------"
   #============================================================================================================================#
   
   #============================================================================================================================#
   echo "--------- Interpolation of humidity in progress ---------"
   
   #Selection of humidity variable
   cdo -seltimestep,$seltimestep_3D -selvar,hus $path_data_in/$fname3d_in $path_data_out/infile1.nc
   
   #Interpolation of humidity in layers to pressure on levels
   cdo intlevelx3d,$path_data_out/tgtcoordinate.nc $path_data_out/infile1.nc $path_data_out/infile2.nc $path_data_out/hus_ifc.nc
   
   #Renaming humidity variable
   ncrename -v hus,hus_ifc $path_data_out/hus_ifc.nc
   
   #Renaming level dimension
   ncrename -d lev,height_2 $path_data_out/hus_ifc.nc
   
   #Removing unnecessary variables
   ncks -C -O -x -v lev,z_ifc $path_data_out/hus_ifc.nc $path_data_out/hus_ifc.nc
   
   #Setting explicit attribute name
   cdo -setattribute,hus_ifc@standard_name="specific_humidity_at_half_level_center" $path_data_out/hus_ifc.nc \
   $path_data_out/hus_ifc_renamed.nc
   
   echo "--------- Interpolation of humidity done ---------"
   #============================================================================================================================#
   
   echo "--------- Interpolation on half-levels done ---------"
   ##############################################################################################################################
   
   ##############################################################################################################################
   echo "--------- 3D generating in progress ---------"
   
   #Extracting 3D variables at selected timestep
   cdo -seltimestep,$seltimestep_3D -selvar,clc,cli,clw,hus,pres,qnc,qr,qs,ta $path_data_in/$fname3d_in \
   $path_data_out/$fout3d_1
   
   #this has the height_2 151
   cdo -seltimestep,$seltimestep_3D -selvar,wa $path_data_in/$fname3d_in \
   $path_data_out/wa.nc
   
   #Converting cloud cover from pourcentage to fraction
   cdo -selname,clc -divc,100 $path_data_out/$fout3d_1 $path_data_out/clc_variable.nc
   cdo replace $path_data_out/$fout3d_1 $path_data_out/clc_variable.nc $path_data_out/$fout3d_2
   cdo -setattribute,clc@units="0-1" $path_data_out/$fout3d_2 $path_data_out/$fout3d_rttov
   
   #Setting explicit variable names for variables located on layers
   cdo -setattribute,hus@standard_name="specific_humidity_at_full_level_center" $path_data_out/$fout3d_rttov \
   $path_data_out/hus_renamed.nc
   cdo -setattribute,pres@standard_name="air_pressure_at_full_level_center" $path_data_out/hus_renamed.nc \
   $path_data_out/pres_renamed.nc
   cdo -setattribute,ta@standard_name="air_temperature_at_full_level_center" $path_data_out/pres_renamed.nc \
   $path_data_out/temp_renamed.nc
   
   mv $path_data_out/temp_renamed.nc $path_data_out/$fout3d_rttov
      
   echo "--------- 3D generated ---------"
   ##############################################################################################################################
   
   ##############################################################################################################################
   echo "--------- 2D generating in progress ---------"
   
   #Gridding of 2D variables
   cdo -P 8 remapnn,myGridDef -setgrid,$grid_file -selname,huss,ps,u_10m,v_10m,tas,t_s \
   $path_data_in/$fname2d_in $path_data_out/$fout2d_1
   
   echo "---------------------- 2D LWP---------------------------------"
   cdo -P 8 remapnn,myGridDef -setgrid,$grid_file -selname,clwvi $path_data_in/$fname2d_lwp $path_data_out/$fout2d_lwp
   echo "---------------- ready 2d LWP ---------------------------------"
#<<COMMENT2
   cdo -O -f nc merge $path_data_out/$fout2d_lwp $path_data_out/$fout2d_1 $path_data_out/$fout2d_name\_2D.nc
  #

   #Extracting 2D variables at selected timestep
   #cdo -seltimestep,$seltimestep_2D -selvar,huss,ps,u_10m,v_10m,tas,t_s $path_data_out/$fout2d_1 $path_data_out/$fout2d_2
   cdo -seltimestep,$seltimestep_2D -selvar,huss,ps,u_10m,v_10m,tas,t_s,clwvi $path_data_out/$fout2d_name\_2D.nc $path_data_out/$fout2d_2

   #Removing unnecessary variables and dimensions
   ncks -C -O -x -v height,height_2 $path_data_out/$fout2d_2 $path_data_out/$fout2d_3
   ncwa -a height,height_2 $path_data_out/$fout2d_3 $path_data_out/$fout2d_rttov
   
   echo "--------- 2D generated ---------"
   ##############################################################################################################################
   
   ##############################################################################################################################
   echo "--------- 1D generating in progress ---------"
   
   #Gridding of landmask variable
   cdo -P 8 remapnn,myGridDef -setgrid,$grid_file -selname,FR_LAND $fname_FR_LAND $fout_landmask_rttov
   
   #Gridding of topography
   cdo -P 8 remapnn,myGridDef -setgrid,$grid_file -selname,topography_c $fname_surface $path_data_out/topography.nc
   
   echo "--------- 1D generated ---------"
   ##############################################################################################################################
   
   #Renaming file containing geometric height at full level center variable
   mv $path_data_out/infile2.nc $path_data_out/z_mc.nc
   
   #Renaming file containing geometric height at half level center variable
   mv $path_data_out/tgtcoordinate.nc $path_data_out/z_ifc.nc
   
   ##############################################################################################################################
   echo "--------- Merging all the data in progress ---------"
      
   cdo merge $path_data_out/pres_ifc_renamed.nc \
             $path_data_out/temp_ifc_renamed.nc \
             $path_data_out/hus_ifc_renamed.nc \
             $path_data_out/z_ifc.nc \
             $path_data_out/wa.nc \
             $path_data_out/half_levels_variables.nc


      
   ncks -h -A $path_data_out/$fout2d_rttov            $path_data_out/icon_input_germany_T${hr}.nc
   ncks -h -A $fout_landmask_rttov                    $path_data_out/icon_input_germany_T${hr}.nc
   ncks -h -A $path_data_out/topography.nc            $path_data_out/icon_input_germany_T${hr}.nc
   ncks -h -A $path_data_out/half_levels_variables.nc $path_data_out/icon_input_germany_T${hr}.nc
   ncks -h -A $path_data_out/$fout3d_rttov            $path_data_out/icon_input_germany_T${hr}.nc
   ncks -h -A $path_data_out/z_mc.nc                  $path_data_out/icon_input_germany_T${hr}.nc
   
   echo "--------- Merging done ---------"
   ##############################################################################################################################

   ##############################################################################################################################
   echo "--------- Cleaning final file in progress ---------"
   
   #Removing unnecessary variables
   ncks -C -O -x -v height_bnds,height_2_bnds,wa $path_data_out/icon_input_germany_T${hr}.nc \
   $path_data_out/icon_input_germany_T${hr}.nc

   #ncks -C -O -x -v height_bnds,height_2_bnds $path_data_out/icon_input_germany_T${hr}.nc \
   #$path_data_out/icon_input_germany_T${hr}.nc
   

   #Removing time dimension
   ncwa -O -a time $path_data_out/icon_input_germany_T${hr}.nc $path_data_out/icon_input_germany_T${hr}_no_time_dim.nc
   
   #Removing unecessary variable attributes
   ncatted -a cell_methods,,d,, $path_data_out/icon_input_germany_T${hr}_no_time_dim.nc
   
   #Removing history from the global attributes header
   ncatted -h -a history,global,d,, $path_data_out/icon_input_germany_T${hr}_no_time_dim.nc \
   $path_data_out/icon_input_germany_T${hr}_no_time_dim_no_history.nc
   

   #Cutting the bottom part of the domain (NaN values)
   #Whole domain
   ncks -h -d lat,47.599,54.5 $path_data_out/icon_input_germany_T${hr}_no_time_dim_no_history.nc \
   $path_data_out/icon_input_germany_02052013_T${hr}.nc
   
   ########## T16 
   # ncks -h -d lat,47.61,54.5 $path_data_out/icon_input_germany_T${hr}_no_time_dim_no_history.nc \
   # $path_data_out/icon_input_germany_02052013_T${hr}.nc
   
   ##########
   
   #Small area for stratocumulus clouds
   #ncks -h -d lon,12.,12.5 -d lat,51.,51.3 $path_data_out/icon_input_germany_02052013_1340pm_cutted.nc \
   #$path_data_out/icon_input_germany_02052013_1340pm_lon_12-12.5_lat_51-51.3_stratocumulus.nc
   
   #Small area for cumulus clouds
   #ncks -h -d lon,8.,8.5 -d lat,53.7,54. $path_data_out/icon_input_germany_02052013_1340pm_cutted.nc \
   #$path_data_out/icon_input_germany_02052013_1340pm_lon_8-8.5_lat_53.7-54_cumulus.nc
   echo "------------- $path_data_out/icon_input_germany_02052013_T${hr}.nc ---------------------------"

   echo "--------- Cleaning done ---------"
   ##############################################################################################################################
   

   module load python3
   python verify_data.py --path-data-in $path_data_in/$fname3d_in --path-data-copied $path_data_out/icon_input_germany_02052013_T${hr}.nc --type-data '3D' --n-timestep $seltimestep_3D
   python verify_data.py --path-data-in $path_data_out/$fout2d_1  --path-data-copied $path_data_out/icon_input_germany_02052013_T${hr}.nc --type-data '2D' --n-timestep $seltimestep_2D --hour ${hr}
   python verify_data.py --path-data-in $path_data_out/topography.nc  --path-data-copied $path_data_out/icon_input_germany_02052013_T${hr}.nc --type-data 'surface'
   
   
   ##############################################################################################################################
   echo "--------- Removing temporary files in progress ---------"

   rm $path_data_out/$fout3d_1 $path_data_out/$fout3d_2 $path_data_out/clc_variable.nc
   rm $path_data_out/infile1.nc
   rm $path_data_out/hus_ifc.nc $path_data_out/pres_ifc.nc $path_data_out/temp_ifc.nc
   rm $path_data_out/hus_renamed.nc $path_data_out/pres_renamed.nc
   rm $path_data_out/hus_ifc_renamed.nc $path_data_out/pres_ifc_renamed.nc $path_data_out/temp_ifc_renamed.nc
   rm $path_data_out/$fout2d_1 $path_data_out/$fout2d_2 $path_data_out/$fout2d_3
   rm $path_data_out/topography.nc $path_data_out/z_mc.nc $path_data_out/z_ifc.nc
   rm $path_data_out/$fout3d_rttov $path_data_out/$fout2d_rttov $fout_landmask_rttov
   rm $path_data_out/half_levels_variables.nc
   rm $path_data_out/wa.nc

   #Removing remaining temporary files
   rm $path_data_out/icon_input_germany_T${hr}.nc
   rm $path_data_out/icon_input_germany_T${hr}_no_time_dim.nc
   rm $path_data_out/icon_input_germany_T${hr}_no_time_dim_no_history.nc
   rm $path_data_out/$fout2d_lwp
   rm $path_data_out/$fout2d_name\_2D.nc
   echo "--------- Removing done ---------"
   ##############################################################################################################################
   date >> create_icon.log




done
   

   #rm $path_data_out/icon_input_germany_02052013_1340pm_cutted.nc
   
exit 0
