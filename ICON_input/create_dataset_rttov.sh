#!/bin/sh

#cd /work/bb1036/b381362/github/ICON_RTTOV/ICON_input/
source ~/.bashrc
conda activate phd
module load nco
module load python3
module load netcdf-c

path_data_in=/work/bb1036/b381362/dataset_icon #/home/jvillarreal
path_data_out=/work/bb1036/b381362/dataset #/poorgafile1/climate/hdcp2/2013 


for hr in 13 #09 10 15  #12

do

	fname3d_in="3d_coarse_day_ll_DOM03_ML_20130502T${hr}0000Z.nc"
	fout3d_name=`echo $fname3d_in | cut -f 1 -d '.'`
	fout3d_1=`echo $fout3d_name\_1timestep.nc`
	fout3d_rttov=`echo $fout3d_name\_1timestep_rttov.nc`

	fname2d_lwp="2d_cloud_day_DOM03_ML_20130502T${hr}0000Z.nc"
	fout2d_lwp_name=`echo $fname2d_lwp | cut -f 1 -d '.'`
	fout2d_lwp=`echo $fout2d_lwp_name\_grid.nc`


	fname2d_in="2d_surface_day_DOM03_ML_20130502T${hr}0000Z.nc"
	fout2d_name=`echo $fname2d_in | cut -f 1 -d '.'`
	fout2d_1=`echo $fout2d_name\_grid.nc`

	#fout_2d= `echo $fout2d_name\_2D.nc`

	fout2d_2=`echo $fout2d_name\_1timestep.nc`
	fout2d_rttov=`echo $fout2d_name\_1timestep_rttov.nc`


	gridfile=$path_data_in/hdcp2_de_default_nest_R0156m.nc
	fnameFR_LAND_in=$path_data_in/extpar_hdcp2_de_default_nest_R0156m.nc
	fout_landmask_rttov=$path_data_out/landmask_R0156m_rttov.nc

	fname_surface=$path_data_in/GRID_default_3d_fine_DOM03_ML.nc
	fout_surface=$path_data_out/surface_DOM03_${hr}.nc
	fout_surface_rttov=$path_data_out/surface_DOM03_rttov.nc

#for the other cases I use the first timestep , in some cases it is not the same chech it
    #T12
#	seltimestep_3D=7 #20130502.5208333=12:30pm T12 
#	seltimestep_2D=31 #20130502.5208333=12:30pm T12
    
    # other time T09, 
#	seltimestep_3D=1
#	seltimestep_2D=1

    # T = 13:40
	seltimestep_3D=9 #20130502.569444444
	seltimestep_2D=41

    
	#-------------------------------------------------------------------------------
#<<COMMENT1
	echo ------------------- 3D generating-----------------------------
	cdo -seltimestep,$seltimestep_3D -selvar,pres,ta,hus,qnc,clc,cli,clw $path_data_in/$fname3d_in $path_data_out/$fout3d_1 #select one step  qr,qs  clc=tca  cfraction cloud fraction for simple cloud 0-1  clc=cloud cover is in%
	ncwa -a time $path_data_out/$fout3d_1 $path_data_out/$fout3d_rttov #delete the dimension time of the variables
    rm $path_data_out/$fout3d_1
	cdo -selname,clc -divc,100 $path_data_out/$fout3d_rttov $path_data_out/clc_variable.nc #convert percent to fraction
	cdo replace $path_data_out/$fout3d_rttov $path_data_out/clc_variable.nc $path_data_out/$fout3d_1
    rm $path_data_out/$fout3d_rttov

    cdo -setattribute,clc@units="fraction" $path_data_out/$fout3d_1 $path_data_out/$fout3d_rttov
    rm $path_data_out/$fout3d_1 $path_data_out/clc_variable.nc

	echo ------------------- 3D generated-----------------------------

	echo ------------------- verify 3D--------------------------------
	python verify_data.py --path-data-in  $path_data_in/$fname3d_in   --path-data-copied $path_data_out/$fout3d_rttov  --type-data '3D' --n-timestep $seltimestep_3D 


	echo ---------------- $path_data_out/$fout3d_rttov verified ------------



	echo --------------------- 2D generating----------------------------
	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,tas,huss,ps,u_10m,v_10m,t_s $path_data_in/$fname2d_in $path_data_out/$fout2d_1 #variables 2m tas, huss, ps=surface_air_pressure t_s surface skin temperature?=weighted temperature of ground surface  gridding of the variables
#2m
	echo ---------------------- 2D LWP---------------------------------
	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,clwvi $path_data_in/$fname2d_lwp $path_data_out/$fout2d_lwp  #cct  griding of the variables 
	echo ----------------ready 2d LWP---------------------------------
	cdo -O -f nc merge $path_data_out/$fout2d_lwp $path_data_out/$fout2d_1 $path_data_out/$fout2d_name\_2D.nc    
	echo ----------------------Ready merge 2D---------------- 
	cdo -seltimestep,$seltimestep_2D -selvar,tas,huss,ps,u_10m,v_10m,t_s,clwvi $path_data_out/$fout2d_name\_2D.nc $path_data_out/$fout2d_2 #cct
	ncwa -a height_2,height,time  $path_data_out/$fout2d_2 $path_data_out/$fout2d_rttov #delete some dimensions that are not used
	echo ----------------------ready 2D-------------------------------

	echo ------------------- verify 2D--------------------------------
	python verify_data.py --path-data-in $path_data_out/$fout2d_1  --path-data-copied $path_data_out/$fout2d_rttov --type-data '2D' --n-timestep $seltimestep_2D --hour ${hr}
	rm $path_data_out/$fout2d_name\_2D.nc $path_data_out/$fout2d_2 #$path_data_out/$fout2d_1 
	echo ------------ $fout2d_rttov generated correctly---------------

	
	echo ----------------- landmask generating -----------------------
	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,FR_LAND $fnameFR_LAND_in $fout_landmask_rttov #o,1 land,sea gridding
	echo ----------- $fout_landmask_rttov generated-------------------
    
#COMMENT1

	echo ---------------- topography_c generating---------------------
	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,topography_c $fname_surface $fout_surface #z_ifc,z_mc=create_nc_surface ,z_ifc,z_mc
	echo ------------------ $fout_surface generated ------------

#<<COMMENT2

	
	echo ------------------------ Merging all the data------------------------------
	cdo -O -f nc merge  $path_data_out/$fout3d_rttov $path_data_out/$fout2d_rttov $fout_landmask_rttov $fout_surface $path_data_out/data_rttov_T${hr}.nc  
	echo --------- $path_data_out/data_rttov_T${hr}.nc generated input RTTOV --------


	ncks -C -O -x -v height_bnds $path_data_out/data_rttov_T${hr}.nc $path_data_out/data_rttov_T${hr}.nc #delete variable not used

       
	echo --------------------------- verify the final file -----------------------------------------------------------------
	python verify_data.py --path-data-in  $path_data_in/$fname3d_in   --path-data-copied $path_data_out/data_rttov_T${hr}.nc --type-data '3D' --n-timestep $seltimestep_3D 
	python verify_data.py --path-data-in $path_data_out/$fout2d_1  --path-data-copied $path_data_out/data_rttov_T${hr}.nc --type-data '2D' --n-timestep $seltimestep_2D --hour ${hr}
	python verify_data.py --path-data-in $fout_surface  --path-data-copied $path_data_out/data_rttov_T${hr}.nc --type-data 'surface'

    rm $path_data_out/$fout3d_rttov 
    rm $path_data_out/$fout2d_lwp $path_data_out/$fout2d_1 $path_data_out/$fout2d_rttov 
    rm $fout_landmask_rttov
    #rm $fout_surface


#	ncks -d lon,8.,9. -d lat,48.,50.  $path_data_out/data_rttov_T${hr}.nc $path_data_out/subset_rttov_T${hr}.nc #Npoint=182*59=10738
#	echo --------- $path_data_out/subset_rttov_T${hr}.nc subset-------------------------------------------- 


	echo -------------------------- Cut the buttom part ---------------------------------------------------
    ncks -d lat,47.599,54.5  $path_data_out/data_rttov_T${hr}.nc  $path_data_out/data_rttov_T${hr}_dropupbottom.nc
	echo ------------- "$path_data_out/data_rttov_T${hr}_dropupbottom.nc" generated---------------------------- 
    
    
	echo ------------------------Generate Reff, Nd, LWP ---------------------------------------------------
    python reff_lwp_Nd.py --path-in $path_data_out/data_rttov_T${hr}_dropupbottom.nc
	echo --------- generated------------------------------------ 

    rm $path_data_out/data_rttov_T${hr}_dropupbottom.nc

#COMMENT1
       
    python ../code_test/plot_lwp_Nd_reff.py --path-in $path_data_out/data_rttov_T${hr}_dropupbottom_Reff.nc --path-out $HOME/output/output_ICON # first create the folders output/output_ICON
#COMMENT1
#COMMENT2

done
exit 0


