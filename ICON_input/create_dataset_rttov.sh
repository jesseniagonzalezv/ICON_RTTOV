#!/bin/sh

#cd /work/bb1036/b381362/github/ICON_RTTOV/ICON_input/
source ~/.bashrc
conda activate phd
module load nco
module load python3

path_data_in=/work/bb1036/b381362/dataset #/poorgafile1/climate/hdcp2/2013 

path_data_out=/work/bb1036/b381362/dataset #/home/jvillarreal

for hr in 09 #12

do

	fname3d_in="3d_coarse_day_ll_DOM03_ML_20130502T${hr}0000Z.nc"
	fout3d_name=`echo $fname3d_in | cut -f 1 -d '.'`
	fout3d_1=`echo $fout3d_name\_1timestep.nc`
	fout3d_rttov=`echo $fout3d_name\_1timestep_rttov.nc`
	fout3d_rttov_flip=`echo $fout3d_name\_1timestep_rttov_flip.nc`

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
	fout_surface=$path_data_out/surface_DOM03.nc
	fout_surface_rttov=$path_data_out/surface_DOM03_rttov.nc

	seltimestep_3D=7 #20130502.5208333=12:30pm
	seltimestep_2D=13 #20130502.5208333=12:30pm

	#-------------------------------------------------------------------------------

<<COMMENT1
	echo ------------------- 3D generating-----------------------------

	cdo -seltimestep,$seltimestep_3D -selvar,qnc,cli,clw,clc,hus,qr,qs,pres,ta $path_data_in/$fname3d_in $path_data_out/$fout3d_1
	ncwa -a time $path_data_out/$fout3d_1 $path_data_out/$fout3d_rttov
	cdo  -setattribute,clc@units="fraction" -selname,clc -divc,100 $path_data_out/$fout3d_rttov  clc_variable.nc #convert percent to fraction
	
	cdo replace $path_data_out/$fout3d_rttov clc_variable.nc $path_data_out/$fout3d_1
	

	#rm $path_data_out/$fout3d_rttov 
	echo 3D generated


	echo -----------verify 3D---------
	#python flip-values.py --path-in $path_data_out/$fout3d_rttov --path-out $path_data_out/$fout3d_rttov_flip
	#python verify_data.py --path-data-in $path_data_in/$fname3d_in  --path-data-copied $path_data_out/$fout3d_rttov_flip --type-data '3D'
	python verify_data.py --path-data-in $path_data_in/$fname3d_in  --path-data-copied $path_data_out/$fout3d_1  --type-data '3D'
	#rm $path_data_out/$fout3d_rttov
	echo ---------------- $fout3d_rttov_flip generated correctly------------------




	echo --------------------- 2D generating -----------------------------
	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,v_10m,u_10m,t_s,ps,huss,tas $path_data_in/$fname2d_in $path_data_out/$fout2d_1 



	echo ---------------------- 2D LWP----------------
	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,clwvi,cct $path_data_in/$fname2d_lwp $path_data_out/$fout2d_lwp 


	echo ----------------ready 2d lwp----------------
	cdo -O -f nc merge $path_data_out/$fout2d_lwp $path_data_out/$fout2d_1 $path_data_out/$fout2d_name\_2D.nc

	echo ----------------------Ready merge 2D----------------

	 
	#cdo -seltimestep,$seltimestep_2D -selvar,v_10m,u_10m,t_s,ps,huss,tas,clwvi,cct $path_data_out/2d_surface_day_DOM03_ML_20130502T120000Z_2D.nc $path_data_out/$fout2d_2 
	cdo -seltimestep,$seltimestep_2D -selvar,v_10m,u_10m,t_s,ps,huss,tas,clwvi,cct $path_data_out/$fout2d_name\_2D.nc $path_data_out/$fout2d_2 
	ncwa -a height_2,height,time  $path_data_out/$fout2d_2 $path_data_out/$fout2d_rttov 

	echo ----------------------ready 2D----------------------


	echo ----------verify 2D--------
	python verify_data.py --path-data-in $path_data_in/$fout2d_1  --path-data-copied $path_data_out/$fout2d_rttov --type-data '2D' --n-timestep $seltimestep_2D --hour ${hr}

	rm $path_data_out/$fout2d_2 #$path_data_out/$fout2d_1 

	echo ------------------- $fout2d_rttov generated correctly----------------------

	#COMMENT2



	
	echo -------------------- landmask generating -------------------------

	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,FR_LAND $fnameFR_LAND_in $fout_landmask_rttov
	echo ------------- $fout_landmask_rttov generated----------------------

	echo ----------------z_ifc,z_mc,topography_c generating----------------
	cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,z_ifc,z_mc,topography_c $fname_surface $fout_surface 
	python create_nc_surface_data_DOM03.py --path-in $fout_surface --path-out $fout_surface_rttov
	echo ------------------ $fout_surface_rttov generated -----------------

	python verify_data.py --path-data-in $fout_surface  --path-data-copied $fout_surface_rttov --type-data 'surface'

	

	cdo -O -f nc merge  $path_data_out/$fout3d_1 $path_data_out/$fout2d_rttov $fout_landmask_rttov $fout_surface_rttov $path_data_out/data_rttov_T${hr}.nc  
	##cdo -O -f nc merge $path_data_out/$fout3d_rttov $path_data_out/$fout2d_rttov $fout_landmask_rttov $fout_surface_rttov $fout_test_rttov

	echo ------------------- $path_data_out/data_rttov_T${hr}.nc generated data input to RTTOV ---------------------------------------------
COMMENT1


	echo ------------ verify------------------------
	python verify_data.py --path-data-in  $path_data_in/$fname3d_in   --path-data-copied $path_data_out/data_rttov_T${hr}.nc --type-data '3D' --n-timestep $seltimestep_3D 
	python verify_data.py --path-data-in $path_data_in/$fout2d_1  --path-data-copied $path_data_out/data_rttov_T${hr}.nc --type-data '2D' --n-timestep $seltimestep_2D --hour ${hr}
	python verify_data.py --path-data-in $fout_surface  --path-data-copied $path_data_out/data_rttov_T${hr}.nc --type-data 'surface'
<<COMMENT2
	rm $fout_surface


	ncks -C -O -x -v height_bnds $path_data_out/data_rttov_T${hr}.nc $path_data_out/data_rttov_T${hr}.nc

	echo --------------------------subset-----------------------------------
	ncks -d lon,8.,9. -d lat,48.,50.  $path_data_out/data_rttov_T${hr}.nc $path_data_out/subset_rttov_T${hr}.nc #Npoint=182*59=10738
	echo --------- $path_data_out/subset_rttov_T${hr}.nc subset--------------- 
COMMENT2
	

	done
exit 0
