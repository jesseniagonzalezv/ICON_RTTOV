#!/bin/sh


#source ~/.bashrc
#conda activate phd
#module load nco

path_data_in=/work/bb1036/b381362/dataset #/poorgafile1/climate/hdcp2/2013 

path_data_out=/work/bb1036/b381362/dataset #/home/jvillarreal

for hr in 12

do

fname3d_in="3d_coarse_day_ll_DOM03_ML_20130502T${hr}0000Z.nc"
fout3d_name=`echo $fname3d_in | cut -f 1 -d '.'`
fout3d_1=`echo $fout3d_name\_1timestep.nc`
fout3d_rttov=`echo $fout3d_name\_1timestep_rttov.nc`
fout3d_rttov_flip=`echo $fout3d_name\_1timestep_rttov_flip.nc`

fname2d_in="2d_surface_day_DOM03_ML_20130502T${hr}0000Z.nc"
fout2d_name=`echo $fname2d_in | cut -f 1 -d '.'`
fout2d_1=`echo $fout2d_name\_grid.nc`
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
#<<COMMENT2

echo 3D generating

cdo -seltimestep,$seltimestep_3D -selvar,cli,clw,clc,hus,qr,qs,pres,ta $path_data_in/$fname3d_in $path_data_out/$fout3d_1
ncwa -a time $path_data_out/$fout3d_1 $path_data_out/$fout3d_rttov
rm $path_data_out/$fout3d_1

python flip-values.py --path-in $path_data_out/$fout3d_rttov --path-out $path_data_out/$fout3d_rttov_flip
python verify_data.py --path-data-in $path_data_in/$fname3d_in  --path-data-copied $path_data_out/$fout3d_rttov_flip --type-data '3D'

#rm $path_data_out/$fout3d_rttov

echo $fout3d_rttov_flip generated



echo 2D generating

cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,v_10m,u_10m,t_s,ps $path_data_in/$fname2d_in $path_data_out/$fout2d_1 
cdo -seltimestep,$seltimestep_2D -selvar,t_s,u_10m,v_10m,ps $path_data_out/$fout2d_1  $path_data_out/$fout2d_2 
ncwa -a height_2,time  $path_data_out/$fout2d_2 $path_data_out/$fout2d_rttov 

python verify_data.py --path-data-in $path_data_in/$fout2d_1  --path-data-copied $path_data_out/$fout2d_rttov --type-data '2D' --n-timestep $seltimestep_2D

rm $path_data_out/$fout2d_2 #$path_data_out/$fout2d_1 
echo $fout2d_rttov generated

#COMMENT2

done

#<<COMMENT1
echo landmask generating

cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,FR_LAND $fnameFR_LAND_in $fout_landmask_rttov
echo $fout_landmask_rttov generated

echo z_ifc,z_mc,topography_c generating


#COMMENT1
cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,z_ifc,z_mc,topography_c $fname_surface $fout_surface 
python create_nc_surface_data_DOM03.py --path-in $fout_surface --path-out $fout_surface_rttov
echo $fout_surface_rttov generated
python verify_data.py --path-data-in $fout_surface  --path-data-copied $fout_surface_rttov --type-data 'surface'

#<<COMMENT3

cdo -O -f nc merge  $path_data_out/$fout3d_rttov_flip $path_data_out/$fout2d_rttov $fout_landmask_rttov $fout_surface_rttov $path_data_out/data_rttov_T12.nc  
##cdo -O -f nc merge $path_data_out/$fout3d_rttov $path_data_out/$fout2d_rttov $fout_landmask_rttov $fout_surface_rttov $fout_test_rttov
echo $path_data_out/data_rttov_T12.nc generated data input to RTTOV
#COMMENT3


python verify_data.py --path-data-in  $path_data_in/$fname3d_in   --path-data-copied $path_data_out/data_rttov_T12.nc --type-data '3D' --n-timestep $seltimestep_3D 
python verify_data.py --path-data-in $path_data_in/$fout2d_1  --path-data-copied $path_data_out/data_rttov_T12.nc --type-data '2D' --n-timestep $seltimestep_2D
python verify_data.py --path-data-in $fout_surface  --path-data-copied $path_data_out/data_rttov_T12.nc --type-data 'surface'
rm $fout_surface

ncks -d lon,5.,6. -d lat,48.,50.  $path_data_out/data_rttov_T12.nc $path_data_out/subset_rttov_T12.nc #Npoint=182*59=10738
echo $path_data_out/subset_rttov_T12.nc subset 



exit 0