#!/bin/sh



path_data_in=/work/bb1036/b381362/dataset #/poorgafile1/climate/hdcp2/2013 

path_data_out=/work/bb1036/b381362/dataset #/home/jvillarreal

for hr in 12

do

fname3d_in="3d_coarse_day_ll_DOM03_ML_20130502T${hr}0000Z.nc"
fout3d_name=`echo $fname3d_in | cut -f 1 -d '.'`
fout3d_1=`echo $fout3d_name\_1timestep.nc`
fout3d_rttov=`echo $fout3d_name\_1timestep_rttov.nc`

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

: '
echo 3D generating

cdo -seltimestep,6 -selvar,cli,clw,clc,hus,qr,qs,pres,ta $path_data_in/$fname3d_in $path_data_out/$fout3d_1
ncwa -a time $path_data_out/$fout3d_1 $path_data_out/$fout3d_rttov
rm $path_data_out/$fout3d_1

python verify_data.py --path-data-in $path_data_in/$fname3d_in  --path-data-copied $path_data_out/$fout3d_rttov --type-data '3D'
echo $fout3d_rttov generated
'

: '
echo 2D generating

cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,v_10m,u_10m,t_s,ps $path_data_in/$fname2d_in $path_data_out/$fout2d_1 
cdo -seltimestep,26 -selvar,t_s,u_10m,v_10m,ps $path_data_out/$fout2d_1  $path_data_out/$fout2d_2 
ncwa -a height_2,time  $path_data_out/$fout2d_2 $path_data_out/$fout2d_rttov 

python verify_data.py --path-data-in $path_data_in/$fout2d_1  --path-data-copied $path_data_out/$fout2d_rttov --type-data '2D'

rm $path_data_out/$fout2d_1 $path_data_out/$fout2d_2 
echo $fout2d_rttov generated
'
done

: '
echo landmask generating

cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,FR_LAND $fnameFR_LAND_in /$fout_landmask_rttov
echo $fout_landmask generated
#'

: '
echo z_ifc,z_mc,topography_c generating

cdo -P 8 remapnn,myGridDef -setgrid,$gridfile -selname,z_ifc,z_mc,topography_c $fname_surface $fout_surface 

'
#python create_nc_surface_data_DOM03.py --path-in $fout_surface --path-out $fout_surface_rttov
echo $fout_surface_rttov generated
python verify_data.py --path-data-in $fout_surface  --path-data-copied $fout_surface_rttov --type-data 'surface'

#rm $fout_surface

exit 0