#!/bin/sh

##conda activate modis

path_dataset=/home/jvillarreal/Documents/phd/dataset 
path_output=$HOME/Documents/phd/output 

# python plot_lwp_Nd_reff.py --path-in $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-out $path_output/output_ICON # first create the folders output/output_ICON

# python lwp_nd.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_ICON 

python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rflocal.txt 

################################### LEVANTE #########################################################################################################
# source ~/.bashrc

# module load python3
# conda activate phd
# path_dataset=/work/bb1036/b381362/dataset 
# path_output=$HOME/output   

# #python lwp_nd.py --path-ICON '/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc' --path-output $HOME/output/output_ICON

# python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_levante_split_img3D_PCA_all_3D_var_MLPRegressor.txt
