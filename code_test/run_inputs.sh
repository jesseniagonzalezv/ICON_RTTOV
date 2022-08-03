#!/bin/sh

# conda activate modis

# path_dataset=/home/jvillarreal/Documents/phd/dataset 
# path_output=$HOME/Documents/phd/output 

# python plot_lwp_Nd_reff.py --path-in $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-out $path_output/output_ICON # first create the folders output/output_ICON

# python lwp_nd.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_ICON 

# python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rflocal.txt 

################################### LEVANTE #########################################################################################################
source ~/.bashrc
conda activate phd
module load python3

path_dataset=/work/bb1036/b381362/dataset 
path_output=$HOME/output   

for i in 1 2 3 4 5
do
    python random-forest.py --k-fold $i --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path /work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_ML_split_img3D_PCA_var_test_OPAC_fold_${i}_allPCA.txt 

done



