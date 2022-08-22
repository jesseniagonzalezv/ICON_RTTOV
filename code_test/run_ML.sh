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

#python lwp_nd.py --path-ICON '/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc' --path-output $HOME/output/output_ICON

# python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_levante_split_img3D_PCA_all_3D_var_MLPRegressor_2fold_test_PCA1.txt

# python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path /work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_levante_split_img3D_PCA_3D_var_MLRegressor_2fold_test_all_OPAC.txt


# for i in 1 2 3 4 5
# do
#     python random-forest.py --k-fold $i --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path /work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_levante_split_img3D_PCA_3D_var_test_all_OPAC_fold_${i}.txt 

# done


# path_results=$HOME/output/MLRegression 


########### MLRegression
# for k in 5 #2 
# do
#     for n in  0 1 2 3 4 5 
#     do
# python ML_test.py --k-fold $k --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_RF_fold_${k}_PCA_${n}_MLR_old_output_new_inputs.txt 
#     done
# done

# for k in 5 
# do
#     for n in 0 1 2 3 4 5 
#     do
# python ML_test.py --k-fold $k --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_RF_fold_${k}_PCA_${n}_MLR_old_output_new_inputs_onlyscaler.txt 
#     done
# done

path_results=$HOME/output/RF_testcode 
name_pca="all"
# name_model="MLP"
name_model="RF"


for k in 4  
do
python ML_test.py --k-fold $k --name-PCA $name_pca --name-model $name_model --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_RF_fold_${k}_PCA_old_output_new_inputs_scaler.txt  
done