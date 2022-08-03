#!/bin/bash                                                                                                                                                                                   
#SBATCH --job-name=RFegressor

#SBATCH --partition=compute

#SBATCH --account=bb1036                                                                                                                                                                      
#SBATCH --nodes=20                                                                                                                                                                             
##SBATCH --ntasks=30                                                                                                                                                                           
##SBATCH --cpus-per-task=6                                                                                                                                                                    
#SBATCH --time=04:00:00                                                                                                                                                                       
#SBATCH --job-name=testing                                                                                                                                                              
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications

#SBATCH --output=models-results.out                                                                                                        
#SBATCH --error=models-results.err                                                                                                         

date > models-results.log

source ~/.bashrc

module load python3
conda activate phd
path_dataset=/work/bb1036/b381362/dataset 
path_output=$HOME/output  





date >> models-results.log
# ulimit -s 204800
# ulimit -c 0


# # python test-job.py >> /scratch/b/b381362/models-results_MLPRegressor.log
# python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_levante_split_img3D_PCA_all_3D_var_MLPRegressor.txt


# for i in 1 2 3 4 5
# do
#     python random-forest.py --k-fold $i --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path /work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_split_img3D_PCA_var_test_OPAC_fold_${i}_allPCA.txt 

#     python ML-test.py --k-fold $i --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path /work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_ML_split_img3D_PCA_var_test_OPAC_fold_${i}_allPCA.txt 


# done
# for i in 3 #1 2 3 4 5
# do
#     for n in 0 1 2 3 4 5 
#     do
#         python random-forest.py --k-fold $i --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path /work/bb1036/rttov_share/rttov-131-36-channels-05022013-07182022.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_split_img3D_PCA_var_test_OPAC_fold_${i}_PCA_${n}.txt 

#     done
# done


for i in 2 #1 2 3 4 5
do
    for n in 0 1 2 3 4 5 
    do
        python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_fold_${i}_PCA_${n}.txt 
    done
done

# >> /scratch/b/b381362/models-results_MLPRegressor.log
echo end >> models-results.log
