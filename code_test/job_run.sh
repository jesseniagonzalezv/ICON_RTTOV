#!/bin/bash                                                                                                                                                                                   
#SBATCH --job-name=RFegressor

#SBATCH --partition=compute

#SBATCH --account=bb1036                                                                                                                                                                      
#SBATCH --nodes=1                                                                                                                                                                            
##SBATCH --ntasks=30                                                                                                                                                                           
##SBATCH --cpus-per-task=6                                                                                                                                                                    
#SBATCH --time=07:30:00                                                                                                                                                                       
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


# for i in 2 #1 2 3 4 5
# do
#     for n in 0 1 2 3 4 5 
#     do
#         python random-forest.py  --k-fold $i --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_rf_fold_${i}_PCA_${n}.txt 
#     done
# done

# for i in 1 #2 #1 2 3 4 5
# do
#     for n in 2 #1 #0 #1 2 3 4 5 
#     do
# python ML_test.py --k-fold $i --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/ML_output --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> log_RF_fold_${i}_PCA_${n}_MLR2_old_output_all_PCA.txt 
#     done
# done

# for i in 1 3 4 5 #2
# do
#     for n in  0 2 3 4 5 #1 #0 #1 2 3 4 5 
#     do
# python ML_test.py --k-fold $i --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/ML_new_inputs --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_output/ML_new_inputs/log_RF_fold_${i}_PCA_${n}_MLR2_old_output_new_inputs.txt 
#     done
# done


# path_results=$HOME/output/MLRegression 


########### MLRegression
# for k in 5  #2
# do
#     for n in  0 1 2 3 4 5 
#     do
# python ML_test.py --k-fold $k --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_RF_fold_${k}_PCA_${n}_MLR_old_output_new_inputs.txt 
#     done
# done
# for k in 5 1 3 4  
# do
#     for n in  0 1 2 3 4 5 
#     do
# python ML_test.py --k-fold $k --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_RF_fold_${k}_PCA_${n}_MLR_old_output_new_inputs.txt 
#     done
# done


# for k in 2 
# do
#     for n in  0 
#     do
# python ML_test.py --k-fold $k --name-PCA PCA_${n} --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_RF_fold_${k}_PCA_${n}_MLR_old_output_new_inputs.txt 
#     done
# done


###################test all the pca 

# path_results=$HOME/output/RandomF 

# for k in 5 2 1 3 4  
# do
# python ML_test.py --k-fold $k --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_RF_fold_${k}_PCA_${n}_old_output_new_inputs_scaler.txt 
# done

###################test all the pca only test to check wverything is working


# path_results=$HOME/output/RF_testcode 
path_results=$HOME/output/slides #pca_allinput #RF_drop1000more #RF_changepara #robustscaler 

name_pca="all"
name_model="MLP"
# name_model="RF"


for k in 1 #2 1 3 4 #5
do
python ML_test.py --k-fold $k --name-PCA $name_pca --name-model $name_model --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_{$name_model}_fold_${k}_PCA_old_output_new_inputs.txt  
done


name_model="RF"


for k in 1 #2 1 3 4 #5
do
python ML_test.py --k-fold $k --name-PCA $name_pca --name-model $name_model --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_{$name_model}_fold_${k}_PCA_old_output_new_inputs.txt  
done

# name_model="MLP"


# for k in 1 #5 3 #2 1 3 4  
# do
# python ML_test.py --k-fold $k --name-PCA $name_pca --name-model $name_model --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_{$name_model}_fold_${k}_PCA_old_output_new_inputs.txt  
# done

# path_results=$HOME/output/RF_newoutput 

# for k in 5 2 1 3 4  
# do
# python ML_test.py --k-fold $k --name-PCA $name_pca --name-model $name_model --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_{$name_model}_fold_${k}_PCA_old_output_new_inputs.txt  
# done


# name_model="MLP"
# for k in 4  
# do
# python ML_test.py --k-fold $k --name-PCA $name_pca --name-model $name_model --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_results --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc &> $path_results/log_${name_model}_fold_${k}_PCA_old_output_new_inputs.txt  
# done

# >> /scratch/b/b381362/models-results_MLPRegressor.log
echo end >> models-results.log
