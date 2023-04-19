#!/bin/bash                                                                                                                                                                                   
#SBATCH --job-name=MODIS

#SBATCH --partition=compute

#SBATCH --account=bb1036

#SBATCH --nodes=1

#SBATCH --time=01:00:00                                                                                                                                                                       
#SBATCH --job-name=MODIS                                                                                                                                                              
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jessenia.gv@gmail.com # Email address for notifications

## SBATCH --output=models-results.out                                                                                                        
## SBATCH --error=models-results.err                                                                                                         
source ~/.bashrc
module load python3

#python modis_l1b_read_plot.py --path-dataset "/work/bb1036/b381362/dataset" --path-output "/work/bb1036/b381362/output" --task "spectral" &> log_creation_MODIS.txt  


main_path=/work/bb1036/b381362/dataset
outpath=/work/bb1036/b381362/output/new_results


python modis_l1b_read_plot.py --path-dataset $main_path --path-output $outpath --task "RGB" 
