#!/bin/sh

source ~/.bashrc
conda activate phd
module load python3

# python lwp_nd.py --path-ICON '/home/jvillarreal/Documents/phd/dataset/data_rttov_T12.nc' --path-output '/home/jvillarreal/Documents/phd/output' 

#python random-forest.py --path-ICON '/home/jvillarreal/Documents/phd/dataset/data_rttov_T12_dropupbottom_Reff.nc' --path-output $HOME/output/output_RF 


#python lwp_nd.py --path-ICON '/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc' --path-output $HOME/output/output_ICON

python random-forest.py --path-ICON '/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc' --path-output $HOME/output/output_RF 
