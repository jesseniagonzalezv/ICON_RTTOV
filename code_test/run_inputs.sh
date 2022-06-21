#!/bin/sh

# source ~/.bashrc
# conda activate phd
# module load python3

path_dataset=/home/jvillarreal/Documents/phd/dataset #/work/bb1036/b381362/dataset #/home/jvillarreal
path_output=$HOME/Documents/phd/output #/poorgafile1/climate

# python plot_lwp_Nd_reff.py --path-in $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-out $path_output/output_ICON # first create the folders output/output_ICON

# python lwp_nd.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_ICON 

python random-forest.py --path-ICON $path_dataset/data_rttov_T12_dropupbottom_Reff.nc --path-output $path_output/output_RF --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc

#--path-rttov-test '/home/jvillarreal/Documents/phd/output/output-rttov/rttov-131-data-icon-1to36-T09.nc' --path_ICON_test = "/home/jvillarreal/Documents/phd/dataset/data_rttov_T09_dropupbottom_Reff.nc"

###################################3333

#python lwp_nd.py --path-ICON '/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc' --path-output $HOME/output/output_ICON

#python random-forest.py --path-ICON '/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc' --path-output $HOME/output/output_RF 
