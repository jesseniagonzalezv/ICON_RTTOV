#!/bin/bash                                                                          

# source ~/.bashrc
# conda activate phd
module load python3
  
  
path_dataset="/work/bb1036/b381362/dataset" 
path_results=$HOME/output/input_output_plots/   #/home/b/b381362/

for hr in 09 12
do
    path_ICON=$path_dataset/data_rttov_T${hr}_dropupbottom_Reff.nc
    echo $path_ICON

    python plots_analize_data.py --path-ICON $path_ICON --path-results $path_results &> $path_results/log_plot_analize_data_${hr}.txt 
    
    
    # --rttov-path-refl-emmis $path_output/output-rttov/rttov-131-data-icon-1to19-26-T12.nc --rttov-path-rad $path_output/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc --path-rttov-test $path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc --path-ICON-test $path_dataset/data_rttov_T09_dropupbottom_Reff.nc
done
            
            
exit 0

