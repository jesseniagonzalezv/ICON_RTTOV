#!/bin/sh



# '''
# Read input and plot some relationships

# '''


module load python3
# path_output=$HOME/output   
path_results=$HOME/output/output_ICON 
path_data_in=/work/bb1036/b381362/dataset #/home/jvillarreal




for hr in 09 12 13 15  #12

do
    path_dataset=/work/bb1036/b381362/dataset/data_rttov_T${hr}_dropupbottom_Reff.nc 
    
    


    python lwp-nd-radiances.py --path-ICON $path_dataset --path-out $path_results | tee -a $path_results/log_lwp_nd.txt #_${hr}.txt  
# &> 
#$path_output/output-rttov/rttov-131-data-icon-1to36-T09.nc 

done


