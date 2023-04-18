#!/bin/bash



#path data
main_path=/work/bb1036/rttov_share/dataset_ICON


module load python3

for hr in 08 #09 10 11 12 13 14 15 16 17 18 19 20 

do
   file_icon="icon_input_germany_02052013_T${hr}.nc"
   python create_lwp_Nd_Reff.py --path-in $main_path/$file_icon 
  # echo "--------- Creating $fname3d_in ---------"

done
   
   
exit 0
