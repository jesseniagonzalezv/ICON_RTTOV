

#python modis_l1b_read_plot.py --path-dataset '/home/jvillarreal/Documents/phd/dataset' --path-output '/home/jvillarreal/Documents/phd/output' --task "get_RGB"

#python modis_l1b_read_plot.py --path-dataset '/home/jvillarreal/Documents/phd/dataset' --path-output '/home/jvillarreal/Documents/phd/output' --task "get_refl_emiss" 

#python plot_comparation.py --rttov-path '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc'  --MODIS-path '/home/jvillarreal/Documents/phd/dataset/MODIS_Germany_radiances_MYD021KM.A2013122.1140.061.2018046032403.nc' --path-output '/home/jvillarreal/Documents/phd/output'--type-variable "radiance" 

python plot_comparation.py --rttov-path-refl-emmis '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-131-data-icon-1to19-26-T12.nc' --rttov-path-rad '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc' --MODIS-path '/home/jvillarreal/Documents/phd/dataset/MODIS_Germany_refl_emis_MYD021KM.A2013122.1140.061.2018046032403.nc' --path-output '/home/jvillarreal/Documents/phd/output' --type-variable 'refl_emis' &> log_plot_comparation.txt



# i need to do it only for the Germany area