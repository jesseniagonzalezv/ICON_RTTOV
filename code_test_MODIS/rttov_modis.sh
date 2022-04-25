

# python plot_comparation.py --rttov-path-refl-emmis '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-131-data-icon-1to19-26-T12.nc' --rttov-path-rad '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-data-icon-1-to-36-not-flip.nc' --MODIS-path '/home/jvillarreal/Documents/phd/dataset/MODIS_Germany_refl_emis_MYD021KM.A2013122.1140.061.2018046032403.nc' --path-output '/home/jvillarreal/Documents/phd/output' --type-variable 'refl_emis' &> log_plot_comparation.txt

###############  Germany simulation RGB ###############
# python plot_germany_rgb.py --path-in '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-131-data-icon-1to19-26-T12.nc' --path-out '/home/jvillarreal/Documents/phd/output' --type  'simulation'

###############  Germany MODIS RGB ###############
# python plot_germany_rgb.py --path-in '/home/jvillarreal/Documents/phd/dataset/Only_valid_range_sample_subset_radiances_38bands_MYD021KM.A2013122.1140.L1B.hdf' --path-out '/home/jvillarreal/Documents/phd/output' --type  'modis'

python plot_germany_rgb.py --path-in '/home/jvillarreal/Documents/phd/dataset/MODIS_Germany_radiances_MYD021KM.A2013122.1140.061.2018046032403.nc' --path-out '/home/jvillarreal/Documents/phd/output' --type  'modis'