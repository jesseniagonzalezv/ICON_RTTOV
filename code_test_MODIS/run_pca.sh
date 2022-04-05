#!/bin/sh



'''
Read simulation and run the code to do PCA 

'''

#python modis_level1B.py --path-in 'C:/Users/jesse/OneDrive/Documents/thesis/dataset' --path-out 'C:/Users/jesse/OneDrive/Documents/thesis/output'

python PCA_radiances.py --path-in '/home/jvillarreal/Documents/phd/github/output-rttov/' --name-input 'rttov-13-data-icon-1-to-36-not-flip.nc' --path-out '/home/jvillarreal/Documents/phd/output' --n-pca 36

