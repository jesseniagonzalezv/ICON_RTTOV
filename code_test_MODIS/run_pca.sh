#!/bin/sh



'''
Read simulation and run the code to do PCA 

'''

# python PCA_radiances.py --path-in '/home/jvillarreal/Documents/phd/github/output-rttov/' --name-input 'rttov-13-data-icon-1-to-36-not-flip.nc' --path-out '/home/jvillarreal/Documents/phd/output' --n-pca 36

python PCA_radiances.py --path-in '/home/jvillarreal/Documents/phd/github/output-rttov/' --name-input 'rttov-131-data-icon-1-3-4-T12.nc' --path-out '/home/jvillarreal/Documents/phd/output' 
