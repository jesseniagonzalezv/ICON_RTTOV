#!/bin/sh



'''
Read MODIS and get values de radiances in W/m2 -µm-sr

'''

#python modis_level1B.py --path-in 'C:/Users/jesse/OneDrive/Documents/thesis/dataset' --path-out 'C:/Users/jesse/OneDrive/Documents/thesis/output'

python PCA_radiances.py --path-in 'C:/Users/jesse/OneDrive/Documents/thesis/output' --path-out 'C:/Users/jesse/OneDrive/Documents/thesis/output'

