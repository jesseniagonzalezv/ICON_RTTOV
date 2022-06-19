import xarray as xr
import argparse
from matplotlib import pyplot as plt
import os


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-in', type=str, default='/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc', help='path of the initial data is')
    arg('--path-out', type=str, default='/home/b/b381362/output/output_ICON', help='path of the copied data is' ) 
    args = parser.parse_args()
    fname = args.path_in 
    file_name= os.path.splitext(os.path.basename(fname))[0][:-5] #os.path.splitext(fname)[0][:-5] with path

    ds=xr.open_dataset(fname)
    
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    ds.lwp.where(ds.lwp != 0).plot(cmap = "jet",vmin=2, vmax = 1200)
    plt.subplot(122)
    ds.Nd_max.where(ds.Nd_max != 0).plot(cmap = "jet",vmin=2, vmax = 800) 
    figure_name = os.sep.join([args.path_out,file_name +'_LWP-Nd.png'])    
    print(figure_name)
    fig.savefig(figure_name) 

    
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(121)
    ds.Nd[119,:,:].where(ds.Nd[119,:,:] != 0).plot(cmap = "jet") #,vmin=2, vmax = 800)
    plt.subplot(122)
    # ds.Reff[119].where(ds.Reff != 0).plot(cmap = "jet",vmin=2, vmax = 800)
    ds.Reff[119,:,].plot(cmap = "jet",vmin=0, vmax = 40)
    figure_name = os.sep.join([args.path_out,file_name +'_Nd-Reff.png'])    
    fig.savefig(figure_name) 
    print(figure_name)

    ds.close()

if __name__ == '__main__':
    main()
