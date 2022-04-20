

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import argparse

def plot_image_rgb(image_3bands,  out_file):

    print(np.shape(image_3bands),np.min(image_3bands),np.max(image_3bands))

    fig = plt.figure(figsize=(20,15))  
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    ax1=plt.subplot(131)
    ax1.imshow(image_3bands,  origin='lower')# 1,4,3 =0,2,1

    plt.title('RGB simulation')

    fig.savefig("{}/image_rgb_simulation.png".format(out_file)) 
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-in', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/', help='path of the dataset is')
    arg('--name-input', type=str, default='rttov-131-data-icon-1-3-4-T12.nc', help='name of the input file' )
    arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/output', help='path of the output data is' )

    args = parser.parse_args()

    fname_in = args.path_in 
    out_file = args.path_out 
    
    
    name_file = fname_in +args.name_input
    print(name_file)
    filehdf = SD(name_file, SDC.READ)
    datasets_dict = filehdf.datasets()
    
    
    sds_refl= filehdf.select('bt_refl_total').get() #radiances

    R = sds_refl[0,:,]  #Channels 1,3,4 (0,1,2) then change for RGB: 1,4,3  (0,2,1)
    G = sds_refl[2,:,]
    B = sds_refl[1,:,]
    

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    
    gamma = 2.2
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)
    B = np.power(B, 1/gamma)

    # Calculate the "True" Green
    G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
    #G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)

    # The final RGB array :)
    RGB = np.dstack([R, G_true, B])
    #RGB = np.dstack([R, G, B])
    print("RGB:", np.shape(RGB))

    plot_image_rgb(image_3bands = RGB,  out_file = out_file)

    #####################################################


        
if __name__ == '__main__':
    main()