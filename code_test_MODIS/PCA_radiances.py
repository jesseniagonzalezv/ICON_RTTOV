
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys

from icecream import ic
from sklearn import preprocessing
from sklearn.decomposition import PCA



def pair_plot(name_plot,X,n_components,out_file):    
    Bandnames = {str(i): f"Band {i+1}" for i in range(n_components)}

    a = sns.pairplot(pd.DataFrame(X[:,:n_components],
                        columns = Bandnames),
                         diag_kind='kde',plot_kws={"s": 3})
    a.fig.suptitle(name_plot, y=1.00)
    plt.tight_layout()
    a.savefig("{}/{}.png".format(out_file,name_plot)) 
    
    plt.close()
    
def cov_eigval_numpy(X_scaled):
        # Covariance
    np.set_printoptions(precision=3)
    cov = np.cov(X_scaled.transpose())

        # Eigen Values
    EigVal,EigVec = np.linalg.eig(cov)

    print("Eigenvalues:\n\n", EigVal,"\n")
    print("Percentage of Variance Explained by Each Component: \n", EigVal/sum(EigVal))
        
        # Ordering Eigen values and vectors
    order = EigVal.argsort()[::-1]
    EigVal = EigVal[order]
    EigVec = EigVec[:,order]
        
        #Projecting data on Eigen vector directions resulting to Principal Components 
    PC = np.matmul(X_scaled,EigVec)   #cross product
    
    tot = sum(EigVal)  #https://medium.com/luca-chuangs-bapm-notes/principal-component-analysis-pca-using-python-scikit-learn-48c4c13e49af
    var_exp = [(i / tot) for i in sorted(EigVal, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)    
    ic(cum_var_exp)
        
    return PC,var_exp,cum_var_exp

def variance_numpy_plot(name_plot,var_exp,cum_var_exp,n_components, out_file): 
    fig, ax = plt.subplots()
    plt.bar(range(1,n_components+1), var_exp, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1,n_components+1), cum_var_exp, where='mid',
             label='Cumulative explained variance',
             color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    fig.savefig("{}/{}.png".format(out_file,name_plot)) 

    plt.close()

def variance_sklearn_plot(name_plot,pca,n_components, out_file):
    fig, ax = plt.subplots()

    plt.bar(range(1,n_components+1), pca.explained_variance_ratio_,
            alpha=0.5,
            align='center')
    plt.step(range(1,n_components+1), np.cumsum(pca.explained_variance_ratio_),
             where='mid',
             color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal Components')
    fig.savefig("{}/{}.png".format(out_file,name_plot)) 

    plt.close()
    
    
def PCA_calculation(X_scaled,name_plot,n_pca,out_file):
    #     ###########numpy PCA####################################################
    PC,var_exp,cum_var_exp=cov_eigval_numpy(X_scaled)
    #     pair_plot("Pair plot of PCs",PC,n_bands,out_file)    
    #     variance_numpy_plot(name_plot,var_exp,cum_var_exp,n_components, out_file)
    #     ###########end numpy PCA####################################################

    # fit_transform() is used to calculate the PCAs from training data
    pca = PCA(n_components=n_pca)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)
    # to get the fit statistics (variance explained per component)
    #print("sklearn var:\n", pca.explained_variance_ratio_) #no ordered
    print(f" sum of explained variance ratios of the PCs : {n_pca,sum(pca.explained_variance_ratio_)}")
    print("explained variance:",np.cumsum(pca.explained_variance_ratio_))  ########????????????????? check here what is the PC

    variance_sklearn_plot(name_plot,pca,n_pca, out_file) 
    
    return PC, X_reduced

def convert_3D(PC, img_shape,n_bands): 
    #https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f
    import cv2
    
    ic(img_shape, np.shape(PC))
    #PC_2d = np.zeros((img_shape[0],img_shape[1],n_bands))
    PC_2d = np.zeros((628,img_shape[1],n_bands))  #REVISAR PARA QUE CUADRE AL ELIMINAR LOS NANS CREO Q SON DE LA PARTE BAJA

    for i in range(n_bands):
        PC_2d[:,:,i] = PC[:,i].reshape(-1,img_shape[1])

    # normalizing between 0 to 255
    #PC_2d_Norm = np.zeros((img_shape[0],img_shape[1],n_bands))
    PC_2d_Norm = np.zeros((628,img_shape[1],n_bands))#REVISAR PARA QUE CUADRE AL ELIMINAR LOS NANS CREO Q SON DE LA PARTE BAJA

    for i in range(n_bands):
        PC_2d_Norm[:,:,i] = cv2.normalize(PC_2d[:,:,i], np.zeros(img_shape),0,255 ,cv2.NORM_MINMAX)
                     
                     
    return PC_2d_Norm
                     
                     
def plot_PC(PC_2d_Norm, n_bands, out_file):
    fig,axes = plt.subplots(6,6,figsize=(50,23),sharex='all',sharey='all')
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    fig.suptitle('Intensities of Principal Components ', fontsize=30)

    axes = axes.ravel()
    for i in range(n_bands):
        axes[i].imshow(PC_2d_Norm[:,:,i],cmap='gray', vmin=0, vmax=255,  origin='lower')
        axes[i].set_title('PC '+str(i+1),fontsize=25)
        axes[i].axis('off')
    #fig.delaxes(axes[-1])                 
    fig.savefig("{}/Intensities PC.png".format(out_file)) 
    plt.close() 
                     
                     
##not used because i deleted some nan points,only if the data is completed##################
def reconstruction_img(X_reduced,pca,img_shape,n_bands):
    X_img_reduced = np.zeros((628,img_shape[1],n_bands)) #REVISAR PARA QUE CUADRE AL ELIMINAR LOS NANS CREO Q SON DE LA PARTE BAJA

    #X_img_reduced = np.zeros((img_shape[0],img_shape[1],n_bands))
    ic(X_reduced.shape)

    X_inv_pca = pca.inverse_transform(X_reduced)
    X_inversed_scaler = scaler.inverse_transform(X_inv_pca)
    ic(X_inversed_scaler.shape)

    for i in range(n_bands):
        X_img_reduced[:,:,i] = X_inversed_pca[:,i].reshape(-1,img_shape[1])
    ###only to visualize the image reduced show is really close to the original
    print("shape image reconstructed",X_img_reduced.shape)
    return X_img_reduced


def dataframe_csv(variable, colum, out_file):
  ### input (a,b,c) a will be the columns of the dataframe
  # datafram  row = b*c, colum = a  
    print('dataframe', np.shape(colum), np.shape(variable))
    X_flated = variable.reshape(-1,variable.shape[2]) # #np.stack(X_list, axis=-1)
    
    print(np.shape(X_flated))
    df=pd.DataFrame(X_flated) 
    
    
    for i in range(len(colum)):
        count_nan = df[i].isnull().sum()
        print ('In band {} values NaN: {}'.format(colum[i], count_nan))  

    
    df_after_drop=df.dropna( how = 'any' ) # subset = [1],‘any’ : If any NA values are present, drop that row or column.  NOSE COMO RECONSTRUIR revisar si esto es los ultimos de la parte baja puedo poner simplemente 0 valor REVISAR
    
    
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df.columns= colum
    
    df.describe().to_csv(out_file + "/radiances_description.csv")    
    print("ok dataframe")
    
    return df_after_drop
        
    #df_after_drop= df.drop([8,9,10,11,12,13,14,15,16,17], axis=1)
    #df_after_drop=df_after_drop.dropna( subset = [1,5,7,18,37], how = 'any' )
    #ic(df_after_drop.count())  
    
def plot_image_PC(PC_2d_Norm, image_3bands,  out_file):

    #img=img.transpose(1,2,0)
    print(np.shape(image_3bands),np.min(image_3bands),np.max(image_3bands))

    #img2 = (img[:,:,:3].astype(np.float32))/np.max(img)
    #inp = (img*255).astype(np.uint8)
    fig = plt.figure(figsize=(50,30))  
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    ax1=plt.subplot(131)
   # ax1.imshow(np.fliplr(image_3bands),  origin='lower')#,vmin=100,vmax=1500)
    ax1.imshow((image_3bands),  origin='lower')#,vmin=100,vmax=1500)

    ax2=plt.subplot(132)
    ax2.imshow(PC_2d_Norm[:,:,:3][:,:,[0,2,1]].astype(int),  origin='lower')
    ax2.axis('off')
    
    #ax2=plt.subplot(132)
    #ax2.imshow(mask_overlay(img2, mask))
    fig.savefig("{}/image_rgb_3PC.png".format(out_file)) 
    plt.close()
    
    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-in', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/', help='path of the dataset is')
    arg('--name-input', type=str, default='rttov-13-data-icon-1-to-36-not-flip.nc', help='name of the input file' )
    arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/output', help='path of the output data is' )
    arg('--n-pca', type=int, default=36, help='number of pca to be used' ) #why 28

    args = parser.parse_args()

    fname_in = args.path_in 
    out_file = args.path_out 
    
    sys.stdout = open(out_file+'/log_PCA_radiances.txt','wt')

    # Read data
    name_file = fname_in +args.name_input
    ic(name_file)
    filehdf = SD(name_file, SDC.READ)
    datasets_dict = filehdf.datasets()

    for idx,sds in enumerate(datasets_dict.keys()):
        ic(idx,sds)
        
    sds_radiances = filehdf.select('Y') #radiances
    sds_data_radiances=sds_radiances.get() 
    
    sds_BRDF= filehdf.select('brdf').get() #radiances

    ic(sds_data_radiances.shape)
    bands =filehdf.select('chan').get()  #channel

    ic("sds_data_radiances:",np.shape(sds_data_radiances),sds_data_radiances.max(), sds_data_radiances.min())

    sds_data_radiances= np.ma.masked_array(sds_data_radiances, np.isnan(sds_data_radiances))
    
    #####################################################
    #MODIS  (Bands 1 4 3 | Red, Green, Blue) 
    R = sds_data_radiances[0,:,]/ np.max(sds_data_radiances[0,:,:])
    G = sds_data_radiances[3,:,]/ np.max(sds_data_radiances[3,:,:])
    B = sds_data_radiances[2,:,]/ np.max(sds_data_radiances[2,:,:])
    
#     R = sds_BRDF[0,:,]/ np.max(sds_BRDF[0,:,:])
#     G = sds_BRDF[3,:,]/ np.max(sds_BRDF[3,:,:])
#     B = sds_BRDF[2,:,]/ np.max(sds_BRDF[2,:,:])
    
    ic("r,g,b max", np.max(R),np.max(G), np.max(B))

    # image_R = (R- np.mean(R))/np.std(R)
    # image_G = (G - np.mean(G))/np.std(G)
    # image_B = (B- np.mean(B))/np.std(B)
    RGB = np.dstack([R, G, B])
    print("RGB:", np.shape(RGB))
    ic("After scaled -mean/std:", pd.DataFrame(RGB.reshape(-1,3)).describe() )

    #####################################################


    #X_flated= np.ma.masked_array(X_flated, np.isnan(X_flated))
    #X_flat= np.ma.masked_array(X_flat, np.isnan(X_flat))
    X_data = sds_data_radiances
    
    
    
      

     ###### split train val ###########################
#     #from sklearn.model_selection import train_test_split
#     #df_train, df_test = train_test_split(X_scaled, test_size = .3, random_state = 42)
#     #df_train_2, df_test_2 = train_test_split(MB_matrix_scaled, test_size = .3, random_state = 42)
    #X_train =  X_data[:,:,:400].transpose(1,2,0) #CON 400 ESTABA ELIMINANDO MUCHAS NUBES
    X_train =  X_data[:,:,].transpose(1,2,0)

    img_shape = np.shape(X_train) #x,y,ch
#     print(df_train.shape,img_shape,df_train.max()) ##########no seria adecuado serapar asi xq de ahi nose como obtener una imagen
     ###### flat ###########################

    df = dataframe_csv(variable = X_train, colum = bands, out_file = out_file)
 
    print("Before scaled -mean/std:", df.describe())

     ###### standard scaler ###########################\
    scaler = preprocessing.StandardScaler().fit(df)  #Standardize features by removing the mean and scaling to unit variance
    
    X_scaled = scaler.transform(df)
    #df_normalized=(df - df.mean()) / df.std()
    
    print("After scaled -mean/std:", pd.DataFrame(X_scaled).describe() )

    #X_scaled= np.ma.masked_array(X_scaled, np.isnan(X_scaled))      
    ###### analysis  PCA###########################
    name_plot= "PCA_variance"
    n_pca= args.n_pca
    
    PC, X_reduced = PCA_calculation(X_scaled,name_plot,n_pca,out_file)
    
    n_bands = len(bands)
    
                     
    PC_2d_Norm = convert_3D(PC, img_shape,n_bands)
    
    plot_PC(PC_2d_Norm, n_bands, out_file)
    # Rearranging 1-d arrays to 2-d arrays of image size


    plot_image_PC(PC_2d_Norm= PC_2d_Norm, image_3bands = RGB,  out_file = out_file)

    #X_reduced_test = pca.transform(scale(X_test))[:,:1]


    sys.stdout.close()
   # ic.disable()
    
if __name__ == '__main__':
    main()