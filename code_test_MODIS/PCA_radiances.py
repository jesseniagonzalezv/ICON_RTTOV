
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
    #     PC,var_exp,cum_var_exp=cov_eigval_numpy(X_scaled)
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
    print("explained variance:",np.cumsum(pca.explained_variance_ratio_))

    variance_sklearn_plot(name_plot,pca,n_pca, out_file) 

    return pca,X_reduced

##not used because i deleted some nan points,only if the data is completed##################
def reconstruction_img(X_reduced,pca,img_shape,n_bands): 
    X_img_reduced = np.zeros((img_shape[0],img_shape[1],n_bands))
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
    X_flated = variable.transpose(1,2,0).reshape(-1,variable.shape[0]) # 
    
    print(np.shape(X_flated))
    df=pd.DataFrame(X_flated) 
    
    
    for i in range(len(colum)):
        count_nan = df[i].isnull().sum()
        print ('In band {} values NaN: {}'.format(colum[i], count_nan))  

    ic("Before drop", df.shape)  
    ic(df.count()) 
    
    df_after_drop=df.dropna( how = 'any' ) # subset = [1],‘any’ : If any NA values are present, drop that row or column.

    ic("After the drop",df_after_drop.shape)  
    ic(df_after_drop.count()) 
    
    
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df.columns= colum
    
    df.describe().to_csv(out_file + "/radiances_description.csv")    
    print("ok dataframe")
    
    return df_after_drop
        
    #df_after_drop= df.drop([8,9,10,11,12,13,14,15,16,17], axis=1)
    #df_after_drop=df_after_drop.dropna( subset = [1,5,7,18,37], how = 'any' )
    #ic(df_after_drop.count())  

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path-in', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/', help='path of the dataset is')
    arg('--name-input', type=str, default='rttov-13-data-icon-1-to-36-not-flip.nc', help='name of the input file' )
    arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/output', help='path of the output data is' )
    arg('--n-pca', type=str, default=28, help='number of pca to be used' )

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
    ic(sds_data_radiances.shape)
    bands =filehdf.select('chan').get()  #channel

    
    sds_data_radiances= np.ma.masked_array(sds_data_radiances, np.isnan(sds_data_radiances))
    #X_flated= np.ma.masked_array(X_flated, np.isnan(X_flated))
    #X_flat= np.ma.masked_array(X_flat, np.isnan(X_flat))
    X_data = sds_data_radiances
    ic(np.shape(X_data),X_data.max(), X_data.min())
      

     ###### split train val ###########################
#     #from sklearn.model_selection import train_test_split
#     #df_train, df_test = train_test_split(X_scaled, test_size = .3, random_state = 42)
#     #df_train_2, df_test_2 = train_test_split(MB_matrix_scaled, test_size = .3, random_state = 42)
    X_train =  X_data[:,:,:400]
    img_shape = X_train.shape[1:]
#     print(df_train.shape,img_shape,df_train.max()) ##########no seria adecuado serapar asi xq de ahi nose como obtener una imagen
     ###### flat ###########################

    df = dataframe_csv(variable = X_train, colum = bands, out_file = out_file)
 
     ###### standard scaler ###########################
    scaler = preprocessing.StandardScaler().fit(df)  #Standardize features by removing the mean and scaling to unit variance
    X_scaled = scaler.transform(df)
    #X_scaled= np.ma.masked_array(X_scaled, np.isnan(X_scaled))      
    ###### analysis  PCA###########################
    name_plot= "PCA_variance"
    n_pca= args.n_pca
    PCA_calculation(X_scaled,name_plot,n_pca,out_file)
    


    #X_reduced_test = pca.transform(scale(X_test))[:,:1]


    sys.stdout.close()
   # ic.disable()
    
if __name__ == '__main__':
    main()