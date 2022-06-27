
import os
import numpy as np
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import argparse
import sys
# from icecream import ic
from sklearn import preprocessing
from sklearn.decomposition import PCA



def pair_plot(name_plot,X,n_components,path_output):    
    Bandnames = {str(i): f"Band {i+1}" for i in range(n_components)}

    a = sns.pairplot(pd.DataFrame(X[:,:n_components],
                        columns = Bandnames),
                         diag_kind='kde',plot_kws={"s": 3})
    a.fig.suptitle(name_plot, y=1.00)
    plt.tight_layout()
    a.savefig("{}/{}.png".format(path_output,name_plot)) 
    
    plt.close()
    
def cov_eigval_numpy(X_scaled):
        # Covariance
    np.set_printoptions(precision=3)
    cov = np.cov(X_scaled.transpose())

        # Eigen Values
    EigVal, EigVec = np.linalg.eig(cov)

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
    print("Cumulative explained variance \n",cum_var_exp)
        
    return PC,var_exp,cum_var_exp

def variance_numpy_plot(name_plot,var_exp,cum_var_exp,n_components, path_output): 
    fig, ax = plt.subplots()
    plt.bar(range(1,n_components+1), var_exp, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1,n_components+1), cum_var_exp, where='mid',
             label='Cumulative explained variance',
             color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    fig.savefig("{}/{}.png".format(path_output,name_plot)) 

    plt.close()

def variance_sklearn_plot(name_plot,pca,n_components, path_output):
    fig, ax = plt.subplots()

    plt.bar(range(1,n_components+1), pca.explained_variance_ratio_,
            alpha=0.5,
            align='center')
    plt.step(range(1,n_components+1), np.cumsum(pca.explained_variance_ratio_),
             where='mid',
             color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal Components')
    fig.savefig("{}/{}.png".format(path_output,name_plot)) 
    print("{}/{}.png".format(path_output,name_plot))
    plt.close()
    
    
def PCA_calculation(X_scaled,name_plot,n_pca,path_output):
    #     ###########numpy PCA####################################################
    PC,var_exp,cum_var_exp=cov_eigval_numpy(X_scaled) #it allows to show all the PCs
    #     pair_plot("Pair plot of PCs",PC,n_bands,path_output)    
    #     variance_numpy_plot(name_plot,var_exp,cum_var_exp,n_components, path_output)
    #     ###########end numpy PCA####################################################

    # fit_transform() is used to calculate the PCAs from training data
    pca = PCA(n_components=n_pca) #PCA(n_components=24)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)
    # to get the fit statistics (variance explained per component)
    #print("sklearn var:\n", pca.explained_variance_ratio_) #no ordered
    print(f" sum of explained variance ratios of the PCs : {n_pca,sum(pca.explained_variance_ratio_)}")
    print("explained variance:",np.cumsum(pca.explained_variance_ratio_))  ########????????????????? check here what is the PC

    variance_sklearn_plot(name_plot,pca,n_pca, path_output) 
    
    # pk.dump(PC, open("pca_target.pkl","wb"))

    return X_reduced, pca #PC = X_reduced

                     
def plot_PC(PC_2d_Norm, n_bands, path_output):
    
    fig,axes = plt.subplots(6,6,figsize=(50,23),sharex='all',sharey='all')
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    fig.suptitle('Intensities of Principal Components ', fontsize=30)

    axes = axes.ravel()
    for i in range(n_bands):
        axes[i].imshow(PC_2d_Norm[:,:,i],cmap='gray', vmin=0, vmax=255,  origin='lower')
        axes[i].set_title('PC '+str(i+1),fontsize=25)
        axes[i].axis('off')
    #fig.delaxes(axes[-1])                 
    fig.savefig("{}/Intensities PC.png".format(path_output)) 
    plt.close() 
                     

def dataframe_csv(variable, column, path_output, name_file):
    '''
    input: 
        variable (H,W,CH):  CH will be the columns of the dataframe
        column: values of channels
    output:  dataframe  row = H*W, colum = CH 
    '''
    print('dataframe', np.shape(column), np.shape(variable))
    X_flated = variable.reshape(-1,variable.shape[2]) # #np.stack(X_list, axis=-1)
    
    print(np.shape(X_flated))
    df=pd.DataFrame(X_flated) 
    
    
    for i in range(len(column)):
        count_nan = df[i].isnull().sum()
        print ('In band {} values NaN: {}'.format(column[i], count_nan))  

    
    #df_after_drop=df.dropna( how = 'any' ) # it should not be neede because i cut the buttompart # subset = [1],‘any’ : If any NA values are present, drop that row or column.  NOSE COMO RECONSTRUIR revisar si esto es los ultimos de la parte baja puedo poner simplemente 0 valor REVISAR
    
    
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df.columns= column
    
    df.describe().to_csv("{}/{}.csv".format(path_output, name_file))     
    print("ok dataframe")
    
    return df                    

    
# def main():
#     parser = argparse.ArgumentParser()
#     arg = parser.add_argument
#     arg('--path-in', type=str, default='/home/jvillarreal/Documents/phd/github/output-rttov/', help='path of the dataset is')
#     arg('--name-input', type=str, default='rttov-13-data-icon-1-to-36-not-flip.nc', help='name of the input file' )
#     arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/output', help='path of the output data is' )
#     arg('--n-pca', type=int, default=36, help='number of pca to be used' ) #why 28

 
# if __name__ == '__main__':
#     main()