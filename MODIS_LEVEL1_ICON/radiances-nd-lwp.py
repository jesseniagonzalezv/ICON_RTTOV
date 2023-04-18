import xarray as xr
import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sklearn import preprocessing
from sklearn.decomposition import PCA
import sys

# import cv2


################################### kmeans ###################################


def pair_plot(name_plot, X, n_components, path_output):
    Bandnames = {str(i): f"Band {i + 1}" for i in range(n_components)}

    a = sns.pairplot(pd.DataFrame(X[:, :n_components],
                                  columns=Bandnames),
                     diag_kind='kde', plot_kws={"s": 3})
    a.fig.suptitle(name_plot, y=1.00)
    plt.tight_layout()
    a.savefig("{}/{}.png".format(path_output, name_plot))

    plt.close()
    plt.show()



def cov_eigval_numpy(X_scaled):
    # Covariance
    np.set_printoptions(precision=3)
    cov = np.cov(X_scaled.transpose())

    # Eigen Values
    EigVal, EigVec = np.linalg.eig(cov)

    print("Eigenvalues:\n\n", EigVal, "\n")
    print("Percentage of Variance Explained by Each Component: \n", EigVal / sum(EigVal))

    # Ordering Eigen values and vectors
    order = EigVal.argsort()[::-1]
    EigVal = EigVal[order]
    EigVec = EigVec[:, order]

    # Projecting data on Eigen vector directions resulting to Principal Components
    PC = np.matmul(X_scaled, EigVec)  # cross product

    tot = sum(
        EigVal)  # https://medium.com/luca-chuangs-bapm-notes/principal-component-analysis-pca-using-python-scikit-learn-48c4c13e49af
    var_exp = [(i / tot) for i in sorted(EigVal, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    # ic(cum_var_exp)

    return PC, var_exp, cum_var_exp


def variance_numpy_plot(name_plot, var_exp, cum_var_exp, n_components, path_output):
    fig, ax = plt.subplots()
    plt.bar(range(1, n_components + 1), var_exp, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, n_components + 1), cum_var_exp, where='mid',
             label='Cumulative explained variance',
             color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    fig.savefig("{}/{}.png".format(path_output, name_plot))

    plt.close()
    plt.show()



def variance_sklearn_plot(name_plot, pca, n_components, path_output):
    fig, ax = plt.subplots()

    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_,
            alpha=0.5,
            align='center')
    plt.step(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_),
             where='mid',
             color='red')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal Components')
    fig.savefig("{}/{}.png".format(path_output, name_plot))
    print("{}/{}.png".format(path_output, name_plot))
    plt.close()
    plt.show()



def PCA_calculation(X_scaled, name_plot, n_pca, path_output):
    #     ###########numpy PCA####################################################
    PC, var_exp, cum_var_exp = cov_eigval_numpy(X_scaled)  # it allows to show all the PCs
    #     pair_plot("Pair plot of PCs",PC,n_bands,path_output)
    #     variance_numpy_plot(name_plot,var_exp,cum_var_exp,n_components, path_output)
    #     ###########end numpy PCA####################################################

    # fit_transform() is used to calculate the PCAs from training data
    pca = PCA(n_components=n_pca)  # PCA(n_components=24)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)
    # to get the fit statistics (variance explained per component)
    # print("sklearn var:\n", pca.explained_variance_ratio_) #no ordered
    print(f" sum of explained variance ratios of the PCs : {n_pca, sum(pca.explained_variance_ratio_)}")
    print("explained variance:",
          np.cumsum(pca.explained_variance_ratio_))  ########????????????????? check here what is the PC

    variance_sklearn_plot(name_plot, pca, n_pca, path_output)

    # pk.dump(PC, open("pca_target.pkl","wb"))

    return X_reduced, pca  # PC = X_reduced


# def convert_3D(PC, img_shape, n_bands):
#     # https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f
#     import cv2
#
#     # ic(img_shape, np.shape(PC))
#     # PC_2d = np.zeros((img_shape[0],img_shape[1],n_bands))
#     PC_2d = np.zeros(
#         (628, img_shape[1], n_bands))  # REVISAR PARA QUE CUADRE AL ELIMINAR LOS NANS CREO Q SON DE LA PARTE BAJA
#
#     for i in range(n_bands):
#         PC_2d[:, :, i] = PC[:, i].reshape(-1, img_shape[1])
#
#     # normalizing between 0 to 255
#     # PC_2d_Norm = np.zeros((img_shape[0],img_shape[1],n_bands))
#     PC_2d_Norm = np.zeros(
#         (628, img_shape[1], n_bands))  # REVISAR PARA QUE CUADRE AL ELIMINAR LOS NANS CREO Q SON DE LA PARTE BAJA
#
#     for i in range(n_bands):
#         PC_2d_Norm[:, :, i] = cv2.normalize(PC_2d[:, :, i], np.zeros(img_shape), 0, 255, cv2.NORM_MINMAX)
#
#     return PC_2d_Norm

def from2to3d(PC, img_shape, n_bands):
    columns = n_bands
    x_3d = np.zeros(( img_shape[0], img_shape[1], columns))  # 628,589)) lat, lon

    for i in range(columns):
        x_3d[:, :, i] = PC[:, i].reshape(-1, img_shape[1])

    return x_3d
def plot_PC(PC_2d_Norm, n_bands, path_output):
    fig, axes = plt.subplots(6, 6, figsize=(50, 23), sharex='all', sharey='all')
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    fig.suptitle('Intensities of Principal Components ', fontsize=30)

    axes = axes.ravel()
    for i in range(n_bands):
        axes[i].imshow(PC_2d_Norm[:, :, i], cmap='gray', vmin=0, vmax=255, origin='lower')
        axes[i].set_title('PC ' + str(i + 1), fontsize=25)
        axes[i].axis('off')
    # fig.delaxes(axes[-1])
    fig.savefig("{}/Intensities PC.png".format(path_output))
    plt.close()
    plt.show()


        #########################  NO USADO
##not used because i deleted some nan points,only if the data is completed##################
def reconstruction_img(X_reduced, pca, img_shape, n_bands):
    X_img_reduced = np.zeros(
        (628, img_shape[1], n_bands))  # REVISAR PARA QUE CUADRE AL ELIMINAR LOS NANS CREO Q SON DE LA PARTE BAJA

    # X_img_reduced = np.zeros((img_shape[0],img_shape[1],n_bands))
    # ic(X_reduced.shape)

    X_inv_pca = pca.inverse_transform(X_reduced)
    X_inversed_scaler = scaler.inverse_transform(X_inv_pca)
    # ic(X_inversed_scaler.shape)

    for i in range(n_bands):
        X_img_reduced[:, :, i] = X_inversed_pca[:, i].reshape(-1, img_shape[1])
    ###only to visualize the image reduced show is really close to the original
    print("shape image reconstructed", X_img_reduced.shape)
    return X_img_reduced


def dataframe_csv(variable, colum, path_output, name_file):
    ### input (a,b,c) a will be the columns of the dataframe
    # datafram  row = b*c, colum = a
    print('dataframe', np.shape(colum), np.shape(variable))
    X_flated = variable.reshape(-1, variable.shape[2])  # #np.stack(X_list, axis=-1)

    print(np.shape(X_flated))
    df = pd.DataFrame(X_flated)

    for i in range(len(colum)):
        count_nan = df[i].isnull().sum()
        print('In band {} values NaN: {}'.format(colum[i], count_nan))

    df_after_drop = df.dropna(
        how='any')  # subset = [1],‘any’ : If any NA values are present, drop that row or column.  NOSE COMO RECONSTRUIR revisar si esto es los ultimos de la parte baja puedo poner simplemente 0 valor REVISAR

    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    df.columns = colum

    df.describe().to_csv("{}/{}.csv".format(path_output, name_file))
    print("ok dataframe")

    return df_after_drop

    # df_after_drop= df.drop([8,9,10,11,12,13,14,15,16,17], axis=1)
    # df_after_drop=df_after_drop.dropna( subset = [1,5,7,18,37], how = 'any' )
    # ic(df_after_drop.count())


def plot_image_PC(PC_2d_Norm, image_3bands, path_output):
    # img=img.transpose(1,2,0)
    print(np.shape(image_3bands), np.min(image_3bands), np.max(image_3bands))

    # img2 = (img[:,:,:3].astype(np.float32))/np.max(img)
    # inp = (img*255).astype(np.uint8)
    fig = plt.figure(figsize=(50, 30))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    ax1 = plt.subplot(131)
    # ax1.imshow(np.fliplr(image_3bands),  origin='lower')#,vmin=100,vmax=1500)
    ax1.imshow((image_3bands), origin='lower')  # ,vmin=100,vmax=1500)

    ax2 = plt.subplot(132)
    ax2.imshow(PC_2d_Norm[:, :, :3][:, :, [0, 2, 1]].astype(int),
               origin='lower')  # aca creo que xq rgb esta en otro orden
    ax2.axis('off')

    # #ax2=plt.subplot(132)
    ##ax2.imshow(mask_overlay(img2, mask))
    fig.savefig("{}/image_rgb_3PC.png".format(path_output))
    plt.close()
    plt.show()


def plot_image_rgb(image_3bands, path_output):
    print(np.shape(image_3bands), np.min(image_3bands), np.max(image_3bands))

    fig = plt.figure(figsize=(20, 15))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    ax1 = plt.subplot(131)
    ax1.imshow(image_3bands, origin='lower')  # 1,4,3 =0,2,1

    plt.title('RGB')

    fig.savefig("{}/image_rgb_T12.png".format(path_output))
    plt.close()
    plt.show()

from sklearn.cluster import KMeans


def get_kmeans(nbands, imagery, path_output):
    # import natsort  ready

    # create an empty array in which each column will hold a flattened band
    flat_data = np.empty((imagery.shape[0] * imagery.shape[1], nbands))  # x,y

    # loop through each band in the image and add to the data array
    n_PC = 6  # 2 #3 test JQ
    for i in range(n_PC):  # nbands):
        band = imagery[:, :, i]  # ch in the last 2
        flat_data[:, i - 1] = band.flatten()

    # set up the kmeans classification by specifying the number of clusters
    n_cluster = 3 #5
    km = KMeans(n_clusters=n_cluster)
    # begin iteratively computing the position of the two clusters
    km.fit(flat_data)

    # use the sklearn kmeans .predict method to assign all the pixels of an image to a unique cluster
    flat_predictions = km.predict(flat_data)

    # rehsape the flattened precition array into an MxN prediction mask
    prediction_mask = flat_predictions.reshape((imagery.shape[0], imagery.shape[1]))  # x,y
    # ic("prediction_mask", np.shape(prediction_mask))
    # plot the imagery and the prediction mask for comparison

    fig = plt.figure(figsize=(20, 15))

    # plt.imshow(imagery[0,:,:])
    # plt.title("Imagery")
    # plt.axis('off')
    # plt.close()

    plt.imshow(prediction_mask, origin='lower')
    plt.title('kmeans predictions')
    plt.axis('off')

    fig.savefig("{}/kmeans predictions_{}PC_{}cluster.png".format(path_output, n_PC, n_cluster))
    plt.close()

    plt.show()



def n_clusters(nbands, imagery, path_output):
    # create an empty array in which each column will hold a flattened band
    flat_data = np.empty((imagery.shape[0] * imagery.shape[1], nbands))  # x,y

    # loop through each band in the image and add to the data array
    n_PC = 2  # 3 test JQ
    for i in range(n_PC):  # nbands):
        band = imagery[:, :, i]  # ch in the last 2
        flat_data[:, i - 1] = band.flatten()

    data_transformed = flat_data

    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)

    fig = plt.figure(figsize=(20, 15))

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    fig.savefig("{}/Elbow Method For Optimal.png".format(path_output))

    plt.close()
    plt.show()



################################### end kmeans ###################################
def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b

    return (y)


def get_joint_histgram(nx, ny, xmin, xmax, ymin, ymax, x, y):
    """
    :param nx: num of x bins
    :param ny: num of y bins
    :param xmin: lower bound of x bins
    :param xmax: higher bound of x bins
    :param ymin: lower bound of y bins
    :param ymax: higher bound Eof y bins
    :param x: 1D data of x
    :param y: 1D data of y
    :return: jh(ny, nx), x_mid(nx), y_mid(ny)
    """
    jh = np.empty([ny, nx])
    y_median = np.empty([nx])
    x_bins = np.linspace(xmin, xmax, nx + 1)  # bin_edges
    y_bins = np.linspace(ymin, ymax, ny + 1)
    x_mid = x_bins[:-1] + (x_bins[1] - x_bins[0]) / 2
    y_mid = y_bins[:-1] + (y_bins[1] - y_bins[0]) / 2
    # print(x_bins)
    # print(y_bins)
    labels = ['Bin {}'.format(i) for i in range(1, nx + 1)]
    x_bin = pd.cut(x, x_bins, labels=labels)  # x_bin match both x and y
    i = 0
    for bin in labels:
        # print(df.AI[df.x_bins == bin].mean(), df.Nd[df.AI_bins == bin].mean())
        y_median[i] = y[x_bin == bin].median()
        dict_jh = get_PDF_bin_range(y[x_bin == bin], y_bins)
        jh[:, i] = dict_jh['pdf']
        i += 1
    # fig, ax = plt.subplots()
    # cs = ax.contourf(x_mid, y_mid, jh, cmap='Greys')
    # fig.colorbar(cs, ax=ax)
    return (jh, x_mid, y_mid, y_median)


def get_PDF_bin_range(x, x_bins):
    """
    :param x: 1-D array data
    :param x_bins: bin-edges (length = n_bins+1)
    :return: dict: 1) 'x': mid_value of x for each bin
                   2) 'pdf': PDF for each bin (%)
    """
    dict_PDF = {}
    hist, bin_edges = np.histogram(x, bins=x_bins)
    dict_PDF['x'] = bin_edges[0:len(x_bins) - 1] + (bin_edges[1] - bin_edges[0]) / 2  # mid value
    dict_PDF['pdf'] = hist / sum(hist) * 100
    # fig, ax = plt.subplots()
    # ax.plot(dict_PDF['x'], dict_PDF['pdf'], color='black', linewidth=3)
    return (dict_PDF)


def get_values_joint_hist(xedges, yedges):
    """
    :param nx: num of x bins  20
    :param ny: num of y bins  20
    :param xmin: lower bound of x bins
    :param xmax: higher bound of x bins
    :param ymin: lower bound of y bins
    :param ymax: higher bound Eof y bins
    :param x: 1D data of x
    :param y: 1D data of y
    :return: jh(ny, nx), x_mid(nx), y_mid(ny)
    """

    xedges_mid = xedges[:-1] + (xedges[1] - xedges[0]) / 2
    yedges_mid = yedges[:-1] + (yedges[1] - yedges[0]) / 2
    print("========print xedges_mid", xedges_mid)
    print("========print yedges_mid", yedges_mid)

    return (xedges_mid, yedges_mid)

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # arg('--path-in', type=str, default='/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc', help='path of the initial data is')
    arg('--path-ICON', type=str, default='/home/jvillarreal/Documents/phd/dataset/data_rttov_T12_dropupbottom_Reff.nc',
        help='path of the initial data is')
    # arg('--path-rttov', type=str, default='/home/jvillarreal/Documents/phd/dataset/data_rttov_T12_dropupbottom_Reff.nc',
    #     help='path of the initial data is')
    arg('--path-rttov', type=str, default='/home/jvillarreal/Documents/phd/output/output-rttov/rttov-131-36-channels-05022013-07182022.nc',
        help='path of the initial data is')

    # arg('--path-out', type=str, default='/home/b/b381362/output/output_ICON', help='path of the copied data is' )

    arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/output/output_ICON',
        help='path of the copied data is')

    arg('--n-pca', type=int, default=36, help='number of pca to be used' ) #why 28

    args = parser.parse_args()
    fname = args.path_ICON
    file_name = os.path.splitext(os.path.basename(fname))[0][:-5]  # os.path.splitext(fname)[0][:-5] with path
    path_output = args.path_out
    ds = xr.open_dataset(fname)

    rttov_path = args.path_rttov
    # rttov_variable_ds = read_data_refl_emiss_rttov(rttov_path)


    print(" ##################### FILE: {} #####################".format(file_name))

    var_Nd = ds.Nd_max[:, :].values  # lat,lot
    var_LWP = ds.lwp[:, :].values  # lat,lot

    x_Nd = var_Nd.flatten()
    y_lwp = var_LWP.flatten()

    def masked_array(data, threshold):
        return 1-(data > threshold).astype(int)

    mask_array_lwp = masked_array(data = var_LWP , threshold = 0)
    print(mask_array_lwp)
    plt.imshow((mask_array_lwp), cmap='gray',  origin='lower')
    plt.show()
    print("lwp min, max, mean", mask_array_lwp.min(), np.max(mask_array_lwp), np.mean(mask_array_lwp))
    print("lwp uniq", np.unique(mask_array_lwp))

    mask_array = masked_array(data = var_Nd , threshold = 0)
    print(mask_array)
    plt.imshow((mask_array), cmap='gray',  origin='lower')
    plt.show()
    print("var_Nd min, max, mean", mask_array.min(), np.max(mask_array), np.mean(mask_array))
    print("var_Nd uniq", np.unique(mask_array))

    #####################1 ###
    # fig = plt.figure(figsize=(3, 5))
    print("Nd_max min, max", x_Nd.min(), np.max(x_Nd))
    print("LWP min, max", y_lwp.min(), np.max(y_lwp))

    fig = plt.figure(figsize=(14, 4))
    fig.suptitle(file_name, fontsize=10)

    plt.subplot(121)
    ds.Nd_max[:, :].where(var_Nd > 0).plot(cmap="cividis")  # ,
    plt.subplot(122)
    ds.lwp[:, :].where(var_LWP > 0).plot(cmap="cividis")  # ,
    figure_name = '{}_LWP_Nd.png'.format(os.sep.join([path_output, file_name]))

    plt.show()
    fig.savefig(figure_name)

    #####################2 ###
    fig = plt.figure(figsize=(14, 4))
    fig.suptitle(file_name, fontsize=10)

    plt.subplot(121)
    ds.lwp.where(ds.lwp != 0).plot(cmap="cividis", vmin=2, vmax=1200)
    plt.subplot(122)
    ds.Nd_max.where(ds.Nd_max != 0).plot(cmap="cividis", vmin=2, vmax=800)
    figure_name = os.sep.join([args.path_out, file_name + '_LWP2-1200-Nd2-800.png'])

    print(figure_name)
    fig.savefig(figure_name)

    plt.show()
    ##########################
    # count_0

    print(" ============== Nd_max =================")
    arr = x_Nd
    count_no_zeros = np.count_nonzero(arr)
    print(f"number of non-zero: {count_no_zeros}")
    count_zeros = arr.size - np.count_nonzero(arr)
    print(f"number of zeros: {count_zeros}")
    print(f"percent of zeros: {count_zeros / count_no_zeros * 100} %")

    print(" ============== LWP =================")
    arr = y_lwp
    count_no_zeros = np.count_nonzero(arr)
    print(f"number of non-zero: {count_no_zeros}")
    count_zeros = arr.size - np.count_nonzero(arr)
    print(f"number of zeros: {count_zeros}")
    print(f"percent of zeros: {count_zeros / count_no_zeros * 100} %")



    ####################3 ######

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(file_name, fontsize=10)

    plt.subplot(121)
    with plt.style.context('ggplot'):
        plt.scatter(x_Nd, y_lwp)  #
        plt.ylabel("Nd_max (cm-3) ", fontsize=16)
        plt.xlabel("lwp (gm-2)", fontsize=16)

    log_y_lwp = np.log(y_lwp + 1.0e-16)
    log_x_Nd = np.log(x_Nd + 1.0e-16)

    plt.subplot(122)
    with plt.style.context('ggplot'):
        plt.scatter(log_x_Nd, log_y_lwp)  #
        plt.xlabel("log Nd_max (cm-3) ", fontsize=16)
        plt.ylabel("log lwp (gm-2)", fontsize=16)

    figure_name = '{}_LWP_Nd_values.png'.format(os.sep.join([path_output, file_name]))

    fig.savefig(figure_name)

    plt.show()

    ##########################

    df = pd.DataFrame({'LWP': log_y_lwp,'Nd': log_x_Nd})


    count_nan_in_df = df.isnull().sum()
    print("================ values of Nan =====================")
    print(count_nan_in_df)  # can i used 0 in the lwp and nd?
    # print( f"count null: {count_nan_in_df} ")

    ##########################################
    df_2 = pd.DataFrame({'LWP': y_lwp, 'Nd': x_Nd})

    df_2.drop(df_2.loc[df_2['LWP']==0].index, inplace=True)
    print("================ after deleting the LWP =0  =====================")
    count_nan_in_df = df_2.isnull().sum()
    print("================ values of Nan =====================")
    print(count_nan_in_df)  # can i used 0 in the lwp and nd?
    # df =df_2
    count_zeros =  (df_2['LWP'] == 0).sum()
    count_nonzero = arr.size - count_zeros
    print(f"number of zeros lwp: {count_zeros}")
    print(f"number of non-zero LWP: {count_nonzero}")
    print(f"percent of zeros: {count_zeros / count_no_zeros * 100} %")

    count_zeros =  (df_2['Nd'] == 0).sum()
    count_nonzero = arr.size - count_zeros
    print(f"number of zeros Nd: {count_zeros}")
    print(f"number of non-zero LWP: {count_nonzero}")

    print(f"percent of zeros: {count_zeros / count_no_zeros * 100} %")


    # x = np.ma.masked_array(x_Nd, x_Nd == 0)
    # y = np.ma.masked_array(y_lwp, y_lwp == 0)
    df_2["LWP"] = np.log(df_2["LWP"])
    df_2["Nd"] = np.log(df_2["Nd"])
    df = df_2



    ##########################################
    # print(df.AI,df.Nd)
    nx = 30
    ny = 35

    jh, x_mid, y_mid, y_median = get_joint_histgram(nx, ny, np.log(5), np.log(1000), np.log(2), np.log(1000), df.Nd,
                                                    df.LWP)

    p0 = [max(y_median), np.median(x_mid), np.log(1000), min(y_median)]  # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x_mid, y_median, p0, method='dogbox')
    x0 = np.linspace(x_mid.min(), x_mid.max(), 1000)
    y0 = sigmoid(x0, *popt)

    # definitions for the axes
    left, width = 0.12, 0.85
    bottom, height = 0.3, 0.52
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom - 0.13 - spacing, width - 0.17, 0.13]

    parameters = {'axes.labelsize': 20,
                  'axes.titlesize': 35,
                  'xtick.labelsize': 14,
                  'ytick.labelsize': 14,
                  }
    plt.rcParams.update(parameters)

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)

    # no labels
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(direction='in', top=True, right=True)
    ax_histx.tick_params(direction='in', top=True, right=True)
    ax.set_ylabel('LWP (gm$\mathregular{^{-2}}$)')

    ax.set_yticks(np.log(np.array([2, 10, 50, 300, 1000])))
    ax.set_yticklabels(['2', '10', '50', '300', '1000'])

    ax_histx.set_xlabel('$\it{N}$$\mathregular{_d}$ (cm$\mathregular{^{-3}}$)')
    ax_histx.set_ylabel('PDF (%)')

    print("+++++print contourf x,y,jh", np.shape(x_mid), np.shape(y_mid), np.shape(jh))
    cs = ax.contourf(x_mid, y_mid, jh, cmap='viridis')  # Greys

    fig.colorbar(cs, ax=ax, label='PDF (%)')

    ax.scatter(x_mid, y_median, color='blue', marker='o', s=100)
    ax.plot(x0, y0, color='yellow', label='sigmoid fit', lw=2)

    # PDF plot
    dict_PDF = get_PDF_bin_range(df['Nd'], np.linspace(np.log(5), np.log(1000), nx + 1))
    ax_histx.plot(dict_PDF['x'], dict_PDF['pdf'], color='black', linewidth=3)
    ax_histx.fill_between(dict_PDF['x'], 0, dict_PDF['pdf'], facecolor='w', alpha=0.7, hatch='x')

    plt.xticks(np.log(np.array([5, 10, 50, 300, 1000])), np.array([5, 10, 50, 300, 1000]).astype(str))

    figure_name = '{}_relation_LWP-Nd-density-H.png'.format(os.sep.join([path_output, file_name]))

    fig.suptitle(file_name, fontsize=10)

    fig.savefig(figure_name)

    # fig.savefig(figure_name)
    plt.show()
    #

    # count_nan_in_df = df.isnull().sum()
    # print("================ values of Nan =====================")
    # print(count_nan_in_df)  # can i used 0 in the lwp and nd?
    # print(f"count null: {count_nan_in_df} ")


# ############# Nd_max(lat, lon) - Radiances
#     fig = plt.figure(figsize=(14, 4))
#     plt.subplot(121)
#     ds.lwp.where(ds.lwp != 0).plot(cmap="cividis", vmin=2, vmax=1200)
#     plt.subplot(122)
#     ds.Nd_max.where(ds.Nd_max != 0).plot(cmap="cividis", vmin=2, vmax=800)
#     figure_name = os.sep.join([args.path_out, file_name + '_LWP-Nd.png'])
#     print(figure_name)
#     fig.savefig(figure_name)
#
# ############# LWP(lat, lon) - Radiances
#
#     fig = plt.figure(figsize=(14, 4))
#     plt.subplot(121)
#     ds.Nd[119, :, :].where(ds.Nd[119, :, :] != 0).plot(cmap="cividis")  # ,vmin=2, vmax = 800)
#     plt.subplot(122)
#     # ds.Reff[119].where(ds.Reff != 0).plot(cmap = "jet",vmin=2, vmax = 800)
#     ds.Reff[119, :, ].plot(cmap="cividis", vmin=0, vmax=40)
#     figure_name = os.sep.join([args.path_out, file_name + '_Nd-Reff.png'])
#     fig.savefig(figure_name)
#     print(figure_name)
#     plt.show()
#     ds.close()

    # mask_array_lwp

######## kmeans ####################
    rttov_ds = xr.open_dataset(rttov_path).compute()  # write read rttov in a function

    rttov_variable = np.zeros((np.shape(rttov_ds['Radiance_total'].values)))

    rttov_variable[:19] = rttov_ds['BRF_total'][:19]  # refl 1-19, 26 rad 20-25 and 27-36
    rttov_variable[19:25] = rttov_ds['Radiance_total'][19:25]
    rttov_variable[25] = rttov_ds['BRF_total'][
        19]  # solo tengo en este archivo 1-19,26 luego tengo q hacer todo esto en un solo file
    rttov_variable[26:36] = rttov_ds['Radiance_total'][26:36]
    rttov_variable_ds = xr.DataArray(rttov_variable[:], dims=['Channel', 'Latitude', 'Longitude'],
                                     coords=[rttov_ds.Channel.data, rttov_ds.Latitude.data, rttov_ds.Longitude.data])
    bands=  rttov_variable_ds.Channel.values
    X_data = rttov_variable_ds.values

    #########################
    #     R = sds_data_radiances[0,:,]/ np.max(sds_data_radiances[0,:,:])
    #     G = sds_data_radiances[3,:,]/ np.max(sds_data_radiances[3,:,:])
    #     B = sds_data_radiances[2,:,]/ np.max(sds_data_radiances[2,:,:])

    R = X_data[0, :, ] / np.max(X_data[0, :, ])  # 1,3,4 = 0,1,2 then change for it when 0,2,3
    G = X_data[1, :, ] / np.max(X_data[3, :, ])
    B = X_data[2, :, ] / np.max(X_data[2, :, ])

    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)

    gamma = 2.2
    R = np.power(R, 1 / gamma)
    G = np.power(G, 1 / gamma)
    B = np.power(B, 1 / gamma)

    # Calculate the "True" Green
    # G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)

    # The final RGB array :)
    RGB = np.dstack([R, G_true, B])
    #############################################3
    ###### split train val ###########################
    #     #from sklearn.model_selection import train_test_split
    #     #df_train, df_test = train_test_split(X_scaled, test_size = .3, random_state = 42)
    #     #df_train_2, df_test_2 = train_test_split(MB_matrix_scaled, test_size = .3, random_state = 42)
    # X_train =  X_data[:,:,:400].transpose(1,2,0) #CON 400 ESTABA ELIMINANDO MUCHAS NUBES
    X_train = X_data[:, :, ].transpose (1, 2, 0)* mask_array_lwp.reshape((len(rttov_ds.Latitude.data), len(rttov_ds.Longitude.data), 1)) #('Latitude', 'Longitude','Channel')  #

    img_shape = np.shape(X_train)  # x,y,ch
    #     print(df_train.shape,img_shape,df_train.max()) ##########no seria adecuado serapar asi xq de ahi nose como obtener una imagen
    ###### flat ###########################
    name_file = 'radiances_output'
    df = dataframe_csv(variable=X_train, colum=bands, path_output=path_output, name_file=name_file)

    print("Before scaled -mean/std:", df.describe())

    ###### standard scaler ###########################\
    scaler = preprocessing.StandardScaler().fit(
        df)  # Standardize features by removing the mean and scaling to unit variance

    X_scaled = scaler.transform(df)
    # df_normalized=(df - df.mean()) / df.std()

    print("After scaled -mean/std:", pd.DataFrame(X_scaled).describe())

    # X_scaled= np.ma.masked_array(X_scaled, np.isnan(X_scaled))
    ###### analysis  PCA###########################
    name_plot = "PCA_variance"

    n_pca = args.n_pca  # 2 #test JQ
    n_bands = len(bands)  # 2 #test JQ

    # X_scaled =X_scaled[:,[0,25]] # test JQ

    X_reduced, pca = PCA_calculation(X_scaled, name_plot, n_pca, path_output)

    # PC_2d_Norm = convert_3D(X_reduced, img_shape, n_bands)
    PC_2d_Norm = from2to3d (X_reduced, img_shape, n_bands)

    # ic("PC_2d_Norm x,y,xh:", np.shape(PC_2d_Norm))
    plot_PC(PC_2d_Norm, n_bands, path_output)
    # Rearranging 1-d arrays to 2-d arrays of image size


    plot_image_PC(PC_2d_Norm=PC_2d_Norm, image_3bands=RGB, path_output=path_output)

    # X_reduced_test = pca.transform(scale(X_test))[:,:1]

    get_kmeans(nbands=n_bands, imagery=PC_2d_Norm, path_output=path_output)

    # n_clusters(nbands = n_bands, imagery = PC_2d_Norm, path_output = path_output)

    sys.stdout.close()

if __name__ == '__main__':
    main()
