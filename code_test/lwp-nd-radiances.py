import xarray as xr
import argparse
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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
    arg('--path-ICON', type=str, default='/work/bb1036/b381362/dataset/data_rttov_T12_dropupbottom_Reff.nc', help='path of the initial data is')
    # arg('--path-ICON', type=str, default='/home/jvillarreal/Documents/phd/dataset/data_rttov_T12_dropupbottom_Reff.nc',
        # help='path of the initial data is')
    # arg('--path-in-rttov', type=str, default='/home/jvillarreal/Documents/phd/dataset/data_rttov_T12_dropupbottom_Reff.nc',
    #     help='path of the initial data is')
    arg('--path-out', type=str, default='/home/b/b381362/output/output_ICON', help='path of the copied data is' )

    # arg('--path-out', type=str, default='/home/jvillarreal/Documents/phd/output/output_ICON',
        # help='path of the copied data is')

    

    args = parser.parse_args()
    fname = args.path_ICON
    file_name = os.path.splitext(os.path.basename(fname))[0][:-5]  # os.path.splitext(fname)[0][:-5] with path
    path_output = args.path_out
    ds = xr.open_dataset(fname)

    print(" ##################### FILE: {} #####################".format(file_name))

    var_Nd = ds.Nd_max[:, :].values # lat,lot
    var_LWP = ds.lwp[:, :].values # lat,lot

    x_Nd = var_Nd.flatten()
    y_lwp = var_LWP.flatten()

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
    figure_name = '{}_LWP_Nd.png'.format(os.sep.join([path_output,file_name]))


    plt.show()
    fig.savefig(figure_name)
    
    #####################2 ###
    fig = plt.figure(figsize=(14, 4))
    fig.suptitle(file_name, fontsize=10)

    plt.subplot(121)
    ds.lwp.where(ds.lwp != 0).plot(cmap = "cividis",vmin=2, vmax = 1200)
    plt.subplot(122)
    ds.Nd_max.where(ds.Nd_max != 0).plot(cmap = "cividis",vmin=2, vmax = 800)
    figure_name = os.sep.join([args.path_out,file_name +'_LWP2-1200-Nd2-800.png'])

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
    print( f"percent of zeros: { count_zeros/count_no_zeros*100} %")

    print(" ============== LWP =================")
    arr = y_lwp
    count_no_zeros = np.count_nonzero(arr)
    count_zeros = arr.size - np.count_nonzero(arr)

    print(f"number of non-zero: {count_no_zeros}")
    print(f"number of zeros: {count_zeros}")
    print( f"percent of zeros: { count_zeros/count_no_zeros*100} %")
    
    ######################################## dataframe ######################################## 
    df_2 = pd.DataFrame({'LWP': y_lwp, 'Nd': x_Nd})
    df_2.drop(df_2.loc[df_2['LWP']==0].index, inplace=True)
    print("================ after deleting the LWP =0  =====================")
    count_zeros =  (df_2['LWP'] == 0).sum()
    count_nonzero = arr.size - count_zeros
    print(f"number of zeros lwp: {count_zeros}")
    # print(f"number of non-zero LWP: {count_nonzero}")
    print(f"percent of zeros: {count_zeros / count_no_zeros * 100} %")

    count_zeros =  (df_2['Nd'] == 0).sum()
    count_nonzero = arr.size - count_zeros
    print(f"number of zeros Nd: {count_zeros}")
    print(f"percent of zeros: {count_zeros / count_no_zeros * 100} %")
    ######################################## end dataframe ######################################## 

    
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
        plt.scatter(log_x_Nd , log_y_lwp)  #
        plt.xlabel("log Nd_max (cm-3) ", fontsize=16)
        plt.ylabel("log lwp (gm-2)", fontsize=16)
    
    figure_name = '{}_LWP_Nd_values.png'.format(os.sep.join([path_output,file_name]))

    fig.savefig(figure_name)

    plt.show()

    ##########################

    # df = pd.DataFrame({'LWP': log_y_lwp,'Nd': log_x_Nd})

    x = np.ma.masked_array(x_Nd, x_Nd == 0)
    y = np.ma.masked_array(y_lwp, y_lwp == 0)
    df = pd.DataFrame({'LWP': np.log(np.float_(y)), 'Nd': np.log(np.float_(x))})

    count_nan_in_df = df.isnull().sum()
    print("================ values of nan =====================")
    print (count_nan_in_df)  #can i used 0 in the lwp and nd?
    # print( f"count null: {count_nan_in_df} ")


    # print(df.AI,df.Nd)
    nx = 30
    ny = 35



    jh, x_mid, y_mid, y_median = get_joint_histgram(nx,ny,np.log(5), np.log(1000), np.log(2), np.log(1000),df.Nd,df.LWP)

    p0 = [max(y_median), np.median(x_mid), np.log(1000), min(y_median)]  # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x_mid, y_median, p0, method='dogbox')
    x0 = np.linspace(x_mid.min(), x_mid.max(), 1000)
    y0 = sigmoid(x0, *popt)


    # definitions for the axes
    left, width = 0.12, 0.85
    bottom, height = 0.3, 0.52
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom - 0.13 - spacing, width-0.17, 0.13]


    parameters = {'axes.labelsize': 20,
                  'axes.titlesize': 35,
                  'xtick.labelsize':14,
                  'ytick.labelsize':14,
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
    cs = ax.contourf(x_mid, y_mid, jh, cmap='viridis') #Greys

    fig.colorbar(cs, ax=ax, label='PDF (%)')


    ax.scatter(x_mid, y_median, color='blue', marker='o', s=100)
    ax.plot(x0,y0,color='yellow', label='sigmoid fit', lw=2)



    #PDF plot
    dict_PDF = get_PDF_bin_range(df['Nd'], np.linspace(np.log(5), np.log(1000), nx+1))
    ax_histx.plot(dict_PDF['x'], dict_PDF['pdf'], color='black', linewidth=3)
    ax_histx.fill_between(dict_PDF['x'], 0, dict_PDF['pdf'], facecolor='w', alpha=0.7, hatch = 'x')

    plt.xticks(np.log(np.array([5, 10, 50, 300,1000])),np.array([5, 10, 50, 300,1000]).astype(str))

    figure_name = '{}_relation_LWP-Nd-density-H.png'.format(os.sep.join([path_output,file_name]))
    
    fig.suptitle(file_name, fontsize=10)

    fig.savefig(figure_name)

    # fig.savefig(figure_name)
    plt.show()
#

    # count_nan_in_df = df.isnull().sum()
    # print("================ values of Nan =====================")
    # print (count_nan_in_df)  #can i used 0 in the lwp and nd?
    # print( f"count null: {count_nan_in_df} ")

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


if __name__ == '__main__':
    main()
