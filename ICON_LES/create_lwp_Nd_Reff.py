

from scipy.special import gamma
import os
import argparse
import xarray as xr
import numpy as np

def get_lwp_Nd(path_icon):
    """
    Arguments:
    path_icon -- ncfile with the ICON dataset  ('/work/bb1036/b381362/dataset/data_rttov_T12.nc')

    Returns:
    new netcdf file with the variables in the same folder as the input data but with the extension of _Reff      
    ["tas", "t_s", "ps", "huss", "u_10m", "v_10m", "topography_c", "FR_LAND", "pres", "ta", "hus",  "clc", "clw", "cli", "Reff", "lwp", "Nd_max"]
    Variables created
     lwp -- liquid water path (gm-2)
     Nd_max -- cloud droplet number concentration maximum on the height values (cm-3)
    """
    fname = path_icon
    name_output = os.path.splitext(fname)[0] + "_Reff.nc"

    ds = xr.open_dataset(fname)
    T_c = np.float64(ds.ta) - 273.15
    esat_2013 = 0.611 * np.exp((17.3 * T_c) / (T_c + 237.3)) * 1000.0
    # esat_2013 = np.ma.masked_array(esat_2013,  esat_2013 == 0) ## check it!!!!!!!!
    pres = np.ma.masked_array(ds.pres, ds.pres == 0)  ## check it!!!!!!!!
    qs_2013 = 0.622 * (esat_2013 / pres)  # this is diffent compared with Alexandre code
    r_2013 = ds.hus / (1 - ds.hus)
    RH_2013 = 100 * (r_2013 / qs_2013)
    pv_2013 = (esat_2013 * RH_2013) / 100.0
    pd_2013 = ds.pres - pv_2013
    rho_2013 = (pd_2013 / (287.058 * ds.ta)) + (pv_2013 / (461.495 * ds.ta))  # nana
    cdnc_2013_cm = (rho_2013 * ds.qnc) / 1000000  # convert to cm^-3

    Nd_max = np.nanmax(cdnc_2013_cm, axis=0)
    ds["Nd_max"] = (['lat', 'lon'], Nd_max)  # this is an array
    ds.Nd_max.attrs['units'] = "cm-3"
    ds.Nd_max.attrs['standard_name'] = "Nd_max"
    ds.Nd_max.attrs['long_name'] = "Cloud dropler number maximun"

    ds["Nd"] = cdnc_2013_cm  # thi is a xarray.DataArray
    ds.Nd.attrs['units'] = "cm-3"
    ds.Nd.attrs['standard_name'] = "Nd"
    ds.Nd.attrs['long_name'] = "Cloud dropler number in each layer"


    
    ds = ds.assign(lwp=ds.clwvi * 1000)
    ds.lwp.attrs['units'] = "gm-2"
    ds.lwp.attrs['standard_name'] = "LWP"
    ds.lwp.attrs['long_name'] = "Liquid water path"

    

    # -----------------Calculation of the Reff -------------------------
    L = rho_2013 * ds.clw  # in kgm^-3 #cannot set variable with 3-dimensional data without explicit dimension names. Pass a tuple of (dims, data) instead.

    N = rho_2013 * ds.qnc  # im m^-3
    N2 = np.ma.masked_array(N, N < 2.0e+06)  ## check it!!!!!!!! 2cm'3

    # L.where(L == 0, np.Nan, L)
    #N.where(N < 2.0e+06, np.NAN, N) # !! ask DIPU
    ######constant for size distribution #############
    nu = 1.0
    mu = 1.0
    a = 1.24E-01
    b = 1 / 3

    ###################################
    reff_2013 = (a / 2) * (gamma((3 * b + nu + 1) / (mu)) / gamma((2 * b + nu + 1) / (mu))) * (
                (L / N2) * (gamma((nu + 1) / (mu)) / gamma((nu + 2) / mu))) ** (b)  # m
    reff_2013 = reff_2013 * 1E6

    ds["Reff"] = reff_2013  # thi is a xarray.DataArray
    ds.Reff.attrs['units'] = "Micron"
    ds.Reff.attrs['standard_name'] = "Reff"
    ds.Reff.attrs['long_name'] = "Cloud effective radius"

    # print(
    #     "next values only work with data_rttov_12 and with lat 57 is before cutting otherwise 57-9=48 (i eliminate the buttom 9 pixels of lat)")
    # print('===============T_2013 (height 120, lat 57, lon 227) cm: 276.151153564453 == ', ds.ta[119, 56 - 9, 226])
    # print('===============T_c (height 120, lat 57, lon 227) cm: 3.00115356445315== ', T_c[119, 56 - 9, 226])
    # print('===============esat_2013 (height 120, lat 57, lon 227) 758.360598313415 == ',
    #       esat_2013[119, 56 - 9, 226])
    # print('===============p_2013 (height 120, lat 57, lon 227)75935.328125 == ', ds.pres[119, 56 - 9, 226])
    # print('===============qs_2013 (height 120, lat 57, lon 227) 0.00621186875461262  == ',
    #       qs_2013[119, 56 - 9, 226])
    # print('===============r_2013 (height 120, lat 57, lon 227) 0.0062973626597643 == ', r_2013[119, 56 - 9, 226])
    # print('===============RH_2013 (height 120, lat 57, lon 227) 101.376299283339 == ', RH_2013[119, 56 - 9, 226])
    # print('===============pv_2013 (height 120, lat 57, lon 227) 768.797909793127 == ', pv_2013[119, 56 - 9, 226])
    # print('===============pd_2013 (height 120, lat 57, lon 227) 75166.5302152069 == ', pd_2013[119, 56 - 9, 226])
    # print('===============rho_2013 (height 120, lat 57, lon 227) 0.954250058491486 == ', rho_2013[119, 56 - 9, 226])
    # print('===============cdnc_2013_cm (height 120, lat 57, lon 227) 15.5508091629487 == ', ds.Nd[119, 56 - 9, 226])
    # print('===============Reff (height 120, lat 57, lon 227) 17.20 um == ', ds.Reff[119, 56 - 9, 226])

    print("Reff min, max", reff_2013.min().values, np.max(reff_2013).values)

    variable_2D = ["tas", "t_s", "ps", "huss", "u_10m", "v_10m", "topography_c", "FR_LAND"]

    variable_3D = ["pres", "ta", "hus",  "clc", "clw", "cli"]
    
#    variable_calculated = ["Reff", "lwp", "Nd_max"]
    variable_calculated = ["Reff", "lwp", "Nd_max"]

    variables_total = variable_2D + variable_3D + variable_calculated

    ds = ds.get(variables_total)
    
    ds.to_netcdf(name_output)  # '/work/bb1036/b381362/dataset/3D_mod.nc') # rewrite to netcdf
    print("generated the next file:", name_output)
        
    ds.close()
    #return ds #["lwp"], ds["Nd_max"]
    
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--path-in', type=str, default='/work/bb1036/rttov_share/dataset_ICON/icon_input_germany_02052013_T13.nc', help='path of the netcdf with the ICON_LES simulations')

    args = parser.parse_args()
    path_in = args.path_in
    
    get_lwp_Nd(path_in)

if __name__ == '__main__':
    main()

