MODULE ml_types

  IMPLICIT NONE

  ! Few kind definitions for variables
  integer, parameter :: sp = selected_real_kind(6, 37)
  integer, parameter :: dp = selected_real_kind(12, 307)
  integer, parameter :: wp = sp

  ! Type containing variables used by ML for retrievals
  TYPE type_ml
     INTEGER(kind=4) :: &
          nstates, &
          nmeas,   &
          npoints
     INTEGER(kind=4), DIMENSION(:), ALLOCATABLE :: &
          ztop_ice_idx,    &
          zbase_ice_idx,   &
          n_iter, i_stepsize
     REAL(kind=wp), DIMENSION(:), ALLOCATABLE :: &
          g, gi, gip1, g_meas, stepsize
     LOGICAL, DIMENSION(:), ALLOCATABLE :: &
          flag_rttov, flag_testconv
     REAL(kind=wp), DIMENSION(:,:), ALLOCATABLE :: &
          Y, &
          Y_clear, &          
          F, &
          F_clear, &                    
          Xi, &
          Xip1, &          
          Xa, &
          brdf
     REAL(kind=wp), DIMENSION(:,:,:), ALLOCATABLE :: &
          K, &
          Kt, &          
          Sy, &
          Sf, &
          Se, &
          Se_i, &          
          Sx, &
          Sx_i, &
          Sa, &                    
          Sa_i
     REAL(kind=wp), DIMENSION(:,:), ALLOCATABLE :: &
          iwc,             & ! Output measurement vector
          cla
     REAL(kind=wp), DIMENSION(:), ALLOCATABLE :: &
          ztop_ice,    &
          zbase_ice,   &
          iwp,         &
          iwp_model
      
  END TYPE type_ml

  
  ! Type containing variables from ICON simulations
  type type_icon
     integer(kind=4) :: &
          Nlevels, Npoints, Nhydro, Nlat, Nlon, mode! Dimensions 
     real(wp) :: &
          emsfc_lw,     &
          co2,          & ! Carbon dioxide 
          ch4,          & ! Methane 
          n2o,          & ! n2o 
          co              ! Carbon monoxide

     real(wp),dimension(:),allocatable:: &
          lon_orig,  & ! Longitude that won't be regridded (deg)
          lat_orig,  & ! Latitude  that won't be regridded (deg)
          lon,       & ! Longitude (deg)
          lat,       & ! Latitude (deg)
          skt,       & ! Skin temperature (K)
          psfc,      & ! Surface pressure (Pa)
          landmask,  & ! Land/sea mask (0/1)
          seaice,    & ! sea ice mask (0/1)
          orography, & ! surface height (I think?)
          u_wind,    & ! U-component of wind (m/s)
          v_wind,    & ! V-component of wind (m/s)
          sunlit       ! Sunlit flag
     real(wp),dimension(:,:),allocatable :: &
          p,         & ! Model pressure levels (pa)
          p_lev,     & ! Model pressure levels (pa)          
          ph,        & ! Moddel pressure @ half levels (pa)
          zlev,      & ! Model level height (m)
          zlev_half, & ! Model level height @ half-levels (m)
          dz,        & ! Layer thickness (m)
          T,         & ! Temperature (K)
          T_lev,     & ! Temperature (K)          
          sh,        & ! Specific humidity (kg/kg)
          sh_lev,    & ! Specific humidity (kg/kg)          
          rh,        & ! Relative humidity (1)
          tca,       & ! Total cloud fraction (1)
          cca,       & ! Convective cloud fraction (1) 
          mr_lsliq,  & ! Mass mixing ratio for stratiform cloud liquid (kg/kg)
          mr_lsice,  & ! Mass mixing ratio for stratiform cloud ice (kg/kg)
          mr_ccliq,  & ! Mass mixing ratio for convective cloud liquid (kg/kg)
          mr_ccice,  & ! Mass mixing ratio for convective cloud ice (kg/kg)
          mr_ozone,  & ! Mass mixing ratio for ozone (kg/kg)
          mr_ozone_lev,  & ! Mass mixing ratio for ozone (kg/kg)          
          fl_lsrain, & ! Precipitation flux (rain) for stratiform cloud (kg/m^2/s)
          fl_lssnow, & ! Precipitation flux (snow) for stratiform cloud (kg/m^2/s)
          fl_lsgrpl, & ! Precipitation flux (groupel) for stratiform cloud (kg/m^2/s)
          fl_ccrain, & ! Precipitation flux (rain) for convective cloud (kg/m^2/s)
          fl_ccsnow, & ! Precipitation flux (snow) for convective cloud (kg/m^2/s)
          dtau_s,    & ! 0.67micron optical depth (stratiform cloud) (1)
          dtau_c,    & ! 0.67micron optical depth (convective cloud) (1)
          dem_s,     & ! 11micron emissivity (stratiform cloud) 
          dem_c,     & ! 11microm emissivity (convective cloud)
          lwc,       & ! Liquid water content (kg/m3)
          iwc          ! Ice water content (kg/m3)
     
     real(wp),dimension(:,:,:),allocatable :: &
          Reff         ! Subcolumn effective radius
  end type type_icon

  type type_rttov_atm
     integer,pointer :: &
          idx_start,    & ! Index of starting ICON point
          idx_end,      & ! Index of ending ICON point
          nPoints,      & ! Number of profiles to simulate
          nLevels         ! Number of levels
     real(wp),pointer :: &
          co2,          & ! Carbon dioxide 
          ch4,          & ! Methane 
          n2o,          & ! n2o 
          co              ! Carbon monoxide
     real(wp),dimension(:),pointer :: &
          surfem          ! Surface emissivities for the channels
     real(wp),dimension(:),pointer :: &
          h_surf,       & ! Surface height
          u_surf,       & ! U component of surface wind
          v_surf,       & ! V component of surface wind
          t_skin,       & ! Surface skin temperature
          p_surf,       & ! Surface pressure
          t2m,          & ! 2 m Temperature
          q2m,          & ! 2 m Specific humidity
          lsmask,       & ! land-sea mask
          lat,          & ! Latitude
          lon,          & ! Longitude
          seaice          ! Sea-ice? 
     real(wp),dimension(:,:),pointer :: &
          z,            & ! Height @ model levels
          dz,           & ! Layer depth @ model levels           
          p,            & ! Pressure @ model levels
          ph,           & ! Pressure @ model half levels
          t,            & ! Temperature 
          q,            & ! Specific humidity
          o3              ! Ozone
     ! These fields below are needed ONLY for the RTTOV all-sky brightness temperature
     real(wp),dimension(:,:),pointer :: &
          tca,          & ! Cloud fraction
          iwc,          & ! Cloud ice
          lwc,          & ! Cloud liquid
          fl_rain,      & ! Precipitation flux (startiform+convective rain) (kg/m2/s)
          fl_snow         ! Precipitation flux (stratiform+convective snow)     
  end type type_rttov_atm


  type type_rttov_opt
     
     integer ::     &
          dosolar,          &
          nchannels,        &
          platform,         &
          satellite,        &
          instrument,       &
          month                ! Month (needed for surface emissivity calculation)
     integer,dimension(:),allocatable :: &
          channel_list
     real(wp) :: &
          zenangle,azangle,sunzenangle,sunazangle
  end type type_rttov_opt
  

  
END MODULE ml_types
