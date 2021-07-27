
MODULE ML_CONFIG

  use ml_types, only: wp
  
  IMPLICIT NONE


  REAL(KIND=wp), parameter :: &
       apriori_iwp = 1E-1, &    ! in kg.m-2
       apriori_iwp_error = 10.0  ! in %
  
  !! Parallel options
  integer(kind=4) :: &
       RTTOV_NTHREADS = 35

  !! ML options
  REAL(KIND=wp), parameter :: &
       iwp_lay_threshold = 1E-4, &
       lwp_lay_threshold = 1E-4  
 
  INTEGER(KIND=4), parameter :: &
       nstates = 1
  
  !! ICON microphysics
  integer(kind=4), parameter :: Nhydro=9

  !! RTTOV optics
  integer(kind=4), parameter :: &
       !RTTOV_NCHANNELS = 2,  &
       !RTTOV_PLATFORM = 19,      &
       !RTTOV_SATELLITE = 1,     &
       !RTTOV_INSTRUMENT = 65,   &
       !RTTOV_DOSOLAR = 1

       RTTOV_NCHANNELS = 36,  &
       RTTOV_PLATFORM = 9,      &
       RTTOV_SATELLITE = 1,     &
       RTTOV_INSTRUMENT = 13,   &
       RTTOV_DOSOLAR = 1
       
  integer(kind=4), dimension(RTTOV_NCHANNELS), parameter :: &
       !RTTOV_CHANNEL_LIST = (/8,9/)
       RTTOV_CHANNEL_LIST = (/1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36/)

  
  !! Constants below
  
  real(wp) :: &  !! mixing ratios of trace gases
       mr_co2=5.241e-04_wp, &
       mr_ch4=9.139e-07_wp, &
       mr_n2o=4.665e-07_wp, &
       mr_co=2.098e-07_wp

  REAL(wp), PARAMETER :: &
       tmelt  = 273.15_wp,      & ! Melting temperature of ice/snow [K]
       rhoice = 917._wp,        & ! Density of ice [kg/m3]
       rholiq = 1000._wp          ! Density of liquid water [kg/m3]

  ! Molecular weights
  REAL(wp), PARAMETER :: &
       amw   = 18.01534_wp,     & ! Water   [g/mol]
       amd   = 28.9644_wp,      & ! Dry air [g/mol]
       amO3  = 47.9983_wp,      & ! Ozone   [g/mol]
       amCO2 = 44.0096_wp,      & ! CO2     [g/mol]
       amCH4 = 16.0426_wp,      & ! Methane [g/mol]
       amN2O = 44.0129_wp,      & ! N2O     [g/mol]
       amCO  = 28.0102_wp         ! CO      [g/mol]

  ! WMO/SI value
  REAL(wp), PARAMETER :: &
       avo   = 6.023E23_wp,     & ! Avogadro constant used by ISCCP simulator [1/mol]
       grav  = 9.806650_wp        ! Av. gravitational acceleration used by ISCCP simulator [m/s2]

  ! Thermodynamic constants for the dry and moist atmosphere
  REAL(wp), PARAMETER :: &
       rd  = 287.04_wp,         & ! Gas constant for dry air [J/K/Kg]
       cpd = 1004.64_wp,        & ! Specific heat at constant pressure for dry air [J/K/Kg]
       rv  = 461.51_wp,         & ! Gas constant for water vapor [J/K/Kg]
       cpv = 1869.46_wp,        & ! Specific heat at constant pressure for water vapor [J/K/Kg]
       km  = 1.38e-23_wp          ! Boltzmann constant [J/K]

END MODULE ML_CONFIG
