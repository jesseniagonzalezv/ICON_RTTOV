PROGRAM ML_RTTOV

  USE ml_types, ONLY: wp, type_rttov_atm, type_rttov_opt, type_icon, type_ml
  USE ml_config, ONLY: nhydro
  
  USE mod_read_icon, ONLY: read_icon, construct_icon, map_point_to_ll
  USE mod_rttov_interface, ONLY: rttov_init
  USE mod_rttov_setup, ONLY: rttov_setup_opt, rttov_setup_atm
  USE mod_rttov, ONLY: run_rttov
  USE mod_ml_setup, ONLY: ml_setup, ml_update
  USE mod_model_cloud, ONLY: init_zcloud, init_cloudprof
  USE mod_write_output, ONLY: write_output
  USE mod_oe_utils, ONLY: idx_ice
  USE mod_rttov_utils, ONLY: idx_rttov
  USE mod_oe_run, ONLY: oe_run
  
  IMPLICIT NONE

  ! Parameters (for now)
  CHARACTER(LEN=256), PARAMETER :: &
       fname_in= "/work/bb1036/b381362/dataset/data_rttov_T12.nc", & ! "/work/bb1036/b381362/dataset/subset_rttov_T12.nc""/work/bb1036/b381362/dataset/data_rttov_T12.nc"  

       fname_out="/work/bb1036/b381362/output/VF-data_rttov_T13.nc"   !"/work/bb1036/b381362/output/VF-subtest_rttov_T13.nc"  "/work/bb1036/b381362/output/"  output-data_rttov_T12.nc
  INTEGER(KIND=4), PARAMETER :: Nlevels=150, Npoints= 375193 , npoints_it=2000 ! 5369 375193
  LOGICAL, PARAMETER :: flag_ret = .FALSE.
  
  ! Local variables
  TYPE(type_icon), TARGET :: icon
  TYPE(type_rttov_atm) :: rttov_atm, rttov_atm_oe
  TYPE(type_rttov_opt) :: rttov_opt
  TYPE(type_ml) :: ml, oe, oe_ip1, oe_tmp

  REAL(KIND=wp) ::  zenangle,azangle,sunzenangle,sunazangle
  
  INTEGER(KIND=4) :: i, loc, j
  INTEGER(KIND=4) :: nchunks, idx_start, idx_end, nPtsPerIt, ichunk
  INTEGER(KIND=4), TARGET :: month, nidx  
  INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: idx_iwp, idx_oe
  
  LOGICAL :: flag_oe, dealloc_rttov

  zenangle=0._wp; azangle=0._wp
  sunzenangle=30._wp; sunazangle=0._wp
  month = 5 !4
  
  ! Construct the icon pointer
  CALL construct_icon(npoints,nlevels,nhydro,icon)

  ! Read input netcdf file containing ICON outputs
  CALL read_icon(fname_in,icon)
  
  ! Setup the RTTOV optics
  CALL rttov_setup_opt(zenangle,azangle,sunzenangle,sunazangle,month,rttov_opt)
  
  ! Initialize RTTOV (load data)
  CALL rttov_init(rttov_opt)

  ! Setup the ML variables (kept for output)
  flag_oe = .FALSE.
  CALL ml_setup(1,icon%npoints,icon%nlevels,ml,flag_oe)
  
  nChunks = icon%nPoints/nPoints_it
  IF(MOD(icon%npoints,npoints_it)/=0) nchunks = nchunks + 1  
  IF (icon%nPoints .EQ. nPoints_it) nChunks = 1  

  flag_oe = .TRUE.
  DO iChunk=1,nChunks
     WRITE(6,*) ichunk, "/", nchunks

     IF (nChunks .EQ. 1) THEN
        idx_start = 1; idx_end   = nPoints
     ELSE
        idx_start = (iChunk-1)*nPoints_it+1; idx_end   = iChunk*nPoints_it
        if (idx_end .GT. nPoints) idx_end=nPoints
     END IF
     
     ! Subset the atmosphere based on the icon file
     CALL rttov_setup_atm(idx_start,idx_end,icon,rttov_atm)
     CALL rttov_setup_atm(idx_start,idx_end,icon,rttov_atm_oe)     

     ! Setup the oe variables
     CALL ml_setup(rttov_atm%idx_start,rttov_atm%idx_end,icon%nlevels,oe,flag_oe)
     
     IF(flag_ret) THEN

        ! Extract the cloud position for ice and liquid phase and modify rttov_atm
        CALL init_zcloud(rttov_atm_oe,oe)

        ! Set conditions to use RTTOV (now: need an ice cloud)
        idx_iwp = idx_ice(oe,.TRUE.)
        oe%flag_rttov(idx_iwp) = .TRUE.

        idx_oe = idx_rttov(oe)
        IF(SIZE(idx_oe).EQ.0) THEN
           CYCLE
        END IF

        ! Update the cloud profiles in rttov_atm_oe using parameters from oe 
        CALL init_cloudprof(rttov_atm_oe,oe)      
        
        ! Run rttov on the selected atmosphere
        CALL run_rttov(rttov_atm_oe,rttov_opt,oe,dealloc=.FALSE.)

     ELSE
        oe%flag_rttov(:) = .TRUE.

        CALL run_rttov(rttov_atm,rttov_opt,oe,dealloc=.FALSE.)            !!!  comente     
     END IF
                
     !oe%y = oe%f; oe%y_clear = oe%f_clear     yo cambie esto to print tota,clear and reflectances no unit
     oe%iwp_model = oe%iwp
     
     IF(flag_ret) CALL oe_run(oe,rttov_atm_oe,rttov_opt)
     
     ! Update the arrays saved for output file (ml)
     CALL ml_update(idx_start,idx_end,oe,ml)
     
  END DO

  ! Write output file
  CALL write_output(fname_out,icon,ml)
  
END PROGRAM ML_RTTOV



