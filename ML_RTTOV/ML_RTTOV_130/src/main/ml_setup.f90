MODULE mod_ml_setup

  USE ml_types, only:type_ml, wp
  USE ml_config, only: nstates, apriori_iwp, rttov_nchannels
  
  IMPLICIT NONE

CONTAINS
  
  
  
  SUBROUTINE ml_update(idx_start,idx_end,oe,ml)
    
    ! Input variables
    INTEGER(kind=4), INTENT(IN) :: &
         idx_start, idx_end
    TYPE(type_ml), INTENT(IN) :: &
         oe
  
    ! Output variables
    TYPE(type_ml), INTENT(INOUT) :: &
         ml
    
    ml%y(idx_start:idx_end,:) = oe%y(:,:)
    ml%y_clear(idx_start:idx_end,:) = oe%y_clear(:,:)  
 
        !!!!!!!!!!!!!! JGV 
    ml%f(idx_start:idx_end,:) = oe%f(:,:)
    ml%f_clear(idx_start:idx_end,:) = oe%f_clear(:,:)   
     !!!!!!!!!!!!!! end JGV  

     
    ml%brdf(idx_start:idx_end,:) = oe%brdf(:,:)
    ml%Xip1(idx_start:idx_end,:) = oe%Xip1(:,:)
    ml%iwp_model(idx_start:idx_end) = oe%iwp_model(:)
    ml%Gip1(idx_start:idx_end) = oe%Gip1(:)            

  END SUBROUTINE ml_update


  SUBROUTINE ml_setup(idx_start,idx_end,nlevels,y,flag_oe)

    !INTEGER(kind=4), PARAMETER :: nchannels = rttov_nchannels
    !! I am working with 36 channesl    
    INTEGER(kind=4), PARAMETER :: nchannels = rttov_nchannels
    
    ! Input variables
    INTEGER(kind=4), INTENT(IN) :: &
         nlevels, idx_start, idx_end

    LOGICAL :: flag_oe
    
    ! Output variables
    TYPE(type_ml), INTENT(OUT) :: y

    ! Internal variables
    INTEGER(KIND=4) :: npoints, ipoint

    npoints = idx_end - idx_start + 1

    y%npoints = npoints

    y%nstates = nstates
    y%nmeas = rttov_nchannels     
    
    allocate(y%y(npoints,nchannels)); y%y = 0._wp
    allocate(y%f(npoints,nchannels)); y%f = 0._wp
    allocate(y%y_clear(npoints,nchannels)); y%y_clear = 0._wp
    allocate(y%f_clear(npoints,nchannels)); y%f_clear = 0._wp
    allocate(y%Xa(npoints,nstates)); y%Xa = 0._wp
    allocate(y%g(npoints)); y%g = 0._wp
    allocate(y%g_meas(npoints)); y%g_meas = 0._wp
    allocate(y%ztop_ice(npoints)); y%ztop_ice = 0._wp
    allocate(y%zbase_ice(npoints)); y%zbase_ice = 0._wp
    allocate(y%ztop_ice_idx(npoints)); y%ztop_ice_idx = 0
    allocate(y%zbase_ice_idx(npoints)); y%zbase_ice_idx = 0
    allocate(y%iwp(npoints)); y%iwp = 0._wp
    allocate(y%brdf(npoints,nchannels)); y%brdf = 0._wp
    allocate(y%iwp_model(npoints)); y%iwp_model = 0._wp
    allocate(y%xip1(npoints,nstates)); y%xip1 = 0._wp
    allocate(y%gip1(npoints)); y%gip1 = 0._wp
     
    IF(flag_oe) THEN
       allocate(y%iwc(npoints,nlevels)); y%iwc = 0._wp
       allocate(y%cla(npoints,nlevels)); y%cla = 0._wp
       allocate(y%xi(npoints,nstates)); y%xi = 0._wp
       allocate(y%gi(npoints)); y%gi = 0._wp
       allocate(y%stepsize(npoints)); y%stepsize = 0._wp
       allocate(y%i_stepsize(npoints)); y%i_stepsize = 0
       allocate(y%n_iter(npoints)); y%n_iter = 0       
       allocate(y%k(npoints,nstates,nchannels)); y%k = 0._wp
       allocate(y%kt(npoints,nstates,nchannels)); y%kt = 0._wp
       allocate(y%sy(npoints,nchannels,nchannels)); y%sy = 0._wp
       allocate(y%sf(npoints,nchannels,nchannels)); y%sf = 0._wp
       allocate(y%se(npoints,nchannels,nchannels)); y%se = 0._wp
       allocate(y%sx(npoints,nstates,nstates)); y%sx = 0._wp
       allocate(y%sa(npoints,nstates,nstates)); y%sa = 0._wp
       allocate(y%se_i(npoints,nchannels,nchannels)); y%se_i = 0._wp
       allocate(y%sx_i(npoints,nstates,nstates)); y%sx_i = 0._wp
       allocate(y%sa_i(npoints,nstates,nstates)); y%sa_i = 0._wp
       allocate(y%flag_rttov(npoints)); y%flag_rttov = .FALSE.
       allocate(y%flag_testconv(npoints)); y%flag_testconv = .FALSE.
    END IF

    y%Xa(:,1) = apriori_iwp

    
  END SUBROUTINE ml_setup


  
END MODULE mod_ml_setup
