MODULE MOD_OE_UTILS

  USE ml_types, only: wp, type_rttov_atm, type_rttov_opt, type_ml
  USE mod_model_cloud, only: init_cloudprof
  USE mod_rttov, only: run_rttov
  USE ml_config, only: rttov_nchannels, nstates, apriori_iwp_error
  use mod_utils_math, only: inverse
  
  IMPLICIT NONE

CONTAINS

  
  FUNCTION idx_ice(oe,flag_ice)

    TYPE(type_ml), INTENT(IN) :: oe
    
    REAL(KIND=wp), DIMENSION(:), ALLOCATABLE :: idx_ice
    REAL(KIND=wp), DIMENSION(oe%npoints) :: idx_all
    INTEGER(KIND=4) :: idx, i
    LOGICAL, INTENT(IN) :: flag_ice
    LOGICAL :: test
    
    idx_all = 0
    idx=1
    DO i=1,oe%npoints

       IF(flag_ice) THEN
          test = oe%iwp(i).GT.1E-4
       ELSE
          test = oe%iwp(i).LT.1E-4
       END IF
       
       IF(test) THEN
          idx_all(idx) = i
          idx = idx + 1
       END IF
    END DO
    idx=idx-1
    
    ALLOCATE(idx_ice(idx)); idx_ice = idx_all(1:idx)
    
    RETURN
    
  END FUNCTION idx_ice

  
  REAL(KIND=wp) FUNCTION oe_cost(oe,ipoint)
    TYPE(type_ml), INTENT(IN) :: oe
    INTEGER(KIND=4), INTENT(IN) :: ipoint

    oe_cost = oe_cost_meas(oe,ipoint) + oe_cost_state(oe,ipoint)

    RETURN
  END FUNCTION oe_cost

  REAL(KIND=wp) FUNCTION oe_cost_meas(oe,ipoint)
    TYPE(type_ml), INTENT(IN) :: oe
    INTEGER(KIND=4), INTENT(IN) :: ipoint

    oe_cost_meas = DOT_PRODUCT(oe%Y(ipoint,:) - oe%F(ipoint,:), MATMUL(oe%Se_i(ipoint,:,:),oe%Y(ipoint,:) - oe%F(ipoint,:)))

    RETURN
  END FUNCTION oe_cost_meas

  REAL(KIND=wp) FUNCTION oe_cost_state(oe,ipoint)
    TYPE(type_ml), INTENT(IN) :: oe
    INTEGER(KIND=4), INTENT(IN) :: ipoint

    oe_cost_state = DOT_PRODUCT(oe%Xi(ipoint,:) - oe%Xa(ipoint,:),MATMUL(oe%Sa_i(ipoint,:,:),oe%Xi(ipoint,:) - oe%Xa(ipoint,:)))

    RETURN
  END FUNCTION oe_cost_state
  
  
  FUNCTION oe_Sx(oe,ipoint)

    REAL(KIND=wp), DIMENSION(nstates,nstates) :: oe_Sx, Sx_i
    
    TYPE(type_ml), INTENT(IN) :: oe
    INTEGER(KIND=4), INTENT(IN) :: ipoint

    Sx_i(:,:) = (1.0_wp + oe%stepsize(ipoint)) * oe%Sa_i(ipoint,:,:) + MATMUL(oe%Kt(ipoint,:,:),MATMUL(oe%Se_i(ipoint,:,:),oe%K(ipoint,:,:)))
    CALL inverse(Sx_i, oe_Sx, nstates)
    
    RETURN
  END FUNCTION oe_Sx

  FUNCTION oe_Xip1(oe,ipoint)

    REAL(KIND=wp), DIMENSION(nstates) :: oe_Xip1
    
    TYPE(type_ml), INTENT(IN) :: oe
    INTEGER(KIND=4), INTENT(IN) :: ipoint
    
    oe_Xip1(:) = oe%Xi(ipoint,:) + MATMUL(oe%Sx(ipoint,:,:),(MATMUL(oe%Kt(ipoint,:,:),MATMUL(oe%Se_i(ipoint,:,:),(oe%Y(ipoint,:)-oe%F(ipoint,:)))) &
         - MATMUL(oe%Sa_i(ipoint,:,:),(oe%Xi(ipoint,:)-oe%Xa(ipoint,:)))))
    
    RETURN
  END FUNCTION oe_Xip1
  

  

  SUBROUTINE get_jacobian(rttov_atm,rttov_opt,oe)

    USE mod_rttov_utils, only: idx_rttov

    ! Parameters
    REAL(kind=wp), PARAMETER :: diwp=1E-4

    ! Input variables
    TYPE(type_rttov_opt), INTENT(IN) :: rttov_opt
    TYPE(type_rttov_atm), INTENT(IN) :: rttov_atm
    TYPE(type_ml), INTENT(INOUT) :: oe

    ! Local variables
    TYPE(type_ml) :: oe_pert
    TYPE(type_rttov_atm) :: rttov_atm_pert

    REAL(kind=wp), DIMENSION(rttov_atm%nPoints,rttov_opt%nchannels) :: Fp1, Fm1
    INTEGER(KIND=4) :: ipoint
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: idx_oe
    
    oe_pert = oe; rttov_atm_pert = rttov_atm

    idx_oe = idx_rttov(oe)
    
    oe_pert%iwp(idx_oe) = oe%iwp(idx_oe) + diwp
    call init_cloudprof(rttov_atm_pert,oe_pert)
    call run_rttov(rttov_atm_pert,rttov_opt,oe_pert,dealloc=.FALSE.)
    Fp1 = oe_pert%F

    oe_pert%iwp(idx_oe) = oe%iwp(idx_oe) - diwp    
    call init_cloudprof(rttov_atm_pert,oe_pert)
    call run_rttov(rttov_atm_pert,rttov_opt,oe_pert,dealloc=.FALSE.)
    Fm1 = oe_pert%F

    DO ipoint=1,oe_pert%npoints
       if(oe%flag_rttov(ipoint)) oe%K(ipoint,1,:) = (Fp1(ipoint,:) - Fm1(ipoint,:))/(2.0_wp*diwp)
    END DO

  END SUBROUTINE get_jacobian


  SUBROUTINE get_covmat(oe)

    ! Input variables
    TYPE(type_ml), INTENT(INOUT) :: oe

    INTEGER(KIND=4) :: &
         ichannel, istate

    DO ichannel=1,rttov_nchannels
       oe%Sy(:,ichannel,ichannel) = (oe%y(:,ichannel) * 1E-2)**2
    END DO

    DO istate=1,nstates
       oe%Sa(:,istate,istate) = ( oe%Xa(:,istate) * apriori_iwp_error )**2
    END DO

    oe%Se = oe%Sy

  END SUBROUTINE get_covmat


  SUBROUTINE update_oemat(rttov_atm,rttov_opt,oe)

    ! Input variables
    TYPE(type_rttov_opt), INTENT(IN) :: rttov_opt
    TYPE(type_rttov_atm), INTENT(IN) :: rttov_atm
    TYPE(type_ml), INTENT(INOUT) :: oe

    ! Local variables
    TYPE(type_rttov_atm) :: rttov_atm_new

    rttov_atm_new = rttov_atm

    call init_cloudprof(rttov_atm_new,oe)
    call run_rttov(rttov_atm_new,rttov_opt,oe,.FALSE.)
    call get_jacobian(rttov_atm_new,rttov_opt,oe)
    call get_covmat(oe)
    
  END SUBROUTINE update_oemat


  SUBROUTINE update_oe(oe_in,oe_ip1,nidx,idx_oe,rttov_atm,rttov_opt)

    ! Input variables
    INTEGER(KIND=4), INTENT(IN) :: nidx
    INTEGER(KIND=4), DIMENSION(nidx), INTENT(IN) :: idx_oe
    TYPE(type_rttov_opt), INTENT(IN) :: rttov_opt

    TYPE(type_ml), INTENT(IN) :: oe_in
    TYPE(type_rttov_atm), INTENT(INOUT) :: rttov_atm
    
    ! Local variables
    TYPE(type_ml), INTENT(OUT) :: oe_ip1
    TYPE(type_ml) :: oe
    INTEGER(KIND=4) :: ipoint

    oe = oe_in
    
    ! First run of optimal estimation, update Xi
    DO ipoint=1,size(idx_oe)
       oe%Sx(idx_oe(ipoint),:,:) = oe_Sx(oe,idx_oe(ipoint))
       oe%Xip1(idx_oe(ipoint),:) = oe_Xip1(oe,idx_oe(ipoint))
    END DO

    ! Check if Xi is physically consistent, else change stepsize and update again
    DO ipoint=1,size(idx_oe)
       DO WHILE(oe%Xip1(idx_oe(ipoint),1).LT.1.0E-4)
          oe%stepsize(idx_oe(ipoint)) = oe%stepsize(idx_oe(ipoint)) * 2.0D0
          oe%i_stepsize(idx_oe(ipoint)) = oe%i_stepsize(idx_oe(ipoint)) + 1

          oe%Sx(idx_oe(ipoint),:,:) = oe_Sx(oe,idx_oe(ipoint))
          oe%Xip1(idx_oe(ipoint),:) = oe_Xip1(oe,idx_oe(ipoint))
       END DO
    END DO

    oe_ip1=oe
    oe_ip1%iwp(idx_oe) = oe%Xip1(idx_oe,1)
    call update_oemat(rttov_atm,rttov_opt,oe_ip1)

    DO ipoint=1,size(idx_oe)
       oe_ip1%Gip1(idx_oe(ipoint)) = oe_cost(oe_ip1,idx_oe(ipoint))
    END DO
    oe%Gip1 = oe_ip1%Gip1
    
  END SUBROUTINE update_oe
  


  ! SUBROUTINE copy_oe(oe_in,oe_out,nidx,idx_oe)

  !   TYPE(type_ml), INTENT(IN) :: oe_in
  !   TYPE(type_ml), INTENT(OUT) :: oe_out
  !   INTEGER(KIND=4), INTENT(IN) :: nidx
  !   INTEGER(KIND=4), DIMENSION(nidx), INTENT(IN) :: idx_oe

  !   TYPE(type_ml) :: oe_tmp

  !   oe_out = oe_in
    
    

  !   ml%y(npoints,nchannels),ml%f(npoints,nchannels),ml%y_clear(npoints,nchannels),ml%f_clear(npoints,nchannels)
  !   ml%x(npoints,nstates),ml%xa(npoints,nstates), ml%g(npoints), ml%g_meas(npoints)
  !   ml%ztop_ice(npoints),ml%zbase_ice(npoints),ml%ztop_ice_idx(npoints),ml%zbase_ice_idx(npoints)
  !   ml%ztop_liq(npoints),ml%zbase_liq(npoints),ml%ztop_liq_idx(npoints),ml%zbase_liq_idx(npoints)
  !   ml%iwp(npoints),ml%lwp(npoints),ml%brdf(npoints,nchannels)
  !   ml%xi(npoints,nstates),ml%xip1(npoints,nstates), ml%gi(npoints),ml%gip1(npoints),ml%stepsize(npoints)
  !   ml%iloop_stepsize(npoints), ml%iloop_iter(npoints)
  !   ml%k(npoints,nstates,nchannels), ml%kt(npoints,nstates,nchannels)
  !   ml%sy(npoints,nchannels,nchannels), ml%sf(npoints,nchannels,nchannels)
  !   ml%se(npoints,nchannels,nchannels), ml%sx(npoints,nstates,nstates), ml%sa(npoints,nstates,nstates)
  !   ml%se_i(npoints,nchannels,nchannels), ml%sx_i(npoints,nstates,nstates), ml%sa_i(npoints,nstates,nstates)
  !   ml%idx_orig(npoints),ml%flag_rttov(npoints),ml%flag_stepsize(npoints)    
    
   
  ! END SUBROUTINE copy_oe
    

END MODULE MOD_OE_UTILS
