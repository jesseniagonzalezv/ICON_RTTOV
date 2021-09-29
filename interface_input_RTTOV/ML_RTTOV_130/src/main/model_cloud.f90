MODULE mod_model_cloud

  use ml_types, only: type_rttov_atm, type_ml
  use ml_config, only: wp, iwp_lay_threshold, lwp_lay_threshold
  
  IMPLICIT NONE

CONTAINS

  SUBROUTINE init_zcloud(rttov,oe)

    TYPE(type_rttov_atm), INTENT(IN) :: rttov
    TYPE(type_ml), INTENT(INOUT) :: oe

    INTEGER(KIND=4) :: ipoint, ilevel
    
    oe % ztop_ice(:) = 0._wp; oe % zbase_ice(:) = 0._wp
    oe % iwp (:) = 0._wp
    oe % ztop_ice_idx(:) = 0; oe % zbase_ice_idx(:) = 0
    
    DO ipoint=1,rttov%npoints

       DO ilevel=1,rttov%nlevels
          if(rttov%iwc(ipoint,ilevel)*rttov%dz(ipoint,ilevel)*1E3 .GT. iwp_lay_threshold) then
             oe % ztop_ice_idx(ipoint) = ilevel
             exit
          end if
       END DO

       DO ilevel=rttov%nlevels,1,-1
          if(rttov%iwc(ipoint,ilevel)*rttov%dz(ipoint,ilevel)*1E3 .GT. iwp_lay_threshold) then
             oe % zbase_ice_idx(ipoint) = ilevel
             exit
          end if
       END DO

       if(oe % ztop_ice_idx(ipoint)/=0 .AND. oe % ztop_ice_idx(ipoint)/=0) then
          oe % ztop_ice(ipoint) =  rttov%z(ipoint,oe % ztop_ice_idx(ipoint))
          oe % zbase_ice(ipoint) =  rttov%z(ipoint,oe % zbase_ice_idx(ipoint) + 1) ! 1 is added to take the top altitude of the lower layer
          oe % iwp(ipoint) = SUM(rttov%iwc(ipoint,:) * rttov%dz(ipoint,:))
          ! oe % lwp(ipoint) = SUM(rttov%lwc(ipoint,:) * rttov%dz(ipoint,:))
       end if

       IF(oe % iwp(ipoint).LT.1E-4) THEN
          oe % iwp(ipoint) = 0._wp
       END IF
       
    END DO

  END SUBROUTINE init_zcloud


  SUBROUTINE init_cloudprof(rttov,oe)

    USE mod_rttov_utils, only: idx_rttov
    
    TYPE(type_rttov_atm), INTENT(INOUT) :: rttov
    TYPE(type_ml), TARGET, INTENT(INOUT) :: oe

    INTEGER(KIND=4) :: ipoint, ilevel, idx
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: idx_oe

    idx_oe = idx_rttov(oe)    

    oe%cla = rttov%tca
    
    DO ipoint=1,size(idx_oe)
       idx = idx_oe(ipoint)
      
       oe % iwc(idx,:) = 0._wp
       
       if(oe%iwp(idx) .GT. 1E-4) then
          oe%iwc(idx,oe%ztop_ice_idx(idx):oe%zbase_ice_idx(idx)) = oe%iwp(idx) / SUM(rttov%dz(idx,oe%ztop_ice_idx(idx):oe%zbase_ice_idx(idx)))
          oe%cla(idx,oe%ztop_ice_idx(idx):oe%zbase_ice_idx(idx)) = 1.0_wp
       end if
    END DO

    rttov%iwc => oe%iwc
    rttov%tca => oe%cla
    
  END SUBROUTINE init_cloudprof


END MODULE mod_model_cloud
