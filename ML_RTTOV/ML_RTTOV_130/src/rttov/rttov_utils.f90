MODULE mod_rttov_utils

  USE ml_types, only: wp, type_ml
  
  IMPLICIT NONE

CONTAINS


  FUNCTION idx_rttov(oe)

    TYPE(type_ml), INTENT(IN) :: oe
    
    INTEGER(KIND=4), DIMENSION(:), ALLOCATABLE :: idx_rttov
    INTEGER(KIND=4), DIMENSION(oe%npoints) :: idx_all
    INTEGER(KIND=4) :: idx, i

    idx_all = 0
    idx=1
    DO i=1,oe%npoints
       IF(oe%flag_rttov(i)) THEN
          idx_all(idx) = i
          idx = idx + 1
       END IF
    END DO
    idx=idx-1

    ALLOCATE(idx_rttov(idx)); idx_rttov(1:idx) = idx_all(1:idx)

    RETURN
    
  END FUNCTION idx_rttov

END MODULE mod_rttov_utils
