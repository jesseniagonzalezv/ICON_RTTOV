MODULE MOD_UTILS_FORT

  IMPLICIT NONE
  
CONTAINS

  !! In case of error when running the program (inspired from COSPv2)
  SUBROUTINE ML_ERROR(routine_name,message,errcode) 
    character(len = *), intent(in) :: routine_name
    character(len = *), intent(in) :: message
    integer,optional :: errcode
    
    write(6, *) " ********** Failure in ", trim(routine_name)
    write(6, *) " ********** ", trim(message)
    if (present(errcode)) write(6, *) " ********** errcode: ", errcode
    flush(6)
    stop
  END SUBROUTINE ML_ERROR

  

END MODULE MOD_UTILS_FORT
