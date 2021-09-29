MODULE MOD_RTTOV_INTERFACE

  USE ml_config, only: RTTOV_NTHREADS

  USE ml_types, only: wp, type_rttov_opt

  USE rttov_const, ONLY :     &
       surftype_sea,        &
       surftype_land,       &
       sensor_id_mw,        &
       sensor_id_po,        &
       inst_name,           &
       platform_name

  use mod_rttov, only: platform,satellite,sensor,nChannels, &
       opts, errorstatus_success, rttov_exit, coefs, &
       emis_atlas, brdf_atlas, atlas_type, dosolar, nthreads, imonth, &
       dosolar, channel_list

  ! The rttov_emis_atlas_data type must be imported separately
  USE mod_rttov_emis_atlas, ONLY : &
       rttov_emis_atlas_data, &
       atlas_type_ir, atlas_type_mw

  ! The rttov_brdf_atlas_data type must be imported separately
  USE mod_rttov_brdf_atlas, ONLY : rttov_brdf_atlas_data

  IMPLICIT NONE

#include "rttov_direct.interface"
#include "rttov_parallel_direct.interface"
#include "rttov_read_coefs.interface"
#include "rttov_dealloc_coefs.interface"
#include "rttov_alloc_direct.interface"
#include "rttov_user_options_checkinput.interface"
#include "rttov_print_opts.interface"
#include "rttov_print_profile.interface"
#include "rttov_skipcommentline.interface"

  ! Use emissivity atlas
#include "rttov_setup_emis_atlas.interface"
#include "rttov_get_emis.interface"
#include "rttov_deallocate_emis_atlas.interface"

  ! Use BRDF atlas
#include "rttov_setup_brdf_atlas.interface"
#include "rttov_get_brdf.interface"
#include "rttov_deallocate_brdf_atlas.interface"

  
CONTAINS

  SUBROUTINE RTTOV_INIT(rttov_opt)

    type(type_rttov_opt), intent(in) :: rttov_opt
    
    ! Local variables
    character(len=256) :: coef_filename, cld_coef_filename, sat
    integer :: errorstatus

    nthreads= RTTOV_NTHREADS
    
    imonth=rttov_opt%month
    dosolar = rttov_opt%dosolar
    nChannels  = rttov_opt%nchannels
    allocate(channel_list(nchannels))
    channel_list(1:nchannels)=rttov_opt%channel_list
    
    sat="_"
    IF(rttov_opt%satellite.NE.0) THEN
       write(sat,*) rttov_opt%satellite
       sat="_"//trim(adjustl(sat))//"_"
    END IF
    
    coef_filename = "/work/bb1036/b381362/tools/rttov130/rtcoef_rttov13/rttov13pred54L/rtcoef_"//&
         trim(platform_name(rttov_opt%platform))//trim(sat)//trim(inst_name(rttov_opt%instrument))//"_o3.dat"
    cld_coef_filename = "/work/bb1036/b381362/tools/rttov130/rtcoef_rttov13/cldaer_visir/sccldcoef_"//&
         trim(platform_name(rttov_opt%platform))//trim(sat)//trim(inst_name(rttov_opt%instrument))//".dat"    

    
    ! --------------------------------------------------------------------------
    ! 1. Initialise RTTOV options structure
    ! --------------------------------------------------------------------------
    IF (dosolar == 1) THEN
       opts % rt_ir % addsolar = .TRUE.           ! Include solar radiation
    ELSE
       opts % rt_ir % addsolar = .FALSE.          ! Do not include solar radiation
    ENDIF
    opts % interpolation % addinterp   = .TRUE.  ! Allow interpolation of input profile
    opts % interpolation % interp_mode = 1       ! Set interpolation method
    !!    opts % interpolation % reg_limit_extrap = .TRUE.

    opts % rt_all % addrefrac          = .TRUE.  ! Include refraction in path calc
    opts % rt_ir % addaerosl           = .FALSE. ! Don't include aerosol effects
    opts % rt_ir % addclouds           = .TRUE.  ! Don't include cloud effects

    opts % rt_ir % ir_scatt_model      = 2!2 ! 1=DOM ODran estaba 1
    opts % rt_ir % vis_scatt_model     = 2!2 ! 1=DOM Odran estaba 1 
    opts % rt_ir % dom_nstreams        = 8  

   ! opts % rt_ir % ozone_data          = .FALSE. ! Set the relevant flag to .TRUE.
   ! opts % rt_ir % co2_data            = .FALSE. !   when supplying a profile of the
   ! opts % rt_ir % n2o_data            = .FALSE. !   given trace gas (ensure the
   ! opts % rt_ir % ch4_data            = .FALSE. !   coef file supports the gas)
   ! opts % rt_ir % co_data             = .FALSE. !
   ! opts % rt_ir % so2_data            = .FALSE. !
   ! opts % rt_mw % clw_data            = .FALSE.

    opts % config % verbose            = .FALSE.  ! Enable printing of warnings
    opts % config % do_checkinput      = .FALSE.

    opts%rt_all%switchrad = .TRUE.


    ! --------------------------------------------------------------------------
    ! 2. Read coefficients
    ! --------------------------------------------------------------------------
    CALL rttov_read_coefs(errorstatus, coefs, opts, file_coef=coef_filename, &
         file_sccld=cld_coef_filename)
    IF (errorstatus /= errorstatus_success) THEN
       WRITE(*,*) 'fatal error reading coefficients'
       CALL rttov_exit(errorstatus)
    ENDIF

    ! Ensure input number of channels is not higher than number stored in coefficient file
    IF (nchannels > coefs % coef % fmv_chn) THEN
       nchannels = coefs % coef % fmv_chn
    ENDIF

    ! Ensure the options and coefficients are consistent
    CALL rttov_user_options_checkinput(errorstatus, opts, coefs)
    IF (errorstatus /= errorstatus_success) THEN
       WRITE(*,*) 'error in rttov options'
       CALL rttov_exit(errorstatus)
    ENDIF
    

    ! Initialise the RTTOV emissivity atlas
    ! (this loads the default IR/MW atlases: use the atlas_id argument to select alternative atlases)
    IF (coefs%coef%id_sensor == sensor_id_mw .OR. &
         coefs%coef%id_sensor == sensor_id_po) THEN
       atlas_type = atlas_type_mw ! MW atlas
    ELSE
       atlas_type = atlas_type_ir ! IR atlas
    ENDIF
    CALL rttov_setup_emis_atlas(          &
         errorstatus,              &
         opts,                     &
         imonth,                   &
         atlas_type,               & ! Selects MW (1) or IR (2)
         emis_atlas,               &
         path = '/work/bb1036/b381362/tools/rttov130/emis_data', & ! The default path to atlas data
         coefs = coefs) ! This is mandatory for the CNRM MW atlas, ignored by TELSEM2;
    ! if supplied for IR atlases they are initialised for this sensor
    ! and this makes the atlas much faster to access
    IF (errorstatus /= errorstatus_success) THEN
       WRITE(*,*) 'error initialising emissivity atlas'
       CALL rttov_exit(errorstatus)
    ENDIF

    IF (opts % rt_ir % addsolar) THEN

       ! Initialise the RTTOV BRDF atlas
       CALL rttov_setup_brdf_atlas(        &
            errorstatus,            &
            opts,                   &
            imonth,                 &
            brdf_atlas,             &
            path='/work/bb1036/b381362/tools/rttov130/brdf_data', &  ! The default path to atlas data
            coefs = coefs) ! If supplied the BRDF atlas is initialised for this sensor and
       ! this makes the atlas much faster to access
       IF (errorstatus /= errorstatus_success) THEN
          WRITE(*,*) 'error initialising BRDF atlas'
          CALL rttov_exit(errorstatus)
       ENDIF

    ENDIF

    
    
  END SUBROUTINE RTTOV_INIT 



END MODULE  MOD_RTTOV_INTERFACE
