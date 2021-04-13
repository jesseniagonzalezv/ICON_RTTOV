MODULE MOD_RTTOV

  USE ml_config, only:RTTOV_NCHANNELS
  USE ml_types, only: wp, type_rttov_atm, type_rttov_opt, type_ml
  USE mod_rttov_utils, only: idx_rttov
  
  ! rttov_const contains useful RTTOV constants
  USE rttov_const, ONLY :     &
       errorstatus_success, &
       errorstatus_fatal,   &
       platform_name,       &
       inst_name,           &
       surftype_sea,        &
       surftype_land,       &
       watertype_fresh_water, &
       watertype_ocean_water, &         
       sensor_id_mw,        &
       sensor_id_po, &
       wcl_id_stco, &
       wcl_id_stma


  USE rttov_types, ONLY :     &
         rttov_options,       &
         rttov_coefs,         &
         rttov_profile,       &
         rttov_transmission,  &
         rttov_radiance,      &
         rttov_chanprof,      &
         rttov_emissivity,    &
         rttov_reflectance,   &
         rttov_opt_param

  ! The rttov_emis_atlas_data type must be imported separately
  USE mod_rttov_emis_atlas, ONLY : &
        rttov_emis_atlas_data, &
        atlas_type_ir, atlas_type_mw

  ! The rttov_brdf_atlas_data type must be imported separately
  USE mod_rttov_brdf_atlas, ONLY : rttov_brdf_atlas_data
  
  USE rttov_unix_env, ONLY : rttov_exit  

  USE parkind1, ONLY : jpim, jprb, jplm
  
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

  INTEGER(KIND=jpim), PARAMETER :: iup   = 20   ! unit for input profile file
  INTEGER(KIND=jpim), PARAMETER :: ioout = 21   ! unit for output
  
  ! RTTOV variables/structures
  !====================
  TYPE(rttov_options)              :: opts                     ! Options structure
  TYPE(rttov_coefs)                :: coefs                    ! Coefficients structure
  TYPE(rttov_chanprof),    POINTER :: chanprof(:)    => NULL() ! Input channel/profile list
  LOGICAL(KIND=jplm),      POINTER :: calcemis(:)    => NULL() ! Flag to indicate calculation of emissivity within RTTOV
  TYPE(rttov_emissivity),  POINTER :: emissivity(:)  => NULL() ! Input/output surface emissivity
  LOGICAL(KIND=jplm),      POINTER :: calcrefl(:)    => NULL() ! Flag to indicate calculation of BRDF within RTTOV
  TYPE(rttov_reflectance), POINTER :: reflectance(:) => NULL() ! Input/output surface BRDF
  TYPE(rttov_profile),     POINTER :: profiles(:)    => NULL() ! Input profiles
  TYPE(rttov_transmission)         :: transmission             ! Output transmittances
  TYPE(rttov_radiance)             :: radiance                 ! Output radiances
  TYPE(rttov_opt_param)            :: cld_opt_param            ! Input cloud optical parameters
  
  TYPE(rttov_emis_atlas_data)      :: emis_atlas               ! Data structure for emissivity atlas
  TYPE(rttov_brdf_atlas_data)      :: brdf_atlas               ! Data structure for BRDF atlas
  
  INTEGER(KIND=jpim)               :: errorstatus              ! Return error status of RTTOV subroutine calls

  INTEGER(KIND=jpim) :: atlas_type  
  INTEGER(KIND=jpim) :: alloc_status
  CHARACTER(LEN=11)  :: NameOfRoutine = 'example_fwd'

  ! variables for input
  !====================
  CHARACTER(LEN=256) :: coef_filename
  CHARACTER(LEN=256) :: prof_filename, cld_coef_filename
  INTEGER(KIND=jpim) :: nthreads
  INTEGER(KIND=jpim) :: dosolar
  INTEGER(KIND=jpim) :: nlevels
  INTEGER(KIND=jpim) :: nprof
  INTEGER(KIND=jpim) :: nchannels
  INTEGER(KIND=jpim) :: nchanprof
  INTEGER(KIND=jpim), ALLOCATABLE :: channel_list(:)
  REAL(KIND=jprb)    :: trans_out(10)
  ! loop variables
  INTEGER(KIND=jpim) :: j, jch
  INTEGER(KIND=jpim) :: np, nch
  INTEGER(KIND=jpim) :: ilev, nprint
  INTEGER(KIND=jpim) :: iprof, joff, ichan
  INTEGER            :: ios, imonth

  ! Initialization parameters
  integer :: &
       platform,   & ! RTTOV platform
       sensor,     & ! RTTOV instrument
       satellite
  
  
CONTAINS


  ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ! SUBROUTINE rttov_column
  ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  subroutine run_rttov(rttov_atm,rttov_opt,oe,dealloc)

    use ml_config, only: rd
    
    ! Inputs variables
    type(type_rttov_atm), intent(in) :: rttov_atm
    type(type_rttov_opt), intent(in) :: rttov_opt
    
    logical,intent(in) :: &
         dealloc   ! Flag to determine whether to deallocate RTTOV types

    ! Inout/Outputs variables
    type(type_ml), intent(inout) :: oe    
    
    ! Local variables
    integer :: &
         errorstatus, alloc_status, idx_prof, ilevel
    integer, dimension(:), allocatable :: list_points

    
    errorstatus = 0_jpim

    nthreads= 35
    
    list_points =  idx_rttov(oe)    
    nprof=size(list_points); nlevels=rttov_atm%nlevels  

    oe%f(list_points,:) = 0._wp
    oe%f_clear(list_points,:) = 0._wp
    oe%brdf(list_points,:) = 0._wp

    if(nprof.EQ.0) RETURN
    
    ! --------------------------------------------------------------------------
    ! 3. Allocate RTTOV input and output structures
    ! --------------------------------------------------------------------------

    ! Determine the total number of radiances to simulate (nchanprof).
    ! In this example we simulate all specified channels for each profile, but
    ! in general one can simulate a different number of channels for each profile.

    nchanprof = nchannels * nprof

    ! Allocate structures for rttov_direct
    CALL rttov_alloc_direct( &
         errorstatus,             &
         1_jpim,                  &  ! 1 => allocate
         nprof,                   &
         nchanprof,               &
         nlevels,                 &
         chanprof,                &
         opts,                    &
         profiles,                &
         coefs,                   &
         transmission,            &
         radiance,                &
         calcemis=calcemis,       &
         emissivity=emissivity,   &
         calcrefl=calcrefl,       &
         reflectance=reflectance, &
         init=.TRUE._jplm)
    IF (errorstatus /= errorstatus_success) THEN
       WRITE(*,*) 'allocation error for rttov_direct structures'
       CALL rttov_exit(errorstatus)
    ENDIF

    
    ! --------------------------------------------------------------------------
    ! 4. Build the list of profile/channel indices in chanprof
    ! --------------------------------------------------------------------------

    nch = 0_jpim
    DO j = 1, nprof
       DO jch = 1, nchannels
          nch = nch + 1_jpim
          chanprof(nch)%prof = j
          chanprof(nch)%chan = channel_list(jch)
       ENDDO
    ENDDO


    ! --------------------------------------------------------------------------
    ! 5. Read profile data
    ! --------------------------------------------------------------------------

    profiles(:) % gas_units = 1 !! tmp, check this, is necessary for water in kg/kg?

    ! Loop over all profiles and read data for each one
    DO iprof = 1, nprof
       idx_prof = list_points(iprof)
       
       profiles(iprof) % p = rttov_atm % p(idx_prof,:)*1E-2 ! change to hPa
       profiles(iprof) % t = rttov_atm % t(idx_prof,:)
       profiles(iprof) % q(:) = 0._wp !rttov_atm % q(idx_prof,:) ! I left kg/kg, automatic choice in RRTOV?

       profiles(iprof) % s2m % t = rttov_atm % t2m(idx_prof)
       profiles(iprof) % s2m % q = 0._wp !rttov_atm % q2m(idx_prof)
       profiles(iprof) % s2m % p = rttov_atm % p_surf(idx_prof)*1E-2 ! change to hPa
       profiles(iprof) % s2m % u = rttov_atm % u_surf(idx_prof)
       profiles(iprof) % s2m % v = rttov_atm % v_surf(idx_prof)
       profiles(iprof) % s2m % wfetc = 100000 !! used typical value given in documentation

       profiles(iprof) % skin % t = rttov_atm % t_skin(idx_prof)
       profiles(iprof) % skin % salinity = 0.0 !! tmp, use other typical value
       profiles(iprof) % skin % fastem = (/3.0, 5.0, 15.0, 0.1, 0.3/) !! tmp, typical for land, adjust

       if (rttov_atm%lsmask(iprof) < 0.5) then
          profiles(iprof)%skin%surftype  = surftype_sea
       else
          profiles(iprof)%skin%surftype  = surftype_land
       endif
       profiles(iprof) %skin % watertype = watertype_fresh_water !! tmp, adapt this to truth, fresh more likely for ICON-DE simulations

       profiles(iprof) % elevation = rttov_atm % h_surf(idx_prof)*1E-3 !! tmp, this is set to be 0, why? The elevation is an input of the COSP
       profiles(iprof) % latitude = rttov_atm % lat(idx_prof)
       profiles(iprof) % longitude = rttov_atm % lon(idx_prof)

       profiles(iprof) % zenangle = rttov_opt % zenangle 
       profiles(iprof) % azangle = rttov_opt % azangle
       profiles(iprof) % sunzenangle = rttov_opt % sunzenangle
       profiles(iprof) % sunazangle = rttov_opt % sunazangle

       profiles(iprof) % mmr_cldaer = .FALSE.
       profiles(iprof) % ice_scheme = 2  !! Use baran

       profiles(iprof) % cfrac = rttov_atm % tca(idx_prof,:)

       if (rttov_atm%lsmask(iprof) < 0.5) then
          profiles(iprof) % cloud(wcl_id_stco,:) = rttov_atm % lwc(idx_prof,:) * 1E3
       else
          profiles(iprof) % cloud(wcl_id_stma,:) = rttov_atm % lwc(idx_prof,:) * 1E3
       end if
       profiles(iprof) % cloud(6,:) = rttov_atm % iwc(idx_prof,:) * 1E3
    ENDDO
    
    ! --------------------------------------------------------------------------
    ! 6. Specify surface emissivity and reflectance
    ! --------------------------------------------------------------------------

    ! ! In this example we have no values for input emissivity
    ! emissivity(:) % emis_in = 0._jprb
    
    ! Use emissivity atlas
    CALL rttov_get_emis(             &
         errorstatus,           &
         opts,                  &
         chanprof,              &
         profiles,              &
         coefs,                 &
         emis_atlas,            &
         emissivity(:) % emis_in)
    IF (errorstatus /= errorstatus_success) THEN
       WRITE(*,*) 'error reading emissivity atlas'
       CALL rttov_exit(errorstatus)
    ENDIF

    ! Calculate emissivity within RTTOV where the atlas emissivity value is
    ! zero or less
    calcemis(:) = (emissivity(:) % emis_in <= 0._jprb)

    IF (opts % rt_ir % addsolar) THEN

       ! ! In this example we have no values for input reflectances
       ! reflectance(:) % refl_in = 0._jprb
       
       ! Use BRDF atlas
       CALL rttov_get_brdf(              &
            errorstatus,            &
            opts,                   &
            chanprof,               &
            profiles,               &
            coefs,                  &
            brdf_atlas,             &
            reflectance(:) % refl_in)
       IF (errorstatus /= errorstatus_success) THEN
          WRITE(*,*) 'error reading BRDF atlas'
          CALL rttov_exit(errorstatus)
       ENDIF

       ! Calculate BRDF within RTTOV where the atlas BRDF value is zero or less
       calcrefl(:) = (reflectance(:) % refl_in <= 0._jprb)

    ENDIF

    ! Use the RTTOV emissivity and BRDF calculations over sea surfaces
    DO j = 1, SIZE(chanprof)
       IF (profiles(chanprof(j)%prof) % skin % surftype == surftype_sea) THEN
          calcemis(j) = .TRUE.
          calcrefl(j) = .TRUE.
       ENDIF
    ENDDO

    ! Use default cloud top BRDF for simple cloud in VIS/NIR channels
    reflectance(:) % refl_cloud_top = 0._jprb

    ! --------------------------------------------------------------------------
    ! 7. Call RTTOV forward model
    ! --------------------------------------------------------------------------
    IF (nthreads <= 1) THEN
       CALL rttov_direct(                &
            errorstatus,              &! out   error flag
            chanprof,                 &! in    channel and profile index structure
            opts,                     &! in    options structure
            profiles,                 &! in    profile array
            coefs,                    &! in    coefficients structure
            transmission,             &! inout computed transmittances
            radiance,                 &! inout computed radiances
            calcemis    = calcemis,   &! in    flag for internal emissivity calcs
            emissivity  = emissivity, &! inout input/output emissivities per channel
            calcrefl    = calcrefl,   &! in    flag for internal BRDF calcs
            reflectance = reflectance) ! inout input/output BRDFs per channel
    ELSE
       CALL rttov_parallel_direct(     &
            errorstatus,              &! out   error flag
            chanprof,                 &! in    channel and profile index structure
            opts,                     &! in    options structure
            profiles,                 &! in    profile array
            coefs,                    &! in    coefficients structure
            transmission,             &! inout computed transmittances
            radiance,                 &! inout computed radiances
            calcemis    = calcemis,   &! in    flag for internal emissivity calcs
            emissivity  = emissivity, &! inout input/output emissivities per channel
            calcrefl    = calcrefl,   &! in    flag for internal BRDF calcs
            reflectance = reflectance,&! inout input/output BRDFs per channel
            nthreads    = nthreads)    ! in    number of threads to use
    ENDIF

    IF (errorstatus /= errorstatus_success) THEN
       WRITE (*,*) 'rttov_direct error'
       CALL rttov_exit(errorstatus)
    ENDIF

    ! --- Output the results --------------------------------------------------
    DO iprof = 1, nprof
       idx_prof = list_points(iprof)
       
       joff = (iprof-1_jpim) * nchannels
       ichan=1
       DO j=1+joff, nchannels+joff
          if(coefs%coef%ss_val_chn(channel_list(ichan))==0) then
             oe%f(idx_prof,ichan) = radiance % bt(j)
             oe%f_clear(idx_prof,ichan) = radiance % bt_clear(j)           
          else
             oe%f(idx_prof,ichan) = radiance % refl(j)
             oe%f_clear(idx_prof,ichan) = radiance % refl_clear(j)           
          end if
          oe%brdf(idx_prof,ichan) = reflectance(j) % refl_out
          ! write(*,*) oe%brdf(iprof,ichan)
          ichan=ichan+1
       END DO
    END DO
    
    ! --- End of output section -----------------------------------------------

    ! --------------------------------------------------------------------------
    ! 8. Deallocate all RTTOV arrays and structures
    ! --------------------------------------------------------------------------

    ! Deallocate structures for rttov_direct
    CALL rttov_alloc_direct( &
         errorstatus,             &
         0_jpim,                  &  ! 0 => deallocate
         nprof,                   &
         nchanprof,               &
         nlevels,                 &
         chanprof,                &
         opts,                    &
         profiles,                &
         coefs,                   &
         transmission,            &
         radiance,                &
         calcemis=calcemis,       &
         emissivity=emissivity,   &
         calcrefl=calcrefl,       &
         reflectance=reflectance)
    IF (errorstatus /= errorstatus_success) THEN
       WRITE(*,*) 'deallocation error for rttov_direct structures'
       CALL rttov_exit(errorstatus)
    ENDIF

    IF (dealloc) THEN
       CALL rttov_dealloc_coefs(errorstatus, coefs)
       IF (errorstatus /= errorstatus_success) THEN
          WRITE(*,*) 'coefs deallocation error'
       ENDIF

       ! Deallocate emissivity atlas
       CALL rttov_deallocate_emis_atlas(emis_atlas)

       IF (opts % rt_ir % addsolar) THEN
          ! Deallocate BRDF atlas
          CALL rttov_deallocate_brdf_atlas(brdf_atlas)
       ENDIF

    END IF





  end subroutine run_rttov

  
END MODULE MOD_RTTOV
