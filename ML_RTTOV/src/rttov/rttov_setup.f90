MODULE MOD_RTTOV_SETUP

  use ml_types, only: wp
  
  IMPLICIT NONE

CONTAINS

  SUBROUTINE RTTOV_SETUP_OPT(zenangle,azangle,sunzenangle,sunazangle,month,rttov_opt)

    use ml_types, only: type_rttov_opt
    
    use ml_config, only: &
         RTTOV_PLATFORM, RTTOV_PLATFORM, RTTOV_SATELLITE, &
         RTTOV_INSTRUMENT, RTTOV_DOSOLAR, RTTOV_NCHANNELS, &
         RTTOV_CHANNEL_LIST

    ! Input variables
    integer, intent(in) :: month
    real(wp), intent(in) :: zenangle,azangle,sunzenangle,sunazangle
    
    ! Output variables
    type(type_rttov_opt), intent(out) :: rttov_opt
    
    rttov_opt%platform = RTTOV_PLATFORM
    rttov_opt%satellite = RTTOV_SATELLITE
    rttov_opt%instrument = RTTOV_INSTRUMENT
    rttov_opt%dosolar = RTTOV_DOSOLAR
    rttov_opt%nchannels= RTTOV_NCHANNELS
    allocate(rttov_opt%channel_list(rttov_opt%nchannels))
    rttov_opt%channel_list=RTTOV_CHANNEL_LIST   

    rttov_opt%month = month

    rttov_opt % zenangle = zenangle
    rttov_opt % azangle = azangle
    rttov_opt % sunzenangle = sunzenangle
    rttov_opt % sunazangle = sunazangle
    
  END SUBROUTINE RTTOV_SETUP_OPT


  SUBROUTINE rttov_setup_atm(idx_start,idx_end,icon,rttov_atm)

    use ml_types, only: type_rttov_atm, type_icon
    
    ! Input variables
    integer, target, intent(in) :: idx_start, idx_end
    type(type_icon), target, intent(in) :: icon
    
    ! Output variables
    type(type_rttov_atm) :: rttov_atm

    integer, save, target :: nidx
    
    nidx = idx_end - idx_start + 1
    
    rttov_atm%nPoints    => nidx
    rttov_atm%nLevels    => icon%nLevels
    rttov_atm%idx_start  => idx_start
    rttov_atm%idx_end    => idx_end    
    rttov_atm%co2        => icon%co2
    rttov_atm%ch4        => icon%ch4
    rttov_atm%n2o        => icon%n2o
    rttov_atm%co         => icon%co
    rttov_atm%h_surf     => icon%orography(idx_start:idx_end)
    rttov_atm%u_surf     => icon%u_wind(idx_start:idx_end)
    rttov_atm%v_surf     => icon%v_wind(idx_start:idx_end)
    rttov_atm%t_skin     => icon%skt(idx_start:idx_end)
    rttov_atm%p_surf     => icon%psfc(idx_start:idx_end)
    rttov_atm%q2m        => icon%sh_lev(idx_start:idx_end,icon%Nlevels+1)
    rttov_atm%t2m        => icon%skt(idx_start:idx_end)
    rttov_atm%lsmask     => icon%landmask(idx_start:idx_end)
    rttov_atm%lat        => icon%lat(idx_start:idx_end)
    rttov_atm%lon        => icon%lon(idx_start:idx_end)
    rttov_atm%seaice     => icon%seaice(idx_start:idx_end)
    rttov_atm%p          => icon%p_lev(idx_start:idx_end,:)
    rttov_atm%z          => icon%zlev(idx_start:idx_end,:)
    rttov_atm%dz         => icon%dz(idx_start:idx_end,:)        
    rttov_atm%t          => icon%T_lev(idx_start:idx_end,:)
    rttov_atm%q          => icon%sh_lev(idx_start:idx_end,:)
    rttov_atm%o3         => icon%mr_ozone_lev(idx_start:idx_end,:)
    rttov_atm%tca        => icon%tca(idx_start:idx_end,:)
    rttov_atm%iwc        => icon%iwc(idx_start:idx_end,:)
    rttov_atm%lwc        => icon%lwc(idx_start:idx_end,:)
    rttov_atm%fl_rain    => icon%fl_lsrain(idx_start:idx_end,:)
    rttov_atm%fl_snow    => icon%fl_lssnow(idx_start:idx_end,:)

    
  END SUBROUTINE rttov_setup_atm




  

END MODULE MOD_RTTOV_SETUP
