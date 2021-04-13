MODULE MOD_READ_ICON

  use netcdf
  use ml_types, only: wp, type_icon
  use ml_config, only: Nhydro, amd, rd, &
       mr_co2,amCO2,mr_ch4,amCH4,mr_n2o,amN2O,mr_co,amCO,amO3

  use mod_utils_fort, only: ml_error
  use mod_regrid
  
  IMPLICIT NONE

  ! Types to be used as arrays of pointers
  TYPE var1d
     character(len=16) :: name
     character(len=16) :: units
     integer :: dimsid(3)
     integer :: dimssz(2)
     integer :: vid
     logical :: lout
     real(wp),pointer,dimension(:) :: pntr
  END TYPE var1d
  TYPE var2d
     character(len=16) :: name
     character(len=16) :: units
     integer :: dimsid(4)
     integer :: dimssz(3)
     integer :: vid
     logical :: lout
     real(wp),pointer,dimension(:,:) :: pntr
  END TYPE var2d
  TYPE var3d
     character(len=16) :: name
     character(len=16) :: units
     integer :: dimsid(5)
     integer :: dimssz(4)
     integer :: vid
     logical :: lout
     real(wp),pointer,dimension(:,:,:) :: pntr
  END TYPE var3d

  
CONTAINS

  SUBROUTINE READ_ICON(fname,icon)

    !! Parameters
    !! ----------------------------------------------------------------------------------------------------
    character(len=64), parameter :: routine_name='READ_ICON'
    integer,parameter :: NMAX_DIM=5
    !! ----------------------------------------------------------------------------------------------------
    
    !! Inputs
    !! ----------------------------------------------------------------------------------------------------    
    character(len=256), intent(in) :: fname
    !! ----------------------------------------------------------------------------------------------------
    
    !! Inputs/Outputs
    !! ----------------------------------------------------------------------------------------------------    
    type(type_icon), intent(inout) :: icon
    !! ----------------------------------------------------------------------------------------------------
    
    !! Local variables
    !! ----------------------------------------------------------------------------------------------------    
    character(len=256) :: errmsg, straux
    character(len=256) :: dimname(NMAX_DIM), vname
    
    integer(kind=4) :: idim, dimsize(NMAX_DIM), vdimid(NMAX_DIM)
    integer(kind=4) :: ncid, ndims, nvars, ngatts, recdim, errst, vid, vrank
    integer(kind=4) :: Na,Nb,Nc,Nd,Ne,i,j,k
    integer(kind=4) :: npoints, nlevels, nhydro
    integer,dimension(:),allocatable :: plon,plat

    real(wp), dimension(:), allocatable :: lat, lon
    real(wp), dimension(icon%npoints,icon%nlevels) :: ph, zh, rho_atm
    real(wp),dimension(icon%npoints) :: ll
    real(wp),allocatable :: x1(:),x2(:,:),x3(:,:,:),x4(:,:,:,:),x5(:,:,:,:,:) ! Temporary arrays
    
    logical :: Llat, Llon, Lpoint
    !! ----------------------------------------------------------------------------------------------------

    npoints = icon%npoints; nlevels = icon%nlevels; nhydro = icon%nhydro

    errst = nf90_open(fname, nf90_nowrite, ncid)
    if(errst/=0)  then
       errmsg = "Couldn't open "//trim(fname)
       call ml_error(routine_name,errmsg)
    end if

    !! Check the dimentions (track or lat-lon)
    !! -------------------------------------------------------------------------------------------------
    errst = nf90_inquire(ncid, ndims, nvars, ngatts, recdim)
    if (errst /= 0) then
       errmsg="Error in  nf90_inquire"
       call ml_error(routine_name,errmsg,errcode=errst)
    endif

    Llat  =.false.; Llon  =.false.; Lpoint=.false.    
    do idim = 1,ndims
       errst = nf90_Inquire_Dimension(ncid,idim,name=dimname(idim),len=dimsize(idim))
       if (errst /= 0) then
          write(straux, *)  idim
          errmsg="Error in nf90_Inquire_Dimension, idim: "//trim(straux)
          call ml_error(routine_name,errmsg)
       endif

       if ((trim(dimname(idim)).eq.'level').and.(nlevels > dimsize(idim))) then
          errmsg='Number of levels selected is greater than in input file '//trim(fname)
          call ml_error(routine_name,errmsg)
       endif

       if (trim(dimname(idim)).eq.'point') then
          Lpoint = .true.
          if (npoints /= dimsize(idim)) then
             errmsg='Number of points selected is greater than in input file '//trim(fname)
             call ml_error(routine_name,errmsg)
          endif
       endif
       
       if (trim(dimname(idim)).eq.'lon') then
          Llon = .true.; icon%nlon = dimsize(idim)
       endif
       if (trim(dimname(idim)).eq.'lat') then
          Llat = .true.; icon%nlat = dimsize(idim)
       endif
    enddo

    allocate(lon(icon%nlon),lat(icon%nlat),icon%lon_orig(icon%nlon),icon%lat_orig(icon%nlat))
    !! -------------------------------------------------------------------------------------------------

    !! Extract coordinates
    !! -------------------------------------------------------------------------------------------------
    if (Llon.and.Llat) then ! 2D mode
       if (npoints /= icon%nlon*icon%nlat) then
          errmsg='Number of points selected is different from indicated in input file '//trim(fname)
          call ml_error(routine_name,errmsg)
       end if
       lon = -1.0E30; lat = -1.0E30
       icon%mode = 2 ! Don't know yet if (lon,lat) or (lat,lon) at this point
    else if (Lpoint) then ! 1D mode
       icon%nlon = npoints
       icon%nlat = npoints
       icon%mode = 1
    else
       errmsg= trim(fname)//' file contains wrong dimensions'
       call ml_error(routine_name,errmsg)
    endif
    errst = nf90_inq_varid(ncid, 'lon', vid)
    if (errst /= 0) then
       errmsg="Error in nf90_inq_varid, var: lon"
       call ml_error(routine_name,errmsg,errcode=errst)
    endif
    errst = nf90_get_var(ncid, vid, lon, start = (/1/), count = (/icon%nlon/))
    
    if (errst /= 0) then
       errmsg="Error in nf90_get_var, var: lon"
       call ml_error(routine_name,errmsg,errcode=errst)
    endif
    errst = nf90_inq_varid(ncid, 'lat', vid)
    if (errst /= 0) then
       errmsg="Error in nf90_inq_varid, var: lat"
       call ml_error(routine_name,errmsg,errcode=errst)
    endif
    errst = nf90_get_var(ncid, vid, lat, start = (/1/), count = (/icon%nlat/))
    if (errst /= 0) then
       errmsg="Error in nf90_get_var, var: lat"
       call ml_error(routine_name,errmsg,errcode=errst)
    endif

    icon%lon_orig = lon; icon%lat_orig = lat
    !! -------------------------------------------------------------------------------------------------

    !! Extract all variables
    !! -------------------------------------------------------------------------------------------------
    do vid = 1,nvars
       vdimid=0
       errst = nf90_Inquire_Variable(ncid, vid, name=vname, ndims=vrank, dimids=vdimid)
       if (errst /= 0) then
          write(straux, *)  vid
          errmsg='Error in nf90_Inquire_Variable, vid '//trim(straux)
          call ml_error(routine_name,errmsg,errcode=errst)
       endif
       ! Read in into temporary array of correct shape
       if (vrank == 1) then
          Na = dimsize(vdimid(1))
          allocate(x1(Na))
          errst = nf90_get_var(ncid, vid, x1, start=(/1/), count=(/Na/))
       endif
       if (vrank == 2) then
          Na = dimsize(vdimid(1))
          Nb = dimsize(vdimid(2))
          allocate(x2(Na,Nb))
          errst = nf90_get_var(ncid, vid, x2, start=(/1,1/), count=(/Na,Nb/))
       endif
       if (vrank == 3) then
          Na = dimsize(vdimid(1))
          Nb = dimsize(vdimid(2))
          Nc = dimsize(vdimid(3))
          allocate(x3(Na,Nb,Nc))
          errst = nf90_get_var(ncid, vid, x3, start=(/1,1,1/), count=(/Na,Nb,Nc/))
          if ((icon%mode == 2).or.(icon%mode == 3)) then
             if ((Na == icon%nlon).and.(Nb == icon%nlat)) then
                icon%mode = 2
             else if ((Na == icon%nlat).and.(Nb == icon%nlon)) then
                icon%mode = 3
             else
                errmsg='Wrong mode for variable '//trim(vname)
                call ml_error(routine_name,errmsg)
             endif
          endif
       endif
       if (vrank == 4) then
          Na = dimsize(vdimid(1))
          Nb = dimsize(vdimid(2))
          Nc = dimsize(vdimid(3))
          Nd = dimsize(vdimid(4))
          allocate(x4(Na,Nb,Nc,Nd))
          errst = nf90_get_var(ncid, vid, x4, start=(/1,1,1,1/), count=(/Na,Nb,Nc,Nd/))
       endif
       if (vrank == 5) then
          Na = dimsize(vdimid(1))
          Nb = dimsize(vdimid(2))
          Nc = dimsize(vdimid(3))
          Nd = dimsize(vdimid(4))
          Ne = dimsize(vdimid(5))
          allocate(x5(Na,Nb,Nc,Nd,Ne))
          errst = nf90_get_var(ncid, vid, x5, start=(/1,1,1,1,1/), count=(/Na,Nb,Nc,Nd,Ne/))
       endif
       if (errst /= 0) then
          write(straux, *)  vid
          errmsg='Error in nf90_get_var, vid '//trim(straux)
          call ml_error(routine_name,errmsg,errcode=errst)
       endif
       
       ! Map to the right input argument
       select case (trim(vname))
       case ('pfull')
          if (Lpoint) then
             icon%p(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%p)
          endif
       case ('phalf')
          if (Lpoint) then
             ph(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=ph)
          endif
       case ('height')
          if (Lpoint) then
             icon%zlev(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%zlev)
          endif
       case ('height_half')
          if (Lpoint) then
             zh(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=zh)
          endif
       case ('T_abs')
          if (Lpoint) then
             icon%T(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%T)
          endif
       case ('qv')
          if (Lpoint) then
             icon%sh(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%sh)
          endif
       case ('rh')
          if (Lpoint) then
             icon%rh(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%rh)
          endif
       case ('tca')
          if (Lpoint) then
             icon%tca(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%tca)
          endif
       case ('cca')
          if (Lpoint) then
             icon%cca(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%cca)
          endif
       case ('mr_lsliq')
          if (Lpoint) then
             icon%mr_lsliq(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%mr_lsliq)
          endif
       case ('mr_lsice')
          if (Lpoint) then
             icon%mr_lsice(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%mr_lsice)
          endif
       case ('mr_ccliq')
          if (Lpoint) then
             icon%mr_ccliq(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%mr_ccliq)
          endif
       case ('mr_ccice')
          if (Lpoint) then
             icon%mr_ccice(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%mr_ccice)
          endif
       case ('fl_lsrain')
          if (Lpoint) then
             icon%fl_lsrain(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%fl_lsrain)
          endif
       case ('fl_lssnow')
          if (Lpoint) then
             icon%fl_lssnow(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%fl_lssnow)
          endif
       case ('fl_lsgrpl')
          if (Lpoint) then
             icon%fl_lsgrpl(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%fl_lsgrpl)
          endif
       case ('fl_ccrain')
          if (Lpoint) then
             icon%fl_ccrain(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%fl_ccrain)
          endif
       case ('fl_ccsnow')
          if (Lpoint) then
             icon%fl_ccsnow(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%fl_ccsnow)
          endif
       case ('dtau_s')
          if (Lpoint) then
             icon%dtau_s(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%dtau_s)
          endif
       case ('dtau_c')
          if (Lpoint) then
             icon%dtau_c(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%dtau_c)
          endif
       case ('dem_s')
          if (Lpoint) then
             icon%dem_s(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%dem_s)
          endif
       case ('dem_c')
          if (Lpoint) then
             icon%dem_c(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%dem_c)
          endif
       case ('Reff')
          if (Lpoint) then
             icon%Reff(1:npoints,:,:) = x3(1:npoints,1:nlevels,:)
          else
             call map_ll_to_point(Na,Nb,npoints,x4=x4,y3=icon%Reff)
          endif
       case ('skt')
          if (Lpoint) then
             icon%skt(1:npoints) = x1(1:npoints)
          else
             call map_ll_to_point(Na,Nb,npoints,x2=x2,y1=icon%skt)
          endif
       case ('psfc')
          if (Lpoint) then
             icon%skt(1:npoints) = x1(1:npoints)
          else
             call map_ll_to_point(Na,Nb,npoints,x2=x2,y1=icon%psfc)
          endif
       case ('orography')
          if (Lpoint) then
             icon%orography(1:npoints) = x1(1:npoints)
          else
             call map_ll_to_point(Na,Nb,npoints,x2=x2,y1=icon%orography)
          endif
       case ('landmask')
          if (Lpoint) then
             icon%landmask(1:npoints) = x1(1:npoints)
          else
             call map_ll_to_point(Na,Nb,npoints,x2=x2,y1=icon%landmask)
          endif
       case ('mr_ozone')
          if (Lpoint) then
             icon%mr_ozone(1:npoints,:) = x2(1:npoints,1:nlevels)
          else
             call map_ll_to_point(Na,Nb,npoints,x3=x3,y2=icon%mr_ozone)
          endif
       case ('u_wind')
          if (Lpoint) then
             icon%u_wind(1:npoints) = x1(1:npoints)
          else
             call map_ll_to_point(Na,Nb,npoints,x2=x2,y1=icon%u_wind)
          endif
       case ('v_wind')
          if (Lpoint) then
             icon%v_wind(1:npoints) = x1(1:npoints)
          else
             call map_ll_to_point(Na,Nb,npoints,x2=x2,y1=icon%v_wind)
          endif
       case ('sunlit')
          if (Lpoint) then
             icon%sunlit(1:npoints) = x1(1:npoints)
          else
             call map_ll_to_point(Na,Nb,npoints,x2=x2,y1=icon%sunlit)
          endif
       end select
       !! Free memory
       if (vrank == 1) deallocate(x1)
       if (vrank == 2) deallocate(x2)
       if (vrank == 3) deallocate(x3)
       if (vrank == 4) deallocate(x4)
       if (vrank == 5) deallocate(x5)
    enddo
    !! -------------------------------------------------------------------------------------------------


    ! SFC emissivity
    errst = nf90_inq_varid(ncid, 'emsfc_lw', vid)
    if (errst /= 0) then
       if (errst == nf90_enotvar) then ! Does not exist, use 1.0
          icon%emsfc_lw = 1.0
          print *, ' ********* COSP Warning:  emsfc_lw does not exist in input file. Set to 1.0.'
       else  ! Other error, stop
          errmsg='Error in nf90_inq_varid, var: emsfc_lw'
          call ml_error(routine_name,errmsg,errcode=errst)
       endif
    else
       errst = nf90_get_var(ncid, vid, icon%emsfc_lw)
       if (errst /= 0) then
          errmsg='Error in nf90_get_var, var: emsfc_lw'
          call ml_error(routine_name,errmsg,errcode=errst)
       endif
    endif

    ! Fill in the lat/lon vectors with the right values for 2D modes
    ! This might be helpful if the inputs are 2D (gridded) and 
    ! you want outputs in 1D mode
    allocate(plon(npoints),plat(npoints))
    if (icon%mode == 2) then !(lon,lat)
       ll = lat
       do j=1,Nb
          do i=1,Na
             k = (j-1)*Na + i
             plon(k) = i  
             plat(k) = j
          enddo
       enddo
       icon%lon(1:npoints) = lon(plon(1:npoints))
       icon%lat(1:npoints) = ll(plat(1:npoints))
    else if (icon%mode == 3) then !(lat,lon)
       ll = lon
       do j=1,Nb
          do i=1,Na
             k = (j-1)*Na + i
             lon(k) = ll(j)
             lat(k) = lat(i)
          enddo
       enddo
       icon%lon(1:npoints) = ll(plon(1:npoints))
       icon%lat(1:npoints) = lat(plat(1:npoints))
    endif
    deallocate(plon,plat)

    
    errst = nf90_close(ncid)
    if (errst /= 0) then
       errmsg='Error in nf90_close'
       call ml_error(routine_name,errmsg,errcode=errst)
    endif 

    icon%zlev_half(1:Npoints,1:nlevels)=zh(1:npoints,1:nlevels)!; icon%zlev_half(1:npoints,1)=0._wp
    icon%ph(1:Npoints,1:nlevels)= ph(1:npoints,1:nlevels)!; icon%ph(1:npoints,1)=0._wp

    icon%T_lev(1:npoints,1:nlevels) = icon%T(1:npoints,1:nlevels); icon%T_lev(1:npoints,nlevels+1) = icon%skt(1:npoints)
    icon%p_lev(1:npoints,1:nlevels) = icon%p(1:npoints,1:nlevels); icon%p_lev(1:npoints,nlevels+1) = icon%psfc(1:npoints)
    icon%sh_lev(1:npoints,2:nlevels+1) = icon%sh(1:npoints,1:nlevels); icon%sh_lev(1:npoints,1) = 0._wp
    icon%mr_ozone_lev(1:npoints,2:nlevels+1) = icon%mr_ozone(1:npoints,1:nlevels); icon%mr_ozone_lev(1:npoints,1) = 0._wp    
    
    !! Not sure about units here, taken from COSP, check at some point
    icon%co2 = mr_co2*(amd/amCO2)*1e6
    icon%ch4 = mr_ch4*(amd/amCH4)*1e6  
    icon%n2o = mr_n2o*(amd/amN2O)*1e6
    icon%co  = mr_co*(amd/amCO)*1e6
    icon%mr_ozone = icon%mr_ozone*(amd/amO3)*1e6 ! microns

    rho_atm = icon%p / (rd * icon%t)
    icon%iwc = icon%mr_lsice*rho_atm
    icon%lwc = icon%mr_lsliq*rho_atm    

    icon%dz(:,1:150) = abs(icon%zlev_half(:,1:150) - icon%zlev(:,1:150)) * 2._wp
    
  END SUBROUTINE READ_ICON


  subroutine construct_icon(npoints,nlevels,nhydro,y)

    type(type_icon) :: y
    integer(kind=4),intent(in) :: &
         npoints,  & ! Number of horizontal gridpoints
         nlevels,  & ! Number of vertical levels
         nhydro
         
    allocate(y%lon(Npoints),y%lat(Npoints),y%p(Npoints,Nlevels),y%ph(Npoints,Nlevels+1),y%p_lev(Npoints,Nlevels+1),         &
         y%zlev(Npoints,Nlevels),y%zlev_half(Npoints,Nlevels+1),y%T(Npoints,Nlevels), y%T_lev(Npoints,Nlevels+1),          &
         y%sh(Npoints,Nlevels),y%sh_lev(Npoints,Nlevels+1),y%rh(Npoints,Nlevels), y%tca(Npoints,Nlevels),                 &
         y%cca(Npoints,Nlevels),y%mr_lsliq(Npoints,Nlevels),y%mr_lsice(Npoints,Nlevels),     &
         y%mr_ccliq(Npoints,Nlevels),y%mr_ccice(Npoints,Nlevels),                            &
         y%fl_lsrain(Npoints,Nlevels),y%fl_lssnow(Npoints,Nlevels),                          &
         y%fl_lsgrpl(Npoints,Nlevels),y%fl_ccrain(Npoints,Nlevels),                          &
         y%fl_ccsnow(Npoints,Nlevels),y%Reff(Npoints,Nlevels,Nhydro),                        &
         y%dtau_s(Npoints,Nlevels),y%dtau_c(Npoints,Nlevels),y%dem_s(Npoints,Nlevels),       &
         y%dem_c(Npoints,Nlevels),y%skt(Npoints),y%landmask(Npoints),                        &
         y%mr_ozone(Npoints,Nlevels),y%u_wind(Npoints),y%v_wind(Npoints),y%sunlit(Npoints),  &
         y%seaice(Npoints),y%psfc(Npoints),y%orography(Npoints),y%mr_ozone_lev(Npoints,Nlevels+1),&
         y%lwc(Npoints,Nlevels),y%iwc(Npoints,Nlevels),y%dz(Npoints,Nlevels))

    y%nlevels = Nlevels; y%npoints = Npoints; y%nhydro=Nhydro
    
  end subroutine construct_icon

  

END MODULE MOD_READ_ICON
