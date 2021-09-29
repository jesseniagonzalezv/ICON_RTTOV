MODULE MOD_REGRID

  use mod_utils_fort, only: ml_error
  use ml_types, only: wp  
  
  IMPLICIT NONE

CONTAINS

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ! SUBROUTINE map_point_to_ll
  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  SUBROUTINE MAP_POINT_TO_LL(Nx,Ny,geomode,x1,x2,x3,x4,y2,y3,y4,y5)
    ! Input arguments
    integer,intent(in) :: Nx,Ny,geomode
    real(wp),intent(in),optional :: x1(:),x2(:,:),x3(:,:,:), &
         x4(:,:,:,:)
    real(wp),intent(out),optional :: y2(:,:),y3(:,:,:), & 
         y4(:,:,:,:),y5(:,:,:,:,:)
    ! Local variables
    integer :: Npoints
    integer :: px(Nx*Ny),py(Nx*Ny)
    integer :: i,j,k,l,m
    integer :: Ni,Nj,Nk,Nl
    integer :: Mi,Mj,Mk,Ml,Mm
     character(len=128) :: proname='MAP_POINT_TO_LL'
     
     Npoints = Nx*Ny
     
     px=0
     py=0
     ! Obtain pointers to do the mapping
     if (geomode == 2) then ! (lon,lat) mode
        do j=1,Ny
           do i=1,Nx
              k = (j-1)*Nx+i
              px(k) = i  
              py(k) = j  
           enddo
        enddo
     else if (geomode == 3) then ! (lon,lat) mode
        do j=1,Nx
           do i=1,Ny
              k = (j-1)*Ny+i
              px(k) = j
              py(k) = i  
           enddo
        enddo
     else
        print *, ' -- '//trim(proname)//': geomode not supported, ',geomode
        stop
     endif
     
     if (present(x1).and.present(y2)) then
        Ni = size(x1,1)
        Mi = size(y2,1)
        Mj = size(y2,2)
        if (Mi*Mj /= Ni) then
           print *, ' -- '//trim(proname)//': Nlon*Nlat /= Npoints (opt 1)'
           stop
        endif
        do i=1,Npoints
           y2(px(i),py(i)) = x1(i)
        enddo
     else if (present(x2).and.present(y3)) then
        Ni = size(x2,1)
        Nj = size(x2,2)
        Mi = size(y3,1)
        Mj = size(y3,2)
        Mk = size(y3,3)
        if (Mi*Mj /= Ni) then
           print *, ' -- '//trim(proname)//': Nlon*Nlat /= Npoints (opt 2)'
           stop
        endif
        if (Nj /= Mk) then
           print *, ' -- '//trim(proname)//': Nj /= Mk (opt 2)'
           print *, Ni,Nj,Mi,Mj,Mk
           stop
        endif
        do k=1,Mk
           do i=1,Npoints
              y3(px(i),py(i),k) = x2(i,k)
           enddo
        enddo
     else if (present(x3).and.present(y4)) then
        Ni = size(x3,1)
        Nj = size(x3,2)
        Nk = size(x3,3)
        Mi = size(y4,1)
        Mj = size(y4,2)
        Mk = size(y4,3)
        Ml = size(y4,4)
        if (Mi*Mj /= Ni) then
           print *, ' -- '//trim(proname)//': Nlon*Nlat /= Npoints (opt 3)'
           stop
        endif
        if (Nj /= Mk) then
           print *, ' -- '//trim(proname)//': Nj /= Mk (opt 3)'
           stop
        endif
        if (Nk /= Ml) then
           print *, ' -- '//trim(proname)//': Nk /= Ml (opt 3)'
           stop
        endif
        do l=1,Ml
           do k=1,Mk
              do i=1,Npoints
                 y4(px(i),py(i),k,l) = x3(i,k,l)
              enddo
           enddo
        enddo
     else if (present(x4).and.present(y5)) then
        Ni = size(x4,1)
        Nj = size(x4,2)
        Nk = size(x4,3)
        Nl = size(x4,4)
        Mi = size(y5,1)
        Mj = size(y5,2)
        Mk = size(y5,3)
        Ml = size(y5,4)
        Mm = size(y5,5)
        if (Mi*Mj /= Ni) then
           print *, ' -- '//trim(proname)//': Nlon*Nlat /= Npoints (opt 4)'
           stop
        endif
        if (Nj /= Mk) then
           print *, ' -- '//trim(proname)//': Nj /= Mk (opt 4)'
           stop
        endif
        if (Nk /= Ml) then
           print *, ' -- '//trim(proname)//': Nk /= Ml (opt 4)'
           stop
        endif
        if (Nl /= Mm) then
           print *, ' -- '//trim(proname)//': Nl /= Mm (opt 4)'
           stop
        endif
        do m=1,Mm
           do l=1,Ml
              do k=1,Mk
                 do i=1,Npoints
                    y5(px(i),py(i),k,l,m) = x4(i,k,l,m)
                 enddo
              enddo
           enddo
        enddo
     else
        print *, ' -- '//trim(proname)//': wrong option'
        stop
     endif
     
     
   END SUBROUTINE MAP_POINT_TO_LL

   !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ! SUBROUTINE map_ll_to_point
  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  SUBROUTINE MAP_LL_TO_POINT(Nx,Ny,Np,x2,x3,x4,x5,y1,y2,y3,y4)
    ! Input arguments
    integer,intent(in) :: Nx,Ny,Np
    real(wp),intent(in),optional :: x2(:,:),x3(:,:,:), &
         x4(:,:,:,:),x5(:,:,:,:,:)
    real(wp),intent(out),optional :: y1(:),y2(:,:),y3(:,:,:), &
         y4(:,:,:,:)
    ! Local variables
    integer :: px(Nx*Ny),py(Nx*Ny)
    integer :: i,j,k,l,m
    integer :: Ni,Nj,Nk,Nl,Nm
    integer :: Mi,Mj,Mk,Ml
    character(len=128) :: proname='MAP_LL_TO_POINT'
    
    px=0
    py=0
    if (Nx*Ny < Np) then
       print *, ' -- '//trim(proname)//': Nx*Ny < Np'
       stop
    endif
    do j=1,Ny
       do i=1,Nx
          k = (j-1)*Nx+i
          px(k) = i  
          py(k) = j  
       enddo
    enddo
    
    if (present(x2).and.present(y1)) then
       Ni = size(x2,1)
       Nj = size(x2,2)
       Mi = size(y1,1)
       if (Ni*Nj < Mi) then
          print *, ' -- '//trim(proname)//': Nlon*Nlat < Npoints (opt 1)'
          stop
       endif
       do j=1,Np
          y1(j) = x2(px(j),py(j))
       enddo
    else if (present(x3).and.present(y2)) then
       Ni = size(x3,1)
       Nj = size(x3,2)
       Nk = size(x3,3)
       Mi = size(y2,1)
       Mj = size(y2,2)
       if (Ni*Nj < Mi) then
          print *, ' -- '//trim(proname)//': Nlon*Nlat < Npoints (opt 2)'
          stop
       endif
       if (Nk /= Mj) then
          print *, ' -- '//trim(proname)//': Nk /= Mj (opt 2)'
          stop
       endif
       do k=1,Nk
          do j=1,Np
             y2(j,k) = x3(px(j),py(j),k)
          enddo
       enddo
    else if (present(x4).and.present(y3)) then
       Ni = size(x4,1)
       Nj = size(x4,2)
       Nk = size(x4,3)
       Nl = size(x4,4)
       Mi = size(y3,1)
       Mj = size(y3,2)
       Mk = size(y3,3)
       if (Ni*Nj < Mi) then
          print *, ' -- '//trim(proname)//': Nlon*Nlat < Npoints (opt 3)'
          stop
       endif
       if (Nk /= Mj) then
          print *, ' -- '//trim(proname)//': Nk /= Mj (opt 3)'
          stop
       endif
       if (Nl /= Mk) then
          print *, ' -- '//trim(proname)//': Nl /= Mk (opt 3)'
          stop
       endif
       do l=1,Nl
          do k=1,Nk
             do j=1,Np
                y3(j,k,l) = x4(px(j),py(j),k,l)
             enddo
          enddo
       enddo
    else if (present(x5).and.present(y4)) then
       Ni = size(x5,1)
       Nj = size(x5,2)
       Nk = size(x5,3)
       Nl = size(x5,4)
       Nm = size(x5,5)
       Mi = size(y4,1)
       Mj = size(y4,2)
       Mk = size(y4,3)
       Ml = size(y4,4)
       if (Ni*Nj < Mi) then
          print *, ' -- '//trim(proname)//': Nlon*Nlat < Npoints (opt 4)'
          stop
       endif
       if (Nk /= Mj) then
          print *, ' -- '//trim(proname)//': Nk /= Mj (opt 4)'
          stop
       endif
       if (Nl /= Mk) then
          print *, ' -- '//trim(proname)//': Nl /= Mk (opt 4)'
          stop
       endif
       if (Nm /= Ml) then
          print *, ' -- '//trim(proname)//': Nm /= Ml (opt 4)'
          stop
       endif
       do m=1,Nm
          do l=1,Nl
             do k=1,Nk
                do j=1,Np
                   y4(j,k,l,m) = x5(px(j),py(j),k,l,m)
                enddo
             enddo
          enddo
       enddo
    else
       print *, ' -- '//trim(proname)//': wrong option'
       stop
    endif 
  END SUBROUTINE MAP_LL_TO_POINT

END MODULE MOD_REGRID
