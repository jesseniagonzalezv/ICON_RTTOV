MODULE MOD_WRITE_OUTPUT

  USE ml_types, only: type_ml, type_icon, wp
  USE ml_config, only: rttov_nchannels
  USE netcdf
  
  IMPLICIT NONE

CONTAINS

  SUBROUTINE write_output(fname_out,icon,oe)

    USE mod_read_icon, ONLY: map_point_to_ll
    
    ! Input variables
    CHARACTER(len=256), INTENT(IN) :: fname_out
    TYPE(type_icon), INTENT(IN) :: icon
    TYPE(type_ml), INTENT(IN) :: oe
    
    ! Local variables
    !REAL(KIND=wp), DIMENSION(icon%Nlon,icon%Nlat,rttov_nchannels) :: gridded_y, gridded_y_clear, gridded_brdf
        !!!!!!!!!!!!!! JGV 
    REAL(KIND=wp), DIMENSION(icon%Nlon,icon%Nlat,rttov_nchannels) :: gridded_y, gridded_y_clear, gridded_brdf,gridded_f, gridded_f_clear
     !!!!!!!!!!!!!! end JGV 
    REAL(KIND=wp), DIMENSION(icon%Nlon,icon%Nlat) :: gridded_iwp_model, gridded_iwp_ret, gridded_g
    
    INTEGER(KIND=4) :: ichannel
    !INTEGER(KIND=4) :: ncid, varid1, varid2, varid3, varid4, varid5, varid6, dimid_latlon(2), &
     !    dimid_latlonchan(3), errst, dimid_lat, dimid_lon, dimid_chan
    !!!!!!!!!!!!!! JGV 
    INTEGER(KIND=4) :: ncid, varid1, varid2, varid3, varid4, varid5, varid6,varid7,varid8, dimid_latlon(2), &
         dimid_latlonchan(3), errst, dimid_lat, dimid_lon, dimid_chan     
    !!!!!!!!!!!!!! end JGV      
    
    INTEGER(KIND=4) :: varid_lat, varid_lon

    

     
     
    write(*,*) icon%Nlon, icon%Nlat
    
    ! DO ichannel=1,rttov_nchannels
    !    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x1=oe%y(:,ichannel),y2=gridded_y(:,:,ichannel))
    ! END DO
    
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x2=oe%y,y3=gridded_y)
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x2=oe%y_clear,y3=gridded_y_clear)
    
    !!!!!!!!!!!!!! JGV 
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x2=oe%f,y3=gridded_f)
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x2=oe%f_clear,y3=gridded_f_clear)
     !!!!!!!!!!!!!! end JGV 
     
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x2=oe%brdf,y3=gridded_brdf)
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x1=oe%Xip1(:,1),y2=gridded_iwp_ret)
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x1=oe%iwp_model(:),y2=gridded_iwp_model)
    call map_point_to_ll(icon%Nlon,icon%Nlat,icon%mode,x1=oe%gip1(:),y2=gridded_g)                
       
    errst = nf90_create(fname_out,NF90_CLOBBER,ncid)
    
    errst = nf90_def_dim(ncid, "lon", icon%Nlon, dimid_lon)    
    errst = nf90_def_dim(ncid, "lat", icon%Nlat, dimid_lat)
    errst = nf90_def_dim(ncid, "chan", rttov_nchannels, dimid_chan)    

    dimid_latlon =  (/ dimid_lon, dimid_lat /)
    dimid_latlonchan =  (/ dimid_lon, dimid_lat,dimid_chan/)    
    
    errst = nf90_def_var(ncid,"lat", NF90_REAL,dimid_lat,varid_lat)    
    errst = nf90_def_var(ncid,"lon", NF90_REAL,dimid_lon,varid_lon)
    errst = nf90_def_var(ncid,"Y", NF90_REAL,dimid_latlonchan,varid1)
    errst = nf90_def_var(ncid,"Y_clear", NF90_REAL,dimid_latlonchan,varid2)

    errst = nf90_def_var(ncid,"brdf", NF90_REAL,dimid_latlonchan,varid3)
    errst = nf90_def_var(ncid,"iwp_ret", NF90_REAL,dimid_latlon,varid4)
    errst = nf90_def_var(ncid,"iwp_model", NF90_REAL,dimid_latlon,varid5)
    errst = nf90_def_var(ncid,"g", NF90_REAL,dimid_latlon,varid6)         
    !!!!!!!!!!!!!!  JGV 
    errst = nf90_def_var(ncid,"f", NF90_REAL,dimid_latlonchan,varid7)
    errst = nf90_def_var(ncid,"f_clear", NF90_REAL,dimid_latlonchan,varid8)
    !!!!!!!!!!!!!! end JGV            
    errst = nf90_enddef(ncid)
    
    errst = nf90_put_var(ncid, varid_lon, icon%lon_orig)
    errst = nf90_put_var(ncid, varid_lat, icon%lat_orig)
    errst = nf90_put_var(ncid, varid1, gridded_y(:,:,:))
    errst = nf90_put_var(ncid, varid2, gridded_y_clear(:,:,:))
    errst = nf90_put_var(ncid, varid3, gridded_brdf(:,:,:))        
    errst = nf90_put_var(ncid, varid4, gridded_iwp_ret(:,:))
    errst = nf90_put_var(ncid, varid5, gridded_iwp_model(:,:))
    errst = nf90_put_var(ncid, varid6, gridded_g(:,:))            
    
    !!!!!!!!!!!!!!  JGV 
    errst = nf90_put_var(ncid, varid7, gridded_f(:,:,:))
    errst = nf90_put_var(ncid, varid8, gridded_f_clear(:,:,:))  
    !!!!!!!!!!!!!! end JGV     
    errst = nf90_close(ncid)

    write(*,*) "Done writting"
    
  END SUBROUTINE write_output

  
END MODULE MOD_WRITE_OUTPUT
