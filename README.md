# ICON_RTTOV


ML_RTTOV: to adapt the dataset and call rttov121

rttov121: simulator

----------
::MISTRAL
1. Create all the folder. It is organized in the following way:
 
        ├── github
        │      ├── Retrievals
        │            ├── ML_RTTOV
        │                   └── src
        │                   └── Makefile
        ├── storage
        │      ├── data
        │            ├── tools
        │                   ├── RTTOV
        │                         ├── rttov121 ****Main folder RTTOV simulator (check steps 2.)
        │      ├── scripts
        │            ├── tools
        │                   ├── RTTOV
        │                         ├── rttov121-new (files generated when run rttov121)
        │                                    └── obj
        │                                    └── mod
        │                                    └── lib
        │                                    └── include
        │                                    └── bin


2. Compile RTTOV (rttov121)
  -  Change paths on the build/Makefile.local
      - HDF5_PREFIX  = /sw/rhel6-x64/hdf5/hdf5-1.8.14-intel14 
      - LDFLAGS_HDF5 = -L$(HDF5_PREFIX)/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz
      - NETCDF_PREFIX  = /sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-intel14
      - FFLAGS_NETCDF  = -D_RTTOV_NETCDF -I$(NETCDF_PREFIX)/include
      - LDFLAGS_NETCDF = -L$(NETCDF_PREFIX)/lib -lnetcdff
  
  - ../build/rttov_compile.sh
  
   │ Compiling with flags    : ifort
   │ Compiling in directory  : ../../../../scripts/tools/RTTOV/rttov121-new
   RTTOV features available:
   HDF5 coefficient I/O    : y
   Emissivity/BRDF atlases : y
   C/C++ wrapper           : y
   Python wrapper          : n *change to y
   RTTOV GUI               : n *change to y 
     
2.1. Dowload coeff rttov:
  - rttov121/rtcoef_rttov12/rttov_coef_download.sh
    - y
    
2.2. emis_data
  - Dowloaded UW IR atlas data – this includes the angular correction data ([https://nwp-saf.eumetsat.int/site/software/rttov/download/](https://nwp-saf.eumetsat.int/site/software/rttov/download/))
  - The file should be unzipped in the folder of emis_data/ (rttov121/emis_data)

2.3. BRF_data
  - BRDF atlas files (unchanged since v11.3)/ BRDF atlas data ([https://nwp-saf.eumetsat.int/site/software/rttov/download/](https://nwp-saf.eumetsat.int/site/software/rttov/download/))
  - The  file should be unzipped in the folder of brdf_data/ (rttov121/brdf_data)

3. Compile ML_RTTOV
  - module load intel
  - $ML_RTTOV   make
  - $ML_RTTOV ./ml_rttov

