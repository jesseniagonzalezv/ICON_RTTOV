# ICON_RTTOV


ML_RTTOV: to adapt the dataset and call rttov121

rttov121: simulator
 -Download rttov121 https://nwp-saf.eumetsat.int/site/login/?redirect_to=https%3A%2F%2Fnwp-saf.eumetsat.int%2Fsite%2Fuser%2F

----------
::MISTRAL
1. Create all the folder. It is organized in the following way:
        
        $cd /work/bb1036/b381362/
        ├── github
        │      ├── Retrievals
        │            ├── ML_RTTOV
        │                   └── src
        │                   └── Makefile
        ├── tools
        │            ├── rttov121 ****Main folder RTTOV simulator 
        │                                   (files generated) 
        │                                    └── obj
        │                                    └── mod
        │                                    └── lib
        │                                    └── include
        │                                    └── bin

1.1. Dowload coeff rttov:
```
  $ rttov121/rtcoef_rttov12/rttov_coef_download.sh
    - y
```    
1.2. emis_data
  - Dowloaded UW IR atlas data – this includes the angular correction data ([https://nwp-saf.eumetsat.int/site/software/rttov/download/](https://nwp-saf.eumetsat.int/site/software/rttov/download/))
  - The file should be unzipped in the folder of emis_data/ (rttov121/emis_data)

1.3. BRF_data
  - BRDF atlas files (unchanged since v11.3)/ BRDF atlas data ([https://nwp-saf.eumetsat.int/site/software/rttov/download/](https://nwp-saf.eumetsat.int/site/software/rttov/download/))
  - The  file should be unzipped in the folder of brdf_data/ (rttov121/brdf_data)
   
2. Compile RTTOV (rttov121)
  -  Change paths on the build/Makefile.local
      - HDF5_PREFIX  = /sw/rhel6-x64/hdf5/hdf5-1.8.14-intel14 
      - LDFLAGS_HDF5 = -L$(HDF5_PREFIX)/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz
      - NETCDF_PREFIX  = /sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-intel14
      - FFLAGS_NETCDF  = -D_RTTOV_NETCDF -I$(NETCDF_PREFIX)/include
      - LDFLAGS_NETCDF = -L$(NETCDF_PREFIX)/lib -lnetcdff
  ```
  $ source ~/.bashrc
  $ conda activate phd
  $ module load intel 
  $ cd /work/bb1036/b381362/tools/rttov121/src
  $ ../build/rttov_compile.sh
  ```
      - Specify required compiler flag file (leave blank for default: gfortran)
          > ifort
      -  Specify installation directory relative to top-level RTTOV directory (leave blank for default: ./)
          > 
      -  Have you updated the file build/Makefile.local with the location of your HDF5 installation? (y/n)
          > y
         Testing for f2py...
          ...f2py does not appear to be installed, Python interface and RTTOV GUI cannot be compiled

      -  Previous build detected in ./: perform a clean compilation? (y/n)
          Choose y if you have changed the compilation options since the previous build and re-compilation fails.
          > y

      -  Specify any additional flags to pass to make (e.g. -j); leave blank if unsure
          > 
  
        ├── Compiling with flags    : ifort
        ├── Compiling in directory  : ./
        ├──    RTTOV features available:
        ├──    HDF5 coefficient I/O    : y
        ├──    Emissivity/BRDF atlases : y
        ├──    C/C++ wrapper           : y
        ├──    Python wrapper          : n 
        ├──    RTTOV GUI               : n 
  - Verifying the RTTOV
  ```
  $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/rhel6-x64/hdf5/hdf5-1.8.14-intel14/lib/:/sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-intel14/lib/
  $RTTOV/rttov_test ./test_rttov12.sh ARCH=myarch [BIN=bindir]
   ├──  ./test_fwd.sh ARCH=ifort
  $RTTOV/rttov_test ./rttov_test.pl ARCH=myarch [BIN=bindir] TEST_LIST=hirs/001,avhrr/001 DIRECT=1
   ├── ./rttov_test.pl TEST_LIST=modis/201 ARCH=ifort DIRECT=1
  $RTTOV/rttov_test ./run_example_fwd.sh ARCH=myarch [BIN=bindir]

  ```
  The ARCH parameter should match the one used when you compiled RTTOV. The BIN parameter is optional
(indicated by the square brackets []). It is only required if the INSTALLDIR parameter was supplied when compiling
RTTOV i.e. if the location of bin/ is not in the top-level RTTOV directory. If specified BIN must give the location of
the directory containing binary executables relative to the top-level RTTOV distribution directory (e.g. if you specified
INSTALLDIR=install/gfortran when building RTTOV then you should use BIN=install/gfortran/bin).

4. Create the input data to RTTOV (subset_rttov_T12.nc)
  ```
   #source ~/.bashrc
   #conda activate phd
   #module load nco
 
  $ cd /work/bb1036/b381362/dataset
  $ bash create_dataset_rttov.py
  
  ```
5. Compile ML_RTTOV

  ```
  $ cd /work/bb1036/b381362/github/Retrievals/ML_RTTOV/
  $ module load intel
  $ make
  $ ./ml_rttov
  ```


5. Dataset availables in Mistral:
```
 $ cd /work/bb1036/b381362/dataset/test-2.nc
 $ cd cd /work/bb1036/b381362/dataset/data_rttov_T12.nc
 $ cd cd /work/bb1036/b381362/dataset/subset_rttov_T12.nc (My dataset)
 ```
 
 

 

