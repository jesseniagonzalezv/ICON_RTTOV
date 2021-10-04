

prog = ml_rttov

F90      = ifort
# F90FLAGS = -module $(mod) -fpp -qopenmp -O3 #-debug -g -traceback -check bounds
F90FLAGS = -module $(mod) -fpp -qopenmp -O0 -debug -g -traceback -check bounds 

## Let's start with some paths
# ---------------------------------------------------------------------------------------------------------------------------------------
main_home=/work/bb1036/b381362
#basedir = $(HOME)/github/Retrievals/ML_RTTOV
basedir = $(main_home)/github/ICON_RTTOV/interface_input_RTTOV/ML_RTTOV_130

src = $(basedir)/src
obj = $(basedir)/obj
lib = $(basedir)/lib
mod = $(basedir)/mod

path_main = $(src)/main
path_io = $(src)/ml_io
path_utils = $(src)/utils
path_rttov_ml = $(src)/rttov

path_NCDF_C_LIB = /sw/rhel6-x64/netcdf/netcdf_c-4.3.2-intel14/lib
path_NCDF_INC = /sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-intel14/include
path_NCDF_LIB = /sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-intel14/lib

HDF5_LIB    = /sw/rhel6-x64/hdf5/hdf5-1.8.14-intel14/lib

#RTTOV_PATH       = $(HOME)/storage/RTTOV/rttov121-new
RTTOV_PATH       = $(main_home)/tools/rttov130
RTTOV_LIB_PATH   = $(RTTOV_PATH)/lib 
RTTOV_INC_PATH   = $(RTTOV_PATH)/include 
RTTOV_MOD_PATH   = $(RTTOV_PATH)/mod
RTTOV_LIBS       =  -lrttov13_wrapper -lrttov13_mw_scatt -lrttov13_brdf_atlas -lrttov13_emis_atlas -lrttov13_other -lrttov13_parallel -lrttov13_coef_io -lrttov13_hdf -lrttov13_main -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5
# ---------------------------------------------------------------------------------------------------------------------------------------

# List of library files that will be created
# ---------------------------------------------------------------------------------------------------------------------------------------
LIB_MAIN = $(lib)/libmain.a
LIB_IO = $(lib)/lib_io.a
LIB_UTILS = $(lib)/libutils.a
LIB_RTTOVML = $(lib)/librttovml.a
# ---------------------------------------------------------------------------------------------------------------------------------------

# List of object files in each library
# ---------------------------------------------------------------------------------------------------------------------------------------
LIST_OBJ_MAIN = $(obj)/model_cloud.o \
		$(obj)/ml_setup.o \
		$(obj)/oe_utils.o \
		$(obj)/oe_run.o

LIST_OBJ_IO =   $(obj)/regrid.o \
		$(obj)/read_icon.o \
		$(obj)/write_output.o

LIST_OBJ_UTILS = $(obj)/types.o \
		 $(obj)/config.o \
		 $(obj)/utils_math.o \
		 $(obj)/utils_fort.o

LIST_OBJ_RTTOVML = $(obj)/rttov_utils.o \
		   $(obj)/rttov_ml.o \
		   $(obj)/interface_rttov.o \
		   $(obj)/rttov_setup.o

LIST_OBJ = $(LIST_OBJ_UTILS) $(LIST_OBJ_RTTOVML) $(LIST_OBJ_IO) $(LIST_OBJ_MAIN) 
# ---------------------------------------------------------------------------------------------------------------------------------------

# List of flags related to each libraries + final flag
# ---------------------------------------------------------------------------------------------------------------------------------------
FLAGS_NCDF = -I$(path_NCDF_INC) -L${path_NCDF_LIB} -lnetcdff -L${path_NCDF_C_LIB} -lnetcdf -Wl,-rpath,${path_NCDF_LIB} -Wl,-rpath,${path_NCDF_C_LIB}
FLAGS_RTTOV = -I${RTTOV_INC_PATH} -L${RTTOV_LIB_PATH} $(RTTOV_LIBS)
FLAG_HDF5= -L${HDF5_LIB} -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz -lm -Wl,-rpath,${HDF5_LIB}
FLAGS_LOCAL = -L$(lib) -l_io -lrttovml -lmain  -lutils

FLAGS_ALL = $(FLAGS_LOCAL) $(FLAGS_RTTOV) $(FLAG_HDF5) $(FLAGS_NCDF)
# ---------------------------------------------------------------------------------------------------------------------------------------

# Make commands
# ---------------------------------------------------------------------------------------------------------------------------------------
install : $(LIST_OBJ)
	ar r $(LIB_UTILS) $(LIST_OBJ_UTILS)
	ar r $(LIB_MAIN) $(LIST_OBJ_MAIN)
	ar r $(LIB_IO) $(LIST_OBJ_IO)
	ar r $(LIB_RTTOVML) $(LIST_OBJ_RTTOVML)
	$(F90) $(F90FLAGS) $(path_main)/$(prog).f90 -o $(prog) $(FLAGS_ALL)

clean:	
	rm -f $(obj)/*.o $(mod)/*.mod $(lib)/*.a
# ---------------------------------------------------------------------------------------------------------------------------------------


## Objects for subroutines in ./src/main 
# ---------------------------------------------------------------------------------------------------------------------------------------
$(obj)/ml_setup.o : $(path_main)/ml_setup.f90
	$(F90) $(F90FLAGS) -c $< -o $@

$(obj)/model_cloud.o : $(path_main)/model_cloud.f90
	$(F90) $(F90FLAGS) -c $< -o $@

$(obj)/oe_run.o : $(path_main)/oe_run.f90
	$(F90) $(F90FLAGS) -c $< -o $@

$(obj)/oe_utils.o : $(path_main)/oe_utils.f90
	$(F90) $(F90FLAGS) -c $< -o $@
# ---------------------------------------------------------------------------------------------------------------------------------------


## Objects for subroutines in ./src/ml_io 
# ---------------------------------------------------------------------------------------------------------------------------------------
$(obj)/write_output.o : $(path_io)/write_output.f90
	$(F90) $(F90FLAGS) -I $(path_NCDF_INC) -c $< -o $@

$(obj)/read_icon.o : $(path_io)/read_icon.f90
	$(F90) $(F90FLAGS) -I $(path_NCDF_INC) -c $< -o $@

$(obj)/regrid.o : $(path_io)/regrid.f90
	$(F90) $(F90FLAGS) -c $< -o $@
# ---------------------------------------------------------------------------------------------------------------------------------------

## Objects for subroutines in ./src/rttov
# ---------------------------------------------------------------------------------------------------------------------------------------
$(obj)/interface_rttov.o : $(path_rttov_ml)/interface_rttov.f90
	$(F90) $(F90FLAGS) -I $(RTTOV_INC_PATH) -I $(RTTOV_MOD_PATH) -L $(RTTOV_LIB_PATH) -c $< -o $@

$(obj)/rttov_ml.o : $(path_rttov_ml)/rttov.f90
	$(F90) $(F90FLAGS) -I $(RTTOV_INC_PATH) -I $(RTTOV_MOD_PATH) -L $(RTTOV_LIB_PATH) -c $< -o $@

$(obj)/rttov_setup.o : $(path_rttov_ml)/rttov_setup.f90
	$(F90) $(F90FLAGS)  -c $< -o $@

$(obj)/rttov_utils.o : $(path_rttov_ml)/rttov_utils.f90
	$(F90) $(F90FLAGS)  -c $< -o $@
# ---------------------------------------------------------------------------------------------------------------------------------------

## Objects for subroutines in ./src/utils
# ---------------------------------------------------------------------------------------------------------------------------------------
$(obj)/types.o : $(path_utils)/types.f90
	$(F90) $(F90FLAGS) -c $< -o $@

$(obj)/config.o : $(path_utils)/config.f90
	$(F90) $(F90FLAGS) -c $< -o $@

$(obj)/utils_math.o : $(path_utils)/utils_math.f90
	$(F90) $(F90FLAGS) -c $< -o $@

$(obj)/utils_fort.o : $(path_utils)/utils_fort.f90 #$(path_utils)/config.f90
	$(F90) $(F90FLAGS) -c $< -o $@
# ---------------------------------------------------------------------------------------------------------------------------------------

