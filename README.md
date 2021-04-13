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
   ../build/rttov_compile.sh
     



