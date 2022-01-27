#!/bin/sh



# '''
# Read simulations input or output and plot it

# '''



# ''' ya probe
# ## output of the rttov with the test-2 
# # python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/output-test-2.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'output-test-2-old'

# ### output 3 channels Odran example with the test-2  channel 2, 7, 32-----------,dom1 dom1, dom_nstreams 8
# python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/outputs_test_3chan_v1.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'outputs_test_3chan_v1'

# ## output of the rttov version anterior (more cleaning up) with the test-2    channel 7 and 32-----------,dom1 dom1, dom_nstreams 4
# python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-msg-7-32.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-msg-7-32'


# ## output of the rttov version anterior (more cleaning up) with the test-2  channel 32 -----------,dom1 dom1, dom_nstreams 4
# python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-msg-32.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-msg-32'

# '''

# -----------,dom1 dom1, dom_nstreams 8
# python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-msg-2-7-32.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-13-msg-2-7-32'


#-----------,dom1 dom1, dom_nstreams 8
# python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-12-msg-1-36.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-12-msg-1-36' --type-simulation 'radiances'

# python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-msg-7-32.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-msg-7-32' --type-simulation 'radiances'


# python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-msg.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-13-msg-without-ls' --type-simulation 'radiances'

python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-msg-converted.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-13-converted-rad-y' --type-simulation 'radiances'

python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-msg-converter-32-33-34.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-13-converted-rad-32-33-34-y' --type-simulation 'radiances'


python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-13-msg-converter-28-35-36.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-13-converted-rad-28-35-36' --type-simulation 'radiances'

python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/VF-output-test-modis-T12.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'VF-output-test-modis-T12' --type-simulation 'radiances'

###input alexandre
python simulation.py --path-OUTPUT-RTTOV '/home/jvillarreal/Documents/phd/github/output-rttov/rttov-msg-alex.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'rttov-msg-alex.nc' --type-simulation 'radiances'




# '''
# python simulation.py --path-ICON '/home/jvillarreal/Documents/phd/dataset/subset_rttov_T12_old.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'subset-data--old' --variable-input 'clc' --dimension-variable 3 


# python simulation.py --path-ICON '/home/jvillarreal/Documents/phd/dataset/subset_rttov_T12.nc' --path-output '/home/jvillarreal/Documents/phd/output' --input-data 'subset-data-vf' --variable-input 'clc' --dimension-variable 3 
# '''
