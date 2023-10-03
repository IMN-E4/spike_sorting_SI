from pathlib import Path

#################
# Probe params  #
#################
probe_type = 'ASSY-236-H5'
amp_type = 'cambridgeneurotech_mini-amp-64'

# Params
bandpass_range = (300,6000)
order_by_depth = True  # probe tip to probe upper part

# Paths
base_folder = Path('/nas/Cambridge_Recordings/')
path_to_database = Path('/home/eduarda/python-related/github-repos/spike_sorting_with_samuel/main_pipeline/cambridge-pipeline/viztools/from_sniffer.csv')