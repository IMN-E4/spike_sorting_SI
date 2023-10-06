from pathlib import Path
# Params
bandpass_range = (300,6000)
order_by_depth = True  # probe tip to probe upper part

# Paths
base_folder = Path('/nas/Neuropixel_Recordings/')
path_to_database = Path('/home/eduarda/python-related/github-repos/spike_sorting_with_samuel/main_pipeline/neuropixel-pipeline/viztools/from_sniffer.csv')