from pathlib import Path

#################
# Probe params  #
#################
probe_type = 'ASSY-236-H5'
amp_type = 'cambridgeneurotech_mini-amp-64'

# Params
bandpass_range = (1,300)
order_by_depth = True  # probe tip to probe upper part

# Choose sources:
raw_recording = True
mic_spectrogram = False
bandpassed_recording = False
sorting = True

# Paths
base_folder = Path('/nas/Cambridge_Recordings/')