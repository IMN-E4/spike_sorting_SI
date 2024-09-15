from pathlib import Path

# Paths
root = Path('/nas/')
target_folder = Path('Neuropixel_Recordings')

base_folder = root/target_folder


# Envelope computation
min_freq_envelope = 1000 # Hz
max_freq_envelope = 8000 # Hz
envelope_kernel_size = 1000 # points
song_channel_name = 'nidq#XA0'