import spikeinterface.full as si
import os
from pathlib import Path
import numpy as np

# Paths
base_folder = Path('/media/e4/data1/RomansData/bird1/') 

folders = os.listdir(base_folder)

for folder in folders[1:]:
    rec_nidq = si.SpikeGLXRecordingExtractor(base_folder / folder, stream_id='nidq') # microphone
    trace = rec_nidq.get_traces(segment_index=0, channel_ids=['nidq#XA0'], return_scaled=True)
    np.save(base_folder / folder / 'song.npy', trace)