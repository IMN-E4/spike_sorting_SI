import matplotlib.pyplot as plt

import spikeinterface.full as si

from pathlib import Path

import probeinterface as pi

import numpy as np

base_folder = Path('/home/arthur/Documents/SpikeSorting/Test_20210518/') 

data_folder = base_folder / 'raw_awake'

recording = si.read_spikeglx(data_folder, stream_id='imec0.ap')

probe = pi.read_spikeglx(data_folder / 'raw_awake_01_g0_t0.imec0.ap.meta')

#recording = recording.frame_slice(0, 3000000)
#print(recording)


recording = recording.set_probe(probe)
print(recording)

recording_f = si.bandpass_filter(recording, freq_min=300, freq_max=6000)
print(recording_f)

noise_levels = si.get_noise_levels(recording_f)


from spikeinterface.sortingcomponents import detect_peaks

peaks = detect_peaks(recording_f, method='locally_exclusive', 
        peak_sign='neg', detect_threshold=5, n_shifts=5, 
        local_radius_um=100,
        noise_levels=noise_levels,
        random_chunk_kwargs={},
        outputs='numpy_compact',
        n_jobs=8, progress_bar=True, chunk_size=30000, )

np.save('peaks.npy', peaks)


