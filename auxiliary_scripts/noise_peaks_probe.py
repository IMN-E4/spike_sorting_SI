import spikeinterface.full as si
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from probeinterface.plotting import plot_probe

from probeinterface.plotting import plot_probe

rec_path = Path('/media/e4/data1/CorinnasData/raw_data/bird1/')

preproc_path = rec_path / 'raw_awake_preprocessed/'

peaks_path = rec_path / 'raw_awake_peaks/'

sorting_folder = rec_path / 'spike_sorting'

rec = si.load_extractor(preproc_path)
peaks = np.load(peaks_path / 'peaks.npy')

print(rec)
# print(peaks)

def plot_peak_activity(rec):
    fig, ax = plt.subplots()
    si.plot_peak_activity_map(rec, peaks,
        with_contact_color=False,
        with_interpolated_map=True,
        ax=ax,
        # bin_duration_s=30.,   # this should do animation, work in progress
    )

    # ax.set_ylim(-50, 1000)
    # ax.set_xlim(-100, 100)
    plt.show()


def plot_noise(rec, with_contact_color=False, with_interpolated_map=True, ax=None):
    probe = rec.get_probe()

    if ax is None:
        fig, ax= plt.subplots()

    noise_levels_scaled = si.get_noise_levels(rec, return_scaled=True)

    artists = ()
    if with_contact_color:
        poly, poly_contour = plot_probe(probe, ax=ax, contacts_values=noise_levels_scaled,
        probe_shape_kwargs={'facecolor': 'w', 'alpha': .1},
        contacts_kargs={'alpha': 1.},
        show_channel_on_click=True
        )
        artists = artists + (poly, poly_contour)

    if with_interpolated_map:
        image, xlims, ylims = probe.to_image(noise_levels_scaled, pixel_size=0.5,
        num_pixel=None, method='linear',
        xlims=None, ylims=None)
        im = ax.imshow(image, extent=xlims + ylims, origin='lower', alpha=0.5)
        artists = artists + (im,)

    fig.colorbar(artists[0])
    plt.show() 



## aditional commands that might be useful one day
    # print(sorting.unit_ids)
    # # sorting.get_unit_spike_train(unit_id=0, segment_index=0)
    
    # location = rec.get_channel_locations()
    # plt.scatter(location[:,0], location[:,1])
    # plt.show()

    # # probe interface, show channel on click

plot_peak_activity(rec)