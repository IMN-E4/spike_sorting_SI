from probeinterface.plotting import plot_probe
import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np


# visualize drift over time
def plot_drift(peaks, rec_preprocess, peak_locations, name, figure_folder, 
              motion=None, temporal_bins=None, spatial_bins=None, alpha=0.02):
    fig, ax = plt.subplots()
    x = peaks['sample_index'] / rec_preprocess.get_sampling_frequency()
    y = peak_locations['y']
    ax.scatter(x, y, s=1, color='k', alpha=alpha)
    ax.set_title(name)

    if motion is not None:
        for i in range(motion.shape[1]):
            ax.plot(temporal_bins, motion[:, i] + spatial_bins[i], color='m', alpha=0.8)
    if figure_folder is not None:
        fig.savefig(figure_folder / 'peak_drift.png')

def plot_peaks_axis(rec_preprocess, peak_locations, name, figure_folder):
    # visualize peaks clouds on the probe ( , x, z)
    channel_locs = rec_preprocess.get_channel_locations()
    y_max = np.max(channel_locs[:, 1])/2
    y_min = np.min(channel_locs[:, 1])-100

    fig, axs = plt.subplots(ncols=2)
    for number in range(2):
        ax = axs[number]
        plot_probe(rec_preprocess.get_probe(), ax=ax)
        ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.002)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)
        ax.set_ylim(y_min, y_max)
        y_min += y_max
        y_max = y_max*2


    fig.savefig(figure_folder / 'peak_locations.png')

def plot_peaks_activity(peaks, rec_preprocess, peak_locations, name, figure_folder):
    # visualize peaks clouds on the probe
    channel_locs = rec_preprocess.get_channel_locations()
    y_max = np.max(channel_locs[:, 1])/2
    y_min = np.min(channel_locs[:, 1])-100
    fig, axs = plt.subplots(ncols=2)
    for number in range(2):
        ax = axs[number]
        si.plot_peak_activity_map(rec_preprocess, peaks,
            with_contact_color=False,
            with_interpolated_map=True,
            ax=ax,
        )
        ax.set_ylim(y_min, y_max)
        y_min += y_max
        y_max = y_max*2
        ax.set_aspect(0.5)

    fig.savefig(figure_folder / 'peak_activity.png')


def plot_noise(rec_preprocess, figure_folder, with_contact_color=False, with_interpolated_map=True):
    # visualize noise clouds on the probe
    probe = rec_preprocess.get_probe()
    noise_levels_scaled = si.get_noise_levels(rec_preprocess, return_scaled=True)
    channel_locs = rec_preprocess.get_channel_locations()
    y_max = np.max(channel_locs[:, 1])/2
    y_min = np.min(channel_locs[:, 1])-100

    fig, axs = plt.subplots(ncols=2)
    for number in range(2):
        ax = axs[number]
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
        
        ax.set_ylim(y_min, y_max)
        y_min += y_max
        y_max = y_max*2
        ax.set_aspect(0.5)

    fig.colorbar(artists[0])
    fig.savefig(figure_folder / 'noise_plot.png')