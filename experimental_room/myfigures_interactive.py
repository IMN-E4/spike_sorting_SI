from probeinterface.plotting import plot_probe
import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np


# visualize drift over time
def plot_drift(peaks, rec_preprocess, peak_locations, name, time_range=None):
    fs = rec_preprocess.get_sampling_frequency()
    fig, ax = plt.subplots()
    
    x = peaks['sample_ind'] / fs
    y = peak_locations['y']
    if time_range is not None:
        mask = (x > time_range[0]) & (x < time_range[1])
        x = x[mask]
        y = peak_locations['y'][mask]
        
    ax.scatter(x, y, s=1, color='k', alpha=0.05)
    ax.set_title(name)

def plot_peaks_axis(rec_preprocess, peak_locations, name, peaks=None, time_range=None):
    # visualize peaks clouds on the probe ( , x, z)
    fs = rec_preprocess.get_sampling_frequency()
    channel_locs = rec_preprocess.get_channel_locations()
    y_max = np.max(channel_locs[:, 1])/2
    y_min = np.min(channel_locs[:, 1])-100

    fig, axs = plt.subplots(ncols=2)

    x = peak_locations['x']
    y = peak_locations['y']

    if (time_range is not None) and (peaks is not None):
        peaks_in_secs = peaks['sample_ind'] / fs
        mask = (peaks_in_secs > time_range[0]) & (peaks_in_secs < time_range[1])
        x = x[mask]
        y = y[mask]


    for number in range(2):
        ax = axs[number]
        plot_probe(rec_preprocess.get_probe(), ax=ax)
        ax.scatter(x, y, color='k', s=1, alpha=0.002)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)
        ax.set_ylim(y_min, y_max)
        y_min += y_max
        y_max = y_max*2

def plot_peaks_activity(peaks, rec_preprocess, peak_locations, name, time_range=None):
    # visualize peaks clouds on the probe
    fs = rec_preprocess.get_sampling_frequency()
    channel_locs = rec_preprocess.get_channel_locations()
    y_max = np.max(channel_locs[:, 1])/2
    y_min = np.min(channel_locs[:, 1])-100
    fig, axs = plt.subplots(ncols=2)
    
    if time_range is not None:
        peaks_in_secs = peaks['sample_ind'] / fs
        mask = (peaks_in_secs > time_range[0]) & (peaks_in_secs < time_range[1])
        peaks = peaks[mask]
    
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

def plot_noise(rec_preprocess, with_contact_color=False, with_interpolated_map=True, time_range=None):
    # visualize noise clouds on the probe
    probe = rec_preprocess.get_probe()
    fs = rec_preprocess.get_sampling_frequency()
    
    if time_range is not None:
        rec_preprocess = rec_preprocess.frame_slice(start_frame=time_range[0]*fs, end_frame=time_range[1]*fs)
        print(rec_preprocess)
        
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
            im.set_clim(0, 20)
            artists = artists + (im,)
        
        ax.set_ylim(y_min, y_max)
        y_min += y_max
        y_max = y_max*2
        ax.set_aspect(0.5)
    fig.colorbar(artists[0])
