
"""
This will compute and save the pre-spike sorting checks.

yep: it works!!!


Author: Samuel, Eduarda
"""



# Packages
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
import numpy as np
import os
import glob
plt.rcParams["figure.figsize"] = (20, 12)
from probeinterface.plotting import plot_probe
from spikeinterface.sortingcomponents import detect_peaks
from spikeinterface.sortingcomponents import localize_peaks
from spikeinterface.sortingcomponents import (estimate_motion, make_motion_histogram,
    compute_pairwise_displacement, compute_global_displacement)

from spikeinterface.sortingcomponents import correct_motion_on_peaks, correct_motion_on_traces



# Paths
base_input_folder = Path('/media/e4/data1/CorinnasData/raw_data/')




# global kwargs for parallel computing
job_kwargs = dict(
    n_jobs=40,
    chunk_memory='10M',
    progress_bar=True,
)

def find_paths(main_dir, bird, **kwargs):
    """ Flexible way to find files in subdirectories based on keywords
    Parameters
    ----------
    main_dir: str
        Give the main directory where the subjects' folders are stored

    subject: str
        Give the name of the recording to be analyzed

    **kwargs: str
        Give keywords that will be used in the filtering of paths

    Examples
    -------
    Ex.1
    find_paths(main_dir='/home/arthur/Documents/SpikeSorting/',
               bird='Test_20210518/',
               key1='small')
    Returns
    -------
    updatedfilter: list
        List with path strings

    """

    # Check if arguments are in the correct type
    assert isinstance(main_dir, str), 'Argument must be str'
    assert isinstance(bird, str), 'Argument must be str'

    # Filtering step based on keywords
    firstfilter = glob.glob(main_dir + '/' + bird + '/**/*.imec0.ap.bin',
                            recursive=True)

    updatedfilter = firstfilter

    for _, value in kwargs.items():
        # Update list accoring to key value
        updatedfilter = list(filter(lambda path: value in path, updatedfilter))

    final_paths = [Path(path).parent for path in updatedfilter]

    return final_paths



# Get recordings
def get_recordings(path):
    """ Function to get the recordings and set the probe from base_input_folder
    
    """

    rec = si.read_spikeglx(path, stream_id='imec0.ap')   

    return rec

def probe_check(rec):
    fig, ax = plt.subplots()
    plot_probe(rec.get_probe(), ax=ax)
    ax.set_ylim(-50, 400)
    plt.show()


def apply_preprocessing(rec, ref_type='local' ,operator='median', save=True):
    rec_filtered = si.bandpass_filter(rec, freq_min=300., freq_max=6000.)
    rec_preprocessed = si.common_reference(rec_filtered, reference=ref_type,
                                       local_radius=(50, 100), operator=operator)
    if save == True:
        rec_preprocessed.save(folder=preprocess_folder, **job_kwargs)
    
    return rec_preprocessed

def get_peaks(rec_preprocessed, noise_levels, save=True):
    peaks = detect_peaks(rec_preprocessed, method='locally_exclusive', local_radius_um=50,
                 peak_sign='neg', detect_threshold=5, n_shifts=5,
                 noise_levels=noise_levels, **job_kwargs)
    if save==True:
        np.save(peak_folder / 'peaks.npy', peaks)
    return peaks

def get_peaks_location(rec_preprocessed, peaks, method, save=True): #methods can be: 'monopolar_triangulation' or 'center_of_mass'
    if method=='center_of_mass':
        method_kwargs={'local_radius_um': 100.}
    elif method=='monopolar_triangulation':
        method_kwargs={'local_radius_um': 100., 'max_distance_um': 1000.}

    peak_locations = localize_peaks(rec_preprocessed, peaks,
                   ms_before=0.3, ms_after=0.6,
                   method=method, method_kwargs=method_kwargs,
                   **job_kwargs)

    if save==True:            
        np.save(peak_folder / 'peak_locations_' + method + '.npy', peak_locations)
    
    return peak_locations



# Main function
def run_all_pipeline(path, 
                    check_probe=True, 
                    plot_timeseries=True, 
                    check_noise=True, 
                    loc_methods=['monopolar_triangulation', 'center_of_mass'],
                    save_preproc=True, 
                    save_peaks=True,
                    save_loc_plots=True
                    ):

    # Get recording
    rec = get_recordings(path)

    # Choose to visualize or not probe
    if check_probe==True:
        probe_check(rec)

    # Apply preprocessing on rec
    rec_preprocessed = apply_preprocessing(rec, ref_type='local', operator='median', save=save_preproc)

    # Choose to visualize or not the timeseries
    if plot_timeseries==True:
        fig, ax = plt.subplots()
        si.plot_timeseries(rec_preprocessed,
                        time_range=(300, 310),
                        # channel_ids = rec_preprocessed.channel_ids[60:70],
                        ax=ax)

    # Get noise levels (if peaks already computed, no need to get noise levels again)
    if check_noise==True:
        noise_levels = si.get_noise_levels(rec_preprocessed, return_scaled=False)
        fig, ax = plt.subplots()
        ax.hist(noise_levels, bins=np.arange(0, 15, .5))
        plt.show()
    
    # Get peak values
    if peak_folder / 'peaks.npy':   # check if this works 
        # load back
        peaks = np.load(peak_folder / 'peaks.npy')
    else: 
        peaks = get_peaks(rec_preprocessed, noise_levels, save=save_peaks)

    for method in loc_methods:   
        # Get peak locations
        if peak_folder / 'peak_locations_' + method + '.npy': # check if this works
            # load back
            peak_locations = np.load(peak_folder / 'peaks.npy')

        else:
            # Compute peaks
            peak_locations = get_peaks_location(rec_preprocessed, peaks, method, save=True)

        # Plot peaks on probe
        probe = rec_preprocessed.get_probe()
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_probe(probe, ax=ax)
        ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.02)
        # ax.set_ylim(2400, 2900)
        ax.set_ylim(1500, 2500)
        plt.title(method)
        plt.show()
        if save_loc_plots==True:
            fig.savefig(peak_folder / (method + '_scat_peaks.png'))
            
        # Plot raster peaks
        fig, ax = plt.subplots()
        x = peaks['sample_ind'] / rec_preprocessed.get_sampling_frequency()
        y = peak_locations['y']
        ax.scatter(x, y, s=1, color='k', alpha=0.05)
        if save_loc_plots==True:
            fig.savefig(peak_folder / (method + '_motion_raster.png')) # check this works

        # Motion estimate: rigid decentralized
        bin_um = 2
        bin_duration_s=5.
        motion_histogram, temporal_bins, spatial_bins = make_motion_histogram(rec_preprocessed, peaks,
            peak_locations=peak_locations, 
            bin_um=bin_um, bin_duration_s=bin_duration_s,
            direction='y', weight_with_amplitude=False)

        fig, ax = plt.subplots()
        extent = (temporal_bins[0], temporal_bins[-1], spatial_bins[0], spatial_bins[-1])
        im = ax.imshow(motion_histogram.T, interpolation='nearest',
                            origin='lower', aspect='auto', extent=extent)
        im.set_clim(0, 15)
        # ax.set_ylim(1300, 2500)
        ax.set_xlabel('time[s]')
        ax.set_ylabel('depth[um]')
        fig.colorbar(im)

        if save_loc_plots==True:
            fig.savefig(peak_folder / (method + '_motion_histogram.png')) # check this works

        # Pairwise displacement
        if (method + '_pairwise_displacement_conv2d.npy') in peak_folder:
            pairwise_displacement = np.load(peak_folder / (method + '_pairwise_displacement_conv2d.npy')
        else:
            pairwise_displacement = compute_pairwise_displacement(motion_histogram, bin_um, method='conv2d', progress_bar=True)
            np.save(peak_folder / (method + '_pairwise_displacement_conv2d.npy'), pairwise_displacement)  # check this works
        
        fig, ax = plt.subplots()
        extent = (temporal_bins[0], temporal_bins[-1], temporal_bins[0], temporal_bins[-1])
        # extent = None
        im = ax.imshow(pairwise_displacement, interpolation='nearest',
                            cmap='PiYG', origin='lower', aspect='auto', extent=extent)
        im.set_clim(-40, 40)
        ax.set_aspect('equal')
        fig.colorbar(im)
        if save_loc_plots==True:
            fig.savefig(peak_folder / (method + '_pairwise_displacement.pdf'))  # check this works

        motion = compute_global_displacement(pairwise_displacement)
        fig, ax = plt.subplots()
        ax.plot(temporal_bins[:-1], motion)
        if save_loc_plots==True:
            fig.savefig(peak_folder / (method + '_motion_estimated_rigid.pdf'))


        # Single function motion computation? Can I exclude the lines above?
        if (method + 'motion_rigid.npy') & (method + 'motion_rigid.npy') in path: #check this
            print('Rigid motion files already exist')
            motion = np.load(peak_folder / method + 'motion_rigid.npy')
            temporal_bins = np.load(peak_folder/ method + 'temporal_bins_rigid.npy')
        else:
            print('Rigid motion files do not exist')
            motion, temporal_bins, spatial_bins = estimate_motion(rec_preprocessed, peaks, peak_locations=peak_locations,
                    direction='y', bin_duration_s=5., bin_um=10.,
                    method='decentralized_registration', method_kwargs={},
                    non_rigid_kwargs=None, 
                    progress_bar=True, verbose=True)
            np.save(peak_folder / method + 'motion_rigid.npy', motion)
            np.save(peak_folder / method + 'temporal_bins_rigid.npy', temporal_bins)

        fig, ax = plt.subplots()
        x = peaks['sample_ind'] / rec_preprocessed.get_sampling_frequency()
        y = peak_locations['y']
        ax.scatter(x, y, s=1, color='k', alpha=0.05)
        ax.set_ylim(1300, 2500)
        ax.plot(temporal_bins[:-1], motion + 2000, color='r')
        ax.set_xlabel('times[s]')
        ax.set_ylabel('motion [um]')
        if save_loc_plots==True:
            fig.savefig(peak_folder / method + 'pairwise_displacement.pdf')  # check this works

        if [method + 'motion_non_rigid.npy', method + 'temporal_bins_non_rigid.npy', method + 'spatial_bins_non_rigid.npy'] in path: #check this
            print('Non-rigid motion files already exist')
            motion = np.load(peak_folder / method + 'motion_non_rigid.npy')
            temporal_bins = np.load(peak_folder / method + 'temporal_bins_non_rigid.npy')
            spatial_bins = np.load(peak_folder / method + 'spatial_bins_non_rigid.npy')
        else:
            print('Rigid motion files do not exist')
            motion, temporal_bins, spatial_bins = estimate_motion(rec_preprocessed, peaks, peak_locations=peak_locations,
                    direction='y', bin_duration_s=5., bin_um=10.,
                    method='decentralized_registration', method_kwargs={},
                    non_rigid_kwargs=dict(bin_step_um=100),
                    progress_bar=True, verbose=False)

            np.save(peak_folder / method + 'motion_non_rigid.npy', motion)
            np.save(peak_folder / method + 'temporal_bins_non_rigid.npy', temporal_bins)
            np.save(peak_folder / method + 'spatial_bins_non_rigid.npy', spatial_bins)

        fs = rec_preprocessed.get_sampling_frequency()

        fig, ax = plt.subplots()
        ax.scatter(peaks['sample_ind'] / fs, peak_locations['y'], color='k', s=0.1, alpha=0.05)
        for i, _ in enumerate(spatial_bins):
            # several motion vector
            ax.plot(temporal_bins[:-1], motion[:, i] + spatial_bins[i], color='r')
        ax.set_ylim(1300, 2500)
        ax.set_xlabel('times[s]')
        ax.set_ylabel('motion [um]')
        if save_loc_plots==True:
            fig.savefig(peak_folder / method + 'non_rigid_vis.png')

if __name__ == '__main__':

    for path in find_paths(main_dir=base_input_folder.as_posix(), bird='bird1'):
        print('Working on: ', path)
        global preprocess_folder
        preprocess_folder = path / '/preproc_output/rec_preprocessed'
        global peak_folder
        peak_folder = path / '/preproc_output/rec_peaks'
        peak_folder.mkdir(exist_ok=True)

        run_all_pipeline(path, 
                        check_probe=True, 
                        plot_timeseries=True, 
                        check_noise=True, 
                        loc_methods=['monopolar_triangulation', 'center_of_mass'],
                        save_preproc=True, 
                        save_peaks=True,
                        save_loc_plots=True
                        )
        