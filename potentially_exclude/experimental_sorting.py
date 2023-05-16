import shutil
import numpy as np

import spikeinterface.full as si


from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks



def run_experimental_sorting(recording, output_folder=None, delete_existing=False, job_kwargs={}):
    # Check temp folder
    if output_folder.is_dir() and delete_existing:
        shutil.rmtree(output_folder)
    output_folder.mkdir(exist_ok=True)
 


    # step 1 : noise
    print('Starting Step 1')
    if (output_folder / 'noise_levels.npy').exists():
        noise_levels = np.load(output_folder / 'noise_levels.npy')
    else:
        noise_levels = si.get_noise_levels(recording, return_scaled=False)
        np.save(output_folder / 'noise_levels.npy', noise_levels)

    # step 2 : peaks
    print('Starting Step 2')
    if (output_folder / 'peaks.npy').exists():
        peaks = np.load(output_folder / 'peaks.npy')
    else:
        max_peaks_per_channel = 1000
        peaks = detect_peaks(recording, "locally_exclusive",
                             noise_levels=noise_levels, detect_threshold=5, n_shifts=7, local_radius_um=100,
                             **job_kwargs)
        np.save(output_folder / 'peaks.npy', peaks)

    # step 3 peak selection
    print('Starting Step 3')
    if (output_folder / 'some_peaks.npy').exists():
        some_peaks = np.load(output_folder / 'some_peaks.npy')
    else:
        # TODO select better methods
        some_peaks = select_peaks(peaks, 'uniform',  n_peaks=recording.get_num_channels() * 1000, select_per_channel=False)
        np.save(output_folder / 'some_peaks.npy', some_peaks)
    
    # step 4 : peak localization
    print('Starting Step 4')
    if (output_folder / 'some_peaks_locations.npy').exists():
        some_peaks_locations = np.load(output_folder / 'some_peaks_locations.npy')
    else:
        some_peaks_locations = localize_peaks(recording, some_peaks, method='monopolar_triangulation', method_kwargs=dict(optimizer='least_square') , **job_kwargs) #or minimize_with_log_penality
        np.save(output_folder /'some_peaks_locations.npy', some_peaks_locations)

    # step 5 : clustering
    print('Starting Step 5')
    if (output_folder / 'peak_labels.npy').exists():
        peak_labels = np.load(output_folder / 'peak_labels.npy')
    else:
        clustering_path = output_folder / 'clustering_path'
        method_kwargs = dict(
            peak_locations=some_peaks_locations,
            hdbscan_params_spatial = {"min_cluster_size" : 20,  "allow_single_cluster" : True, 'metric' : 'l2'},
            probability_thr = 0,
            apply_norm=True,
            #~ debug=True,
            debug=False,
            output_folder=clustering_path,
            n_components_by_channel=4,
            n_components=4,
            job_kwargs = job_kwargs,
            #waveform_mode="shared_memory",
            waveform_mode="memmap",
        )

        t0 = time.perf_counter()
        labels, peak_labels = find_cluster_from_peaks(recording, some_peaks, 
            method='sliding_hdbscan', method_kwargs=method_kwargs)
        t1 = time.perf_counter()
        print('sliding_hdbscan', t1 -t0)
        np.save(output_folder / 'peak_labels', peak_labels)


    # step 6: construct template average
    print('Starting Step 6')
    if (output_folder / 'pre_waveform_extractor').exists():
        we = si.WaveformExtractor.load_from_folder(output_folder / 'pre_waveform_extractor')
    else:
        mask = peak_labels >= 0
        sorting_np = si.NumpySorting.from_times_labels(some_peaks['sample_ind'][mask], peak_labels[mask], recording.get_sampling_frequency())

        sorting_folder = output_folder / 'pre_sorting'
        pre_sorting = sorting_np.save(folder=sorting_folder)

        wf_folder = output_folder / 'pre_waveform_extractor'
        we = extract_waveforms(recording, pre_sorting, 
                    wf_folder,
                    load_if_exists=False,
                    precompute_template=['average'],
                    ms_before=1.5, ms_after=2.5,
                    max_spikes_per_unit=1000,
                    overwrite=False,
                    return_scaled=False,
                    dtype=None,
                    use_relative_path=True,
                    **job_kwargs)

    # step 7: template matching
    print('Starting Step 7')
    if  (output_folder / 'final_sorting').exists():
        final_sorting = si.load_extractor(output_folder / 'final_sorting')
    else:
        print(we)
        print(we.sorting)
        method_kwargs = {
            'waveform_extractor' : we,
            'noise_levels' : noise_levels,
        }


       
        spikes = find_spikes_from_templates(recording, method='circus-omp', method_kwargs=method_kwargs,
                                            **job_kwargs)

        sorting_np = si.NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], recording.get_sampling_frequency())
        print(sorting_np)

        sorting_folder = output_folder / 'final_sorting'
        final_sorting = sorting_np.save(folder=sorting_folder)



    return sorting