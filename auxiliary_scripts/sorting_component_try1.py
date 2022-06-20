from tempfile import tempdir
import spikeinterface.full as si
import numpy as np
from spikeinterface.toolkit import get_noise_levels
import pylab as plt
import shutil
from pathlib import Path
import time
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface import extract_waveforms, WaveformExtractor


job_kwargs = {
    'progress_bar' : True,
    'n_jobs': 10,
    'chunk_duration': '1s',
}


def run_sorting_components(recording, tmp_folder, delete_existing=False):

    # Check temp folder
    if tmp_folder.is_dir() and delete_existing:
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir(exist_ok=True)
 


    # step 1 : noise
    print('Starting Step 1')
    if (tmp_folder / 'noise_levels.npy').exists():
        noise_levels = np.load(tmp_folder / 'noise_levels.npy')
    else:
        noise_levels = get_noise_levels(recording, return_scaled=False)
        np.save(tmp_folder / 'noise_levels.npy', noise_levels)

    # step 2 : peaks
    print('Starting Step 2')
    if (tmp_folder / 'peaks.npy').exists():
        peaks = np.load(tmp_folder / 'peaks.npy')
    else:
        max_peaks_per_channel = 1000
        peaks = detect_peaks(recording, "locally_exclusive",
                             noise_levels=noise_levels, detect_threshold=5, n_shifts=7, local_radius_um=100,
                             **job_kwargs)
        np.save(tmp_folder / 'peaks.npy', peaks)

    # step 3 peak selection
    print('Starting Step 3')
    if (tmp_folder / 'some_peaks.npy').exists():
        some_peaks = np.load(tmp_folder / 'some_peaks.npy')
    else:
        # TODO select better methods
        some_peaks = select_peaks(peaks, 'uniform',  n_peaks=recording.get_num_channels() * 1000, select_per_channel=False)
        np.save(tmp_folder / 'some_peaks.npy', some_peaks)
    
    # step 4 : peak localization
    print('Starting Step 4')
    if (tmp_folder / 'some_peaks_locations.npy').exists():
        some_peaks_locations = np.load(tmp_folder / 'some_peaks_locations.npy')
    else:
        some_peaks_locations = localize_peaks(recording, some_peaks, method='monopolar_triangulation', method_kwargs=dict(optimizer='least_square') , **job_kwargs) #or minimize_with_log_penality
        np.save(tmp_folder /'some_peaks_locations.npy', some_peaks_locations)

    # step 5 : clustering
    print('Starting Step 5')
    if (tmp_folder / 'peak_labels.npy').exists():
        peak_labels = np.load(tmp_folder / 'peak_labels.npy')
    else:
        clustering_path = tmp_folder / 'clustering_path'
        method_kwargs = dict(
            peak_locations=some_peaks_locations,
            hdbscan_params_spatial = {"min_cluster_size" : 20,  "allow_single_cluster" : True, 'metric' : 'l2'},
            probability_thr = 0,
            apply_norm=True,
            #~ debug=True,
            debug=False,
            tmp_folder=clustering_path,
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
        np.save(tmp_folder / 'peak_labels', peak_labels)


    # step 6: construct template average
    print('Starting Step 6')
    if (tmp_folder / 'pre_waveform_extractor').exists():
        we = si.WaveformExtractor.load_from_folder(tmp_folder / 'pre_waveform_extractor')
    else:
        mask = peak_labels >= 0
        sorting_np = si.NumpySorting.from_times_labels(some_peaks['sample_ind'][mask], peak_labels[mask], recording.get_sampling_frequency())

        sorting_folder = tmp_folder / 'pre_sorting'
        pre_sorting = sorting_np.save(folder=sorting_folder)

        wf_folder = tmp_folder / 'pre_waveform_extractor'
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
    if  (tmp_folder / 'final_sorting').exists():
        final_sorting = si.load_extractor(tmp_folder / 'final_sorting')
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

        sorting_folder = tmp_folder / 'final_sorting'
        final_sorting = sorting_np.save(folder=sorting_folder)


    # step 8: create the final waveform extractor
    print('Starting Step 8')
    if (tmp_folder / 'final_waveform_extractor').exists():
        we = si.WaveformExtractor.load_from_folder(tmp_folder / 'final_waveform_extractor')
    else:
        wf_folder = tmp_folder / 'final_waveform_extractor'
        we = extract_waveforms(recording, final_sorting, 
                    wf_folder,
                    load_if_exists=False,
                    precompute_template=['average', 'std'],
                    ms_before=1.5, ms_after=2.5,
                    max_spikes_per_unit=1000,
                    overwrite=False,
                    return_scaled=False,
                    dtype=None,
                    use_relative_path=True,
                    **job_kwargs)
    
    print('Starting to run compute_spike_amplitudes')
    si.compute_spike_amplitudes(we, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=30,
    return_scaled=True, load_if_exists=True) 

    # compute PCs
    # print('Starting to run compute_principal_components and calculate_pc_metrics')
    # pc = si.compute_principal_components(we, load_if_exists=True,
    #             n_components=2, mode='by_channel_local')

    # compute metrics
    print('Starting to run compute_quality_metrics')
    metrics = si.compute_quality_metrics(we, load_if_exists=False, metric_names=['snr', 'isi_violation', 'num_spikes', 'firing_rate', 'presence_ratio'])
   

    # export report
    print('Starting to run export_report')
    report_folder = tmp_folder / '_report'
    si.export_report(we, report_folder, remove_if_exists=True,
            chunk_size=30000, n_jobs=30, progress_bar=True, verbose=True)
    print("It's finished")


def _test_run_sorting_components():
    base_folder = Path('/data1/ArthursLab/RomansData/Cerebellum/Neuropixel_Recording_18_03_2022_Cb/')
    data_folder = base_folder / 'Rec_18_03_2022_cb_ansth_g0/'

    
    workingdir = base_folder / 'test_sorting'
    preproc_folder = workingdir / 'cached_recording'
    if preproc_folder.exists():
        rec = si.load_extractor(preproc_folder)
    else:
        rec = si.read_spikeglx(data_folder, stream_id='imec0.ap')
        fs = rec.get_sampling_frequency()
        #rec = rec.frame_slice(start_frame=0, end_frame=fs*600)
        rec = si.bandpass_filter(rec, freq_min=300., freq_max=6000.)
        rec = si.common_reference(rec, reference='local',
                                            local_radius=(50, 100), operator='median')
        rec = rec.save(format = 'binary', folder=preproc_folder, **job_kwargs)
        #rec = rec.save(format='zarr', folder=preproc_folder, **job_kwargs)




    # answer = input('temp_folder already exists. Delete and run from scratch? [Y/n]')
    # print(answer)
    # delete_existing = (answer.lower() == 'y')
    # run_sorting_components(rec, workingdir,delete_existing=delete_existing)

    run_sorting_components(rec, workingdir, delete_existing=False)
    

if __name__ == '__main__':
    _test_run_sorting_components()
