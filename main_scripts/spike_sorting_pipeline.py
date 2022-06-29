"""
This is the 'new' pipeline for neuropixel data.
June 2022

This run several sorter on NP data.

preprocess recording are computed on local cache and then deleted
sorting folder are also deleted.



Author: Samuel, Eduarda
"""

# Packages
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


from probeinterface.plotting import plot_probe
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

from datetime import datetime

# this centralise params at the same place
from params import *
from experimental_sorting import run_experimental_sorting
from recording_list import recording_list
from myfigures import *


########### Prep Functions
def apply_preprocess(rec):
    """
    Apply the lazy preprocessing chain.
    """
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    rec = si.common_reference(rec, reference='local', local_radius=(50, 100))
    return rec

def fix_time_range(spikeglx_folder, time_range):
    rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
    print(rec)

    fs = rec.get_sampling_frequency()
 
    if time_range is None:
        duration = rec.get_num_frames() / rec.get_sampling_frequency()
        time_range = (0., duration)
        

    else:
        time_range = tuple(float(e) for e in time_range)
 
        frame_range = (int(t * fs) for t in time_range)
        rec = rec.frame_slice(*frame_range)

    return rec, time_range
    

def get_workdir_folder(spikeglx_folder, time_range):
    """
    Common function to get workdir
    """
    
    name = spikeglx_folder.stem
    implementation_name = spikeglx_folder.parents[1].stem
    time_stamp = datetime.now().strftime('%Y-%m')
    working_folder = base_sorting_cache_folder / implementation_name / 'sorting_cache' / f'{time_stamp}-{name}-{time_range[0]}to{time_range[1]}'
    working_folder.mkdir(exist_ok=True, parents=True)
    print(working_folder)
    
    return working_folder


########### Preprocess & Check
def get_preprocess_recording(spikeglx_folder, time_range=None):
    """
    Function to get preprocessed recording.
    """
    print(f'first time range is {time_range}')
    rec, time_range = fix_time_range(spikeglx_folder, time_range=time_range)

    print(f'second time range is {time_range}')
    working_folder = get_workdir_folder(spikeglx_folder, time_range=time_range)
    
    # preprocessing
    preprocess_folder = working_folder / 'preprocess_recording'
    if preprocess_folder.exists():
        print('Already preprocessed')
        rec_preprocess = si.load_extractor(preprocess_folder)
    else:
        print('Run/save preprocessing')
        rec_preprocess = apply_preprocess(rec)
        rec_preprocess.dump_to_json(working_folder / 'preprocess.json')
        rec_preprocess = rec_preprocess.save(format='binary', folder=preprocess_folder, **job_kwargs)

    return rec_preprocess, working_folder


def run_pre_sorting_checks(spikeglx_folder, time_range=None):

    # print(spikeglx_folder)

    rec_preprocess, working_folder = get_preprocess_recording(spikeglx_folder, time_range=time_range)
    print(rec_preprocess)

    noise_file = working_folder / 'noise_levels.npy'
    if noise_file.exists():
        noise_levels = np.load(noise_file)
    else:
        noise_levels = si.get_noise_levels(rec_preprocess, return_scaled=False)
        np.save(noise_file, noise_levels)

    peaks_file = working_folder / 'peaks.npy'
    if peaks_file.exists():
        peaks = np.load(peaks_file)
    else:
        peaks = detect_peaks(rec_preprocess, noise_levels=noise_levels, **peak_detection_params, **job_kwargs)
        np.save(peaks_file, peaks)
    print(peaks.shape)

    location_file = working_folder / 'peak_locations.npy'
    if location_file.exists():
        peak_locations = np.load(location_file)
    else:
        peak_locations = localize_peaks(rec_preprocess, peaks, **peak_location_params, **job_kwargs)
        np.save(location_file, peak_locations)
    print(peak_locations.shape)

    name = Path(spikeglx_folder).stem
    
    figure_folder = working_folder / 'figures'
    figure_folder.mkdir(exist_ok=True, parents=True)

    plot_drift(peaks, rec_preprocess, peak_locations, name, figure_folder)
    plot_peaks_axis(rec_preprocess, peak_locations, name, figure_folder)
    plot_peaks_activity(peaks, rec_preprocess, peak_locations, name, figure_folder)
    plot_noise(rec_preprocess, figure_folder, with_contact_color=False, with_interpolated_map=True)


########### Run sorting
def run_sorting_pipeline(spikeglx_folder, time_range=None):

    rec_preprocess, working_folder = get_preprocess_recording(spikeglx_folder, time_range=time_range)

    # run some sorters
    for sorter_name, params in sorters.items():
        sorting_folder = working_folder / f'sorting_{sorter_name}'
        if sorting_folder.exists():
            print(f'{sorter_name} already computed ')
            sorting = si.load_extractor(sorting_folder)
        else:
            if sorter_name != 'experimental_sorter1':
                sorting = si.run_sorter(sorter_name, rec_preprocess,
                                    output_folder=working_folder / f'raw_sorting_{sorter_name}',
                                    delete_output_folder=True,
                                    verbose=True,
                                    **params
                                    )
            else:
                sorting = run_experimental_sorting(rec_preprocess,
                                                output_folder=working_folder / f'raw_sorting_{sorter_name}',
                                                job_kwargs=job_kwargs,
                                                **params)
            print(sorting)
            sorting = sorting.save(format='npz', folder=working_folder / f'sorting_{sorter_name}')


    # extract waveforms and compute some metrics
    for sorter_name, params in sorters.items():
        sorting_folder = working_folder / f'sorting_{sorter_name}'
        sorting = si.load_extractor(sorting_folder)

        wf_folder = working_folder / f'waveforms_{sorter_name}'
        we = si.extract_waveforms(rec_preprocess, sorting, folder=wf_folder,
                    load_if_exists=True, **waveform_params, **job_kwargs)
        print(we)

        si.compute_spike_amplitudes(we,  load_if_exists=True, **amplitude_params, **job_kwargs)

        metrics_list = ['snr', 'isi_violation', 'num_spikes', 'firing_rate', 'presence_ratio']
        si.compute_quality_metrics(we, load_if_exists=False, metric_names=metrics_list)


    # report : this is super slow!!!
    for sorter_name, params in sorters.items():
        report_folder = working_folder / f'report_{sorter_name}'
        if not report_folder.exists():
            wf_folder = working_folder / f'waveforms_{sorter_name}'
            we = si.WaveformExtractor.load_from_folder(wf_folder)
            si.export_report(we, report_folder, remove_if_exists=False, **job_kwargs)




########### Post-processing
# def run_postprocessing_sorting(...):
    # compute waveforms for clean
    #  /data1

    # copy back to the NASS




########### Tests
def test_run_sorting_pipeline():
    # need for docker debug
    os.environ["SPIKEINTERFACE_DEV_PATH"] = '/home/analysis_user/Python-related/GitHub/spikeinterface'
    # print(os.getenv('SPIKEINTERFACE_DEV_PATH'))
    
    spikeglx_folder = base_input_folder / 'Imp_10_11_2021/Recordings/Rec_2_19_11_2021_g0'
    print(spikeglx_folder)
    time_range = None
    run_sorting_pipeline(spikeglx_folder, time_range=time_range)



def test_run_pre_sorting_checks():    
    spikeglx_folder = base_input_folder / 'Imp_10_11_2021/Recordings/Rec_2_19_11_2021_g0'
    # spikeglx_folder = base_input_folder / 'Imp_10_11_2021/Recordings/Rec_1_18_11_2021_g0'
    print(spikeglx_folder)
    run_pre_sorting_checks(spikeglx_folder, time_range=None)



def run_all():
    for implementation_name, name, time_range in recording_list:
        spikeglx_folder = base_input_folder / implementation_name / 'Recordings' / name
        print(spikeglx_folder)
        # run_pre_sorting_checks(spikeglx_folder, time_range=time_range)

        # run_sorting_pipeline(spikeglx_folder, time_range=time_range)

        # run_postprocessing_sorting()




if __name__ == '__main__':
    test_run_sorting_pipeline()
    # test_run_pre_sorting_checks()

    # run_all()




