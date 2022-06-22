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

import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

# this centralise params at the same place
from params import *

from experimental_sorting import run_experimental_sorting


def apply_preprocess(rec):
    """
    Apply the lazy preprocessing chain.
    """
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    rec = si.common_reference(rec, reference='local', local_radius=(50, 100))
    return rec



def run_sorting_pipeline(spikeglx_folder, time_range=None):
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
    
    # print(rec)


    name = spikeglx_folder.stem
    implementation_name = spikeglx_folder.parents[1].stem

    working_folder = base_sorting_cache_folder / implementation_name / f'{name} - {time_range[0]}to{time_range[1]}'

    working_folder.mkdir(exist_ok=True, parents=True)
    

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

    # run some sorters
    sorters = {
        # 'tridesclous': tridesclous_params,
        'kilosort2_5' : kilosort2_5_params,
        # 'kilosort2' : kilosort2_params,
        # 'experimental_sorter1': dict(delete_existing=True),
    }
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



# def postprocessing_sorting(...):



    # compute waveforms for clean
    #  /data1

    # copy back to the NASS






if __name__ == '__main__':

    # need for docker debug
    os.environ["SPIKEINTERFACE_DEV_PATH"] = '/home/analysis_user/Python-related/GitHub/spikeinterface'
    # print(os.getenv('SPIKEINTERFACE_DEV_PATH'))

    spikeglx_folder = base_input_folder / 'Rec_5_11_03_2022_g0'
    time_range = (10000., 11000.)
    #  time_range = None
    run_sorting_pipeline(spikeglx_folder, time_range=time_range)



