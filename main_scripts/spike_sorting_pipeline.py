#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the 'new' pipeline for neuropixel data.
June 2022

This run several sorters on NP data.

preprocessed recording and waveforms are saved in local cache, final clean sorting
is saved in NAS.
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2022/06/1"  ### Date it was created
__status__ = (
    "Production"  ### Production = still being developed. Else: Concluded/Finished.
)


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
from datetime import datetime
import shutil
import os
from pathlib import Path

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from probeinterface import write_prb
import probeinterface as pi

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from params import *
from experimental_sorting import run_experimental_sorting
from recording_list import recording_list
from myfigures import *


########### Preparatory Functions
def apply_preprocess(rec):
    """
    Apply the lazy preprocessing chain.
    """
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    rec = si.common_reference(rec, reference="local", local_radius=(50, 100))
    return rec


def fix_time_range(spikeglx_folder, time_range):
    rec = si.read_spikeglx(spikeglx_folder, stream_id="imec0.ap")
    # print(rec)

    fs = rec.get_sampling_frequency()

    if time_range is None:
        duration = rec.get_num_frames() / rec.get_sampling_frequency()
        time_range = (0.0, duration)

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
    implant_name = spikeglx_folder.parents[1].stem
    time_stamp = datetime.now().strftime("%Y-%m")
    working_folder = (
        base_sorting_cache_folder
        / implant_name
        / "sorting_cache"
        / f"{time_stamp}-{name}-{time_range[0]}to{time_range[1]}"
    )
    working_folder.mkdir(exist_ok=True, parents=True)
    print(working_folder)

    return working_folder


########### Preprocess & Checks
def get_preprocess_recording(spikeglx_folder, time_range=None):
    """
    Function to get preprocessed recording.
    """
    print(f"first time range is {time_range}")
    rec, time_range = fix_time_range(spikeglx_folder, time_range=time_range)

    print(f"second time range is {time_range}")
    working_folder = get_workdir_folder(spikeglx_folder, time_range=time_range)

    # preprocessing
    preprocess_folder = working_folder / "preprocess_recording"
    if preprocess_folder.exists():
        print("Already preprocessed")
        rec_preprocess = si.load_extractor(preprocess_folder)
    elif (working_folder / "preprocess.json").exists():
        rec_preprocess = si.load_extractor(working_folder / "preprocess.json")
        rec_preprocess = rec_preprocess.save(
            format="binary", folder=preprocess_folder, **job_kwargs
        )
    else:
        print("Run/save preprocessing")
        rec_preprocess = apply_preprocess(rec)
        rec_preprocess.dump_to_json(working_folder / "preprocess.json")
        rec_preprocess = rec_preprocess.save(
            format="binary", folder=preprocess_folder, **job_kwargs
        )

    probe_group = pi.ProbeGroup()
    probe_group.add_probe(rec_preprocess.get_probe())
    write_prb(working_folder / "arch.prb", probe_group)

    return rec_preprocess, working_folder


def run_pre_sorting_checks(spikeglx_folder, time_range=None):
    """
    Function to apply pre-sorting checks.

    This function will result in plots from plot_drift,  plot_peaks_axis, plot_peaks_activity, and plot_noise.
    """

    print('################ Starting presorting checks! ################')

    rec_preprocess, working_folder = get_preprocess_recording(
        spikeglx_folder, time_range=time_range
    )
    # print(rec_preprocess)

    noise_file = working_folder / "noise_levels.npy"
    if noise_file.exists():
        noise_levels = np.load(noise_file)
    else:
        noise_levels = si.get_noise_levels(rec_preprocess, return_scaled=False)
        np.save(noise_file, noise_levels)

    peaks_file = working_folder / "peaks.npy"
    if peaks_file.exists():
        peaks = np.load(peaks_file)
    else:
        peaks = detect_peaks(
            rec_preprocess,
            noise_levels=noise_levels,
            **peak_detection_params,
            **job_kwargs,
        )
        np.save(peaks_file, peaks)
    # print(peaks.shape)

    location_file = working_folder / "peak_locations.npy"
    if location_file.exists():
        peak_locations = np.load(location_file)
    else:
        peak_locations = localize_peaks(
            rec_preprocess, peaks, **peak_location_params, **job_kwargs
        )
        np.save(location_file, peak_locations)
    # print(peak_locations.shape)

    name = Path(spikeglx_folder).stem

    figure_folder = working_folder / "figures"
    figure_folder.mkdir(exist_ok=True, parents=True)

    plot_drift(peaks, rec_preprocess, peak_locations, name, figure_folder)
    plot_peaks_axis(rec_preprocess, peak_locations, name, figure_folder)
    plot_peaks_activity(peaks, rec_preprocess, peak_locations, name, figure_folder)
    plot_noise(
        rec_preprocess,
        figure_folder,
        with_contact_color=False,
        with_interpolated_map=True,
    )


########### Run sorting
def run_sorting_pipeline(spikeglx_folder, time_range=None):
    """
    Function to run sorting with different sorters and params.

    This function will result in sorting, waveform, report, and phy output folders.
    """

    print('################ Runninng sorters! ################')
    rec_preprocess, working_folder = get_preprocess_recording(
        spikeglx_folder, time_range=time_range
    )

    # run some sorters
    for sorter_name, params in sorters.items():
        sorting_folder = working_folder / f"sorting_{sorter_name}"
        if sorting_folder.exists():
            print(f"{sorter_name} already computed ")
            sorting = si.load_extractor(sorting_folder)
        else:
            if sorter_name != "experimental_sorter1":
                print(f"Computing {sorter_name}")
                sorting = si.run_sorter(
                    sorter_name,
                    rec_preprocess,
                    output_folder=working_folder / f"raw_sorting_{sorter_name}",
                    delete_output_folder=True,
                    verbose=True,
                    **params,
                )
            else:
                print(f"Computing {sorter_name}")
                sorting = run_experimental_sorting(
                    rec_preprocess,
                    output_folder=working_folder / f"raw_sorting_{sorter_name}",
                    job_kwargs=job_kwargs,
                    **params,
                )
            # print(sorting)
            sorting = sorting.save(
                format="npz", folder=working_folder / f"sorting_{sorter_name}"
            )

    # extract waveforms and compute some metrics
    for sorter_name, params in sorters.items():
        sorting_folder = working_folder / f"sorting_{sorter_name}"
        sorting = si.load_extractor(sorting_folder)

        wf_folder = working_folder / f"waveforms_{sorter_name}"
        we = si.extract_waveforms(
            rec_preprocess,
            sorting,
            folder=wf_folder,
            load_if_exists=True,
            **waveform_params,
            **job_kwargs,
        )
        # print(we)

        si.compute_spike_amplitudes(
            we, load_if_exists=True, **amplitude_params, **job_kwargs
        )

        si.compute_quality_metrics(we, load_if_exists=False, metric_names=metrics_list)


########### Post-processing
def run_postprocessing_sorting(spikeglx_folder, time_range=None):
    """
    Function to run post-processing on different sorters and params.

    The idea is to have bad units removed according to metrics, and run auto-merging of units.

    This function will result in clean sorting, waveform, report, and phy output folders.
    """
    
    print('################ Starting postprocessing! ################')

    rec_preprocess, working_folder = get_preprocess_recording(
        spikeglx_folder, time_range=time_range
    )
    name = working_folder.stem
    implant_name = spikeglx_folder.parents[1].stem

    for sorter_name, params in sorters.items():
        wf_folder = working_folder / f"waveforms_{sorter_name}"
        we = si.WaveformExtractor.load_from_folder(wf_folder)
        sorting_no_dup = si.remove_redundant_units(we, remove_strategy="minimum_shift")
        # print(sorting_no_dup.unit_ids)

        metrics = si.compute_quality_metrics(we, load_if_exists=True)
        our_query = f"snr < {cleaning_params['snr_threshold']} | firing_rate < {cleaning_params['firing_rate']}"
        remove_unit_ids = metrics.query(our_query).index
        # print('remove_unit_ids', remove_unit_ids)

        clean_sorting = sorting_no_dup.remove_units(remove_unit_ids)
        # print(clean_sorting.unit_ids)

        sorting_clean_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / sorter_name
        )
        if sorting_clean_folder.exists():
            print("remove exists clean", sorting_clean_folder)
            shutil.rmtree(sorting_clean_folder)
        wf_clean_folder = working_folder / f"waveforms_clean_{sorter_name}"
        if wf_clean_folder.exists():
            shutil.rmtree(wf_clean_folder)

        clean_sorting = clean_sorting.save(folder=sorting_clean_folder)

        we_clean = si.extract_waveforms(
            rec_preprocess,
            clean_sorting,
            folder=wf_clean_folder,
            load_if_exists=False,
            **waveform_params,
            **job_kwargs,
        )
        print(we_clean)

        si.compute_spike_amplitudes(
            we_clean, load_if_exists=True, **amplitude_params, **job_kwargs
        )

        si.compute_quality_metrics(
            we_clean, load_if_exists=True, metric_names=metrics_list
        )

        report_clean_folder = working_folder / f"report_clean_{sorter_name}"
        if report_clean_folder.exists():
            print("report already there for ", report_clean_folder)
            continue
        else:
            si.export_report(
            we_clean, report_clean_folder, remove_if_exists=False, **job_kwargs
            )

        # # export to phy
        # phy_folder = working_folder / f"phy_clean_{sorter_name}"
        # wf_folder = working_folder / f"waveforms_{sorter_name}"
        # we = si.WaveformExtractor.load_from_folder(wf_folder)
        # si.export_to_phy(we, phy_folder, remove_if_exists=False, **job_kwargs)


### re build the preprocess folder (not sure what this is for)


def rebuild_preprocess():
    spikeglx_folder = base_input_folder / "Imp_16_08_2022/Recordings/Rec_18_08_2022_g0"
    time_range = None
    get_preprocess_recording(spikeglx_folder, time_range=time_range)

########### Run Batch
def run_all():
    for implant_name, name, time_range in recording_list:
        spikeglx_folder = base_input_folder / implant_name / "Recordings" / name

        print(spikeglx_folder)

        # run_pre_sorting_checks(spikeglx_folder, time_range=time_range)

        run_sorting_pipeline(spikeglx_folder, time_range=time_range)

        run_postprocessing_sorting(spikeglx_folder, time_range=time_range)


if __name__ == "__main__":
    run_all()
