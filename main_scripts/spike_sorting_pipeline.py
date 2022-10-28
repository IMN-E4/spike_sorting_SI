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
__status__ = "Production"  ### Production = still being developed. Else: Concluded/Finished.


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
from datetime import datetime
from distutils.command.clean import clean
import shutil
import os
from pathlib import Path

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import (
    localize_peaks,
)
from probeinterface import write_prb
import probeinterface as pi

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from params import *
from experimental_sorting import run_experimental_sorting
from recording_list import recording_list
from myfigures import *


########### Preparatory Functions ###########
def apply_preprocess(rec):
    """Apply lazy preprocessing chain.

    Parameters
    ----------
    rec: spikeinterface object
        recording to apply preprocessing on.

    Returns
    -------
    rec_preproc: spikeinterface object
        preprocessed rec
    """
    # Bandpass filter
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)

    # Common referencing
    rec_preproc = si.common_reference(
        rec, reference="local", local_radius=(50, 100)
    )
    return rec_preproc


def get_workdir_folder(spikeglx_folder, time_range, depth_range, stream_id="imec0.ap", load_sync_channel=False):
    """Create working directory

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    Returns
    -------
    rec: spikeinterface object
        recording

    working_folder: Path
        working folder
    """
    rec = si.read_spikeglx(spikeglx_folder, stream_id=stream_id, load_sync_channel=load_sync_channel)
    # print(rec)

    fs = rec.get_sampling_frequency()

    name = spikeglx_folder.stem
    implant_name = spikeglx_folder.parents[1].stem
    time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{name}-full"
        )

    else:
        time_range = tuple(float(e) for e in time_range)

        frame_range = (int(t * fs) for t in time_range)
        rec = rec.frame_slice(*frame_range)

        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{name}-{int(time_range[0])}to{int(time_range[1])}"
        )

    working_folder.mkdir(exist_ok=True, parents=True)
    print(working_folder)

    if depth_range is not None and not load_sync_channel:
        print(
            f"Depth slicing between {depth_range[0]} and {depth_range[1]}"
        )
        yloc = rec.get_channel_locations()[:, 1]
        keep = (yloc >= depth_range[0]) & (yloc <=depth_range[1])
        keep_chan_ids = rec.channel_ids[keep]
        rec = rec.channel_slice(channel_ids=keep_chan_ids)
    else:
        print(f"Using all channels")

    return rec, working_folder


########### Preprocess & Checks ###########
def get_preprocess_recording(
    spikeglx_folder, time_range=None, depth_range=None
):
    """Get preprocessed recording

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    Returns
    -------
    rec_preprocess: spikeinterface object
        preprocessed recording

    working_folder: Path
        working folder
    """
    rec, working_folder = get_workdir_folder(
        spikeglx_folder,
        time_range=time_range,
        depth_range=depth_range,
    )

    # Preprocessing
    preprocess_folder = working_folder / "preprocess_recording"
    if preprocess_folder.exists():
        print("Already preprocessed")
        rec_preprocess = si.load_extractor(preprocess_folder)

    elif (working_folder / "preprocess.json").exists():
        rec_preprocess = si.load_extractor(
            working_folder / "preprocess.json"
        )
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
    write_prb(working_folder / "arch.prb", probe_group)  # for lussac

    return rec_preprocess, working_folder


def run_pre_sorting_checks(
    spikeglx_folder, time_range=None, depth_range=None
):
    """Apply pre-sorting checks

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    Returns
    -------
    This function will result in plots from plot_drift,  plot_peaks_axis,
    plot_peaks_activity, and plot_noise.

    """

    print(
        "################ Starting presorting checks! ################"
    )

    # Get recording and working dir
    rec_preprocess, working_folder = get_preprocess_recording(
        spikeglx_folder,
        time_range=time_range,
        depth_range=depth_range,
    )

    # Load/compute noise levels
    noise_file = working_folder / "noise_levels.npy"
    if noise_file.exists():
        noise_levels = np.load(noise_file)
    else:
        noise_levels = si.get_noise_levels(
            rec_preprocess, return_scaled=False
        )
        np.save(noise_file, noise_levels)

    # Load/compute peaks
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

    # Load/compute peak locations
    location_file = working_folder / "peak_locations.npy"
    if location_file.exists():
        peak_locations = np.load(location_file)
    else:
        peak_locations = localize_peaks(
            rec_preprocess, peaks, **peak_location_params, **job_kwargs
        )
        np.save(location_file, peak_locations)

    # Save plots
    name = Path(spikeglx_folder).stem
    figure_folder = working_folder / "figures"
    figure_folder.mkdir(exist_ok=True, parents=True)

    plot_drift(
        peaks, rec_preprocess, peak_locations, name, figure_folder
    )
    plot_peaks_axis(rec_preprocess, peak_locations, name, figure_folder)
    plot_peaks_activity(
        peaks, rec_preprocess, peak_locations, name, figure_folder
    )
    plot_noise(
        rec_preprocess,
        figure_folder,
        with_contact_color=False,
        with_interpolated_map=True,
    )


########### Run sorting ###########
def run_sorting_pipeline(
    spikeglx_folder, time_range=None, depth_range=None
):
    """Run sorting with different sorters and params

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    Returns
    -------
    This function will result in sorting and waveform  folders.

    """

    print("################ Runninng sorters! ################")
    rec_preprocess, working_folder = get_preprocess_recording(
        spikeglx_folder,
        time_range=time_range,
        depth_range=depth_range,
    )

    # Run sorters and respective params
    for sorter_name, params in sorters.items():
        sorting_folder = working_folder / f"sorting_{sorter_name}"

        if sorting_folder.exists():
            print(f'{sorter_name} already computed ')
            sorting = si.load_extractor(sorting_folder)
        else:
            sorting = si.run_sorter(sorter_name, rec_preprocess,
                                output_folder=working_folder / f'raw_sorting_{sorter_name}',
                                delete_output_folder=True,
                                verbose=True,
                                **params
                                )
            print(sorting)
            sorting = sorting.save(format='npz', folder=working_folder / f'sorting_{sorter_name}')

    # Extract waveforms and compute some metrics
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

        si.compute_spike_amplitudes(
            we, load_if_exists=True, **amplitude_params, **job_kwargs
        )

        si.compute_quality_metrics(
            we, load_if_exists=False, metric_names=metrics_list
        )


########### Post-processing ###########
def run_postprocessing_sorting(
    spikeglx_folder, time_range=None, depth_range=None
):
    """Run post-processing on different sorters and params
    The idea is to have bad units removed according to metrics, and run auto-merging of units.

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    Returns
    -------
    This function will result in clean sorting, waveform, report, and phy output folders.

    """

    print("################ Starting postprocessing! ################")

    rec_preprocess, working_folder = get_preprocess_recording(
        spikeglx_folder,
        time_range=time_range,
        depth_range=depth_range,
    )
    name = working_folder.stem
    implant_name = spikeglx_folder.parents[1].stem

    for sorter_name, params in sorters.items():
        # Read existing waveforms
        wf_folder = working_folder / f"waveforms_{sorter_name}"
        we = si.WaveformExtractor.load_from_folder(wf_folder)

        sorting_no_dup = si.remove_redundant_units(
            we, remove_strategy="minimum_shift"
        )

        # Collect metrics and clean sorting
        metrics = si.compute_quality_metrics(we, load_if_exists=True)
        our_query = f"snr < {cleaning_params['snr_threshold']} | firing_rate < {cleaning_params['firing_rate']}"
        remove_unit_ids = metrics.query(our_query).index

        clean_sorting = sorting_no_dup.remove_units(remove_unit_ids)

        if clean_sorting.unit_ids.size == 0:
            print("no units to work on")
            continue

        sorting_clean_folder = (
            base_input_folder
            / implant_name
            / "Sortings_clean"
            / name
            / sorter_name
        )

        # Delete tree before recomputing
        if sorting_clean_folder.exists():
            print("remove exists clean", sorting_clean_folder)
            shutil.rmtree(sorting_clean_folder)

        # Update Wf and create report with clean sorting
        wf_clean_folder = (
            working_folder / f"waveforms_clean_{sorter_name}"
        )
        report_clean_folder = (
            working_folder / f"report_clean_{sorter_name}"
        )

        # Delete any existing folders
        if wf_clean_folder.exists():
            shutil.rmtree(wf_clean_folder)
        if report_clean_folder.exists():
            shutil.rmtree(report_clean_folder)

        clean_sorting = clean_sorting.save(
            folder=sorting_clean_folder
        )  # To NAS

        # Compute Wf and report for cleaned sorting
        we_clean = si.extract_waveforms(
            rec_preprocess,
            clean_sorting,
            folder=wf_clean_folder,
            load_if_exists=True,
            **waveform_params,
            **job_kwargs,
        )
        print(we_clean)

        print("computing spike amplitudes")
        si.compute_spike_amplitudes(
            we_clean,
            load_if_exists=True,
            **amplitude_params,
            **job_kwargs,
        )

        print("computing quality metrics")
        si.compute_quality_metrics(
            we_clean, load_if_exists=True, metric_names=metrics_list
        )

        print("computing locations")
        si.compute_unit_locations(
            we_clean,
            method="monopolar_triangulation",
            radius_um=150,
            max_distance_um=1000,
            optimizer="minimize_with_log_penality",
            load_if_exists=True,
        )

        print("compute correlograms")
        si.compute_correlograms(
            we_clean, window_ms=50.0, bin_ms=1.0, load_if_exists=True
        )

        if report_clean_folder.exists():
            print("report already there for ", report_clean_folder)
        else:
            print("exporting report")
            si.export_report(
                we_clean,
                report_clean_folder,
                remove_if_exists=False,
                **job_kwargs,
            )

        # # export to phy

        # phy_folder = working_folder / f"phy_clean_{sorter_name}"
        # wf_folder = working_folder / f"waveforms_{sorter_name}"
        # we = si.WaveformExtractor.load_from_folder(wf_folder)
        # si.export_to_phy(we, phy_folder, remove_if_exists=False, **job_kwargs)


def compare_sorter_cleaned(spikeglx_folder, time_range=None):
    """Comparison between sorters

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    Returns
    -------
    Agreement matrix.

    """
    rec_preprocess, working_folder = get_preprocess_recording(
        spikeglx_folder, time_range=time_range
    )
    name = working_folder.stem
    implant_name = spikeglx_folder.parents[1].stem

    sortings = []
    for sorter_name, params in sorters.items():
        sorting_clean_folder = (
            base_input_folder
            / implant_name
            / "Sortings_clean"
            / name
            / sorter_name
        )
        sorting = si.load_extractor(sorting_clean_folder)
        sortings.append(sorting)

    sorter_names = list(sorters.keys())
    n = len(sorters)
    for i in range(n - 1):
        for j in range(i + 1, n):

            comp = si.compare_two_sorters(
                sortings[i],
                sortings[j],
                sorting1_name=sorter_names[i],
                sorting2_name=sorter_names[j],
                delta_time=0.4,
                match_score=0.5,
                chance_score=0.1,
                n_jobs=1,
            )

            fig, ax = plt.subplots()
            si.plot_agreement_matrix(comp, ax=ax)
            comparison_figure_file = (
                working_folder
                / f"comparison_clean_{sorter_names[i]}_{sorter_names[j]}.pdf"
            )
            print(comparison_figure_file)
            plt.show()
            # fig.savefig(comparison_figure_file)


#################################
########### Run Batch ###########
#################################


def run_all(
    pre_check=True, sorting=True, postproc=True, compare_sorters=True
):
    for implant_name, name, time_range, depth_range in recording_list:
        spikeglx_folder = (
            base_input_folder / implant_name / "Recordings" / name
        )

        print(spikeglx_folder)

        if pre_check:
            # Run pre-sorting checks
            run_pre_sorting_checks(
                spikeglx_folder,
                time_range=time_range,
                depth_range=depth_range,
            )

        if sorting:
            # Run sorting pipeline
            run_sorting_pipeline(
                spikeglx_folder,
                time_range=time_range,
                depth_range=depth_range,
            )

        if postproc:
            # Run postprocessing
            run_postprocessing_sorting(
                spikeglx_folder,
                time_range=time_range,
                depth_range=depth_range,
            )

        if compare_sorters:
            # Compare sorters
            compare_sorter_cleaned(
                spikeglx_folder, time_range=time_range
            )


if __name__ == "__main__":
    pre_check = True
    sorting = True
    postproc = True
    compare_sorters = False

    run_all(
        pre_check=pre_check,
        sorting=sorting,
        postproc=postproc,
        compare_sorters=compare_sorters,
    )
