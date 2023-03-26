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
from distutils.command.clean import clean
import json
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
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording
from probeinterface import write_prb
import probeinterface as pi
from scipy.stats import linregress
import networkx as nx
import pandas as pd

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from params_NP import *
from recording_list_NP import recording_list
from myfigures import *
from path_handling import (
    get_spikeglx_folder,
    get_synchro_file,
    get_working_folder,
    get_sorting_folder,
)


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
    )  ## global for cambridge probe, move to params (make conditions in params!)
    return rec_preproc


def correct_drift(rec, working_folder):
    motion_file0 = working_folder / "motion.npy"
    motion_file1 = working_folder / "motion_temporal_bins.npy"
    motion_file2 = working_folder / "motion_spatial_bins.npy"

    if motion_file0.exists():
        motion = np.load(motion_file0)
        temporal_bins = np.load(motion_file1)
        spatial_bins = np.load(motion_file2)
    else:
        raise "drift params not computer! run pre sorting checks first!"

    recording_corrected = CorrectMotionRecording(
        rec, motion, temporal_bins, spatial_bins
    )
    return recording_corrected


def slice_rec_time(rec, time_range):
    fs = rec.get_sampling_frequency()

    # Time slicing
    print(f"Time slicing between {time_range[0]} and {time_range[1]}")
    time_range = tuple(float(e) for e in time_range)
    frame_range = (int(t * fs) for t in time_range)
    rec = rec.frame_slice(*frame_range)
    return rec


def slice_rec_depth(rec, depth_range):
    fs = rec.get_sampling_frequency()

    # Channel Slicing
    print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
    yloc = rec.get_channel_locations()[:, 1]
    keep = (yloc >= depth_range[0]) & (yloc <= depth_range[1])
    keep_chan_ids = rec.channel_ids[keep]
    rec = rec.channel_slice(channel_ids=keep_chan_ids)
    return rec


def read_rec(
    spikeglx_folder, stream_id, time_range, depth_range, load_sync_channel=False
):
    rec = si.read_spikeglx(
        spikeglx_folder, stream_id=stream_id, load_sync_channel=load_sync_channel
    )

    if time_range is not None:
        rec = slice_rec_time(rec, time_range)

    elif depth_range is not None:
        rec = slice_rec_depth(rec, depth_range)

    return rec


########### Preprocess & Checks ###########
def get_preprocess_recording(
    spikeglx_folder,
    working_folder,
    time_range=None,
    depth_range=None,
    stream_id="imec0.ap",
    load_sync_channel=False,
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

    time_stamp: str
        time stamp on folder. default = current month

    Returns
    -------
    rec_preprocess: spikeinterface object
        preprocessed recording

    working_folder: Path
        working folder
    """
    rec = read_rec(
        spikeglx_folder, stream_id, time_range, depth_range, load_sync_channel
    )

    # Preprocessing
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
    write_prb(working_folder / "arch.prb", probe_group)  # for lussac

    print(rec_preprocess)

    return rec_preprocess


def run_pre_sorting_checks(rec_preprocess, working_folder):
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

    print("################ Starting presorting checks! ################")

    print(rec_preprocess)
    # Load/compute noise levels
    noise_file = working_folder / "noise_levels.npy"
    if noise_file.exists():
        noise_levels = np.load(noise_file)
    else:
        noise_levels = si.get_noise_levels(rec_preprocess, return_scaled=False)
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

    # compute the motion

    motion_file0 = working_folder / "motion.npy"
    motion_file1 = working_folder / "motion_temporal_bins.npy"
    motion_file2 = working_folder / "motion_spatial_bins.npy"

    if motion_file0.exists():
        motion = np.load(motion_file0)
        temporal_bins = np.load(motion_file1)
        spatial_bins = np.load(motion_file2)
    else:
        motion, temporal_bins, spatial_bins = estimate_motion(
            rec_preprocess, peaks, peak_locations, **motion_estimation_params
        )
        np.save(motion_file0, motion)
        np.save(motion_file1, temporal_bins)
        np.save(motion_file2, spatial_bins)

    # Save plots
    name = working_folder.stem
    figure_folder = working_folder / "figures"
    figure_folder.mkdir(exist_ok=True, parents=True)

    plot_drift(
        peaks,
        rec_preprocess,
        peak_locations,
        name,
        figure_folder,
        motion=motion,
        temporal_bins=temporal_bins,
        spatial_bins=spatial_bins,
    )
    plot_peaks_axis(rec_preprocess, peak_locations, name, figure_folder)
    plot_peaks_activity(peaks, rec_preprocess, peak_locations, name, figure_folder)
    plot_noise(
        rec_preprocess,
        figure_folder,
        with_contact_color=False,
        with_interpolated_map=True,
    )

    fig, ax = plt.subplots()
    ax.plot(temporal_bins, motion)
    fig.savefig(figure_folder / "motions.png")


########### Run sorting ###########
def run_sorting_pipeline(rec_preprocess, working_folder, drift_correction=False):
    """Run sorting with different sorters and params

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    drift_correction: boolean
        to correct for drift or not

    Returns
    -------
    This function will result in sorting and waveform  folders.

    """

    if drift_correction:
        rec_preprocess = correct_drift(rec_preprocess, working_folder)

    # Run sorters and respective params
    for sorter_name, params in sorters.items():
        sorting_folder = working_folder / f"sorting_{sorter_name}"

        if sorting_folder.exists():
            print(f"{sorter_name} already computed ")
            sorting = si.load_extractor(sorting_folder)
        else:
            sorting = si.run_sorter(
                sorter_name,
                rec_preprocess,
                output_folder=working_folder / f"raw_sorting_{sorter_name}",
                delete_output_folder=True,
                verbose=True,
                **params,
            )
            print(sorting)
            sorting = sorting.save(
                format="npz", folder=working_folder / f"sorting_{sorter_name}"
            )

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

        si.compute_quality_metrics(we, load_if_exists=False, metric_names=metrics_list)


########### Post-processing ###########
def run_postprocessing_sorting(
    rec_preprocess, working_folder, sorting_clean_folder, drift_correction=False
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

    drift_correction: boolean
        to correct for drift or not

    Returns
    -------
    This function will result in clean sorting, waveform, report, and phy output folders.

    """

    def merging_unit(potential_pair_merges, sorting):
        graph = nx.Graph()
        for u1, u2 in potential_pair_merges:
            graph.add_edge(u1, u2)
        final_merges = list(nx.components.connected_components(graph))
        final_merges = [list(np.array(i).tolist()) for i in final_merges]
        sorting = si.MergeUnitsSorting(sorting, final_merges)

        return sorting

    print("################ Starting postprocessing! ################")
    if drift_correction:
        rec_preprocess = correct_drift(rec_preprocess, working_folder)

    for sorter_name, _ in sorters.items():
        sorting_clean_folder = Path(
            str(sorting_clean_folder).replace("None", sorter_name)
        )
        print(sorting_clean_folder)

        # Read existing waveforms
        wf_folder = working_folder / f"waveforms_{sorter_name}"
        we = si.WaveformExtractor.load_from_folder(wf_folder)
        print(we)

        # Collect metrics / cleans sorting based on query / creates temporary we after query
        metrics = si.compute_quality_metrics(
            we, metric_names=metrics_list, load_if_exists=True
        )
        our_query = f"snr < {cleaning_params['snr_threshold']} | firing_rate < {cleaning_params['firing_rate']}"
        remove_unit_ids = metrics.query(our_query).index

        clean_sorting = we.sorting.remove_units(remove_unit_ids)
        print(f"cleaned sorting: {clean_sorting}")
        if clean_sorting.unit_ids.size == 0:
            print("no units to work on")
            continue

        wf_temp_query = working_folder / f"waveforms_temp_query_{sorter_name}"
        temp_query_we = si.extract_waveforms(
            rec_preprocess,
            clean_sorting,
            folder=wf_temp_query,
            load_if_exists=True,
            **waveform_params,
            **job_kwargs,
        )
        shutil.rmtree(wf_temp_query)

        # First round of merges (very strict - mostly for single units)
        first_potential_merges = si.get_potential_auto_merge(
            temp_query_we, **first_merge_params
        )  # list of pairs
        if len(first_potential_merges) > 0:
            print("Running first round of merges")
            print(f"Potential merges: {first_potential_merges}")
            clean_sorting = merging_unit(first_potential_merges, clean_sorting)
            print(clean_sorting)
            wf_temp_merges = working_folder / f"waveforms_temp_merge_{sorter_name}"
            we_clean_first = si.extract_waveforms(
                rec_preprocess,
                clean_sorting,
                folder=wf_temp_merges,
                load_if_exists=True,
                **waveform_params,
                **job_kwargs,
            )
            shutil.rmtree(wf_temp_merges)
        else:
            we_clean_first = temp_query_we
            print("No units to merge in the first round")

        # Second round of merges (less strict - mostly for multi units)
        second_potential_merges = si.get_potential_auto_merge(
            we_clean_first, **second_merge_params
        )
        if len(second_potential_merges) > 0:
            print("Running second round of merges")
            print(f"Potential merges: {second_potential_merges}")
            clean_sorting = merging_unit(second_potential_merges, clean_sorting)
            print(clean_sorting)
        else:
            print("No units to merge in the second round")

        # Delete tree before recomputing
        if sorting_clean_folder.exists():
            print("remove exists clean", sorting_clean_folder)
            shutil.rmtree(sorting_clean_folder)

        # Update Wf and create report with clean sorting
        wf_clean_folder = working_folder / f"waveforms_clean_{sorter_name}"
        report_clean_folder = working_folder / f"report_clean_{sorter_name}"

        # Delete any existing folders
        if wf_clean_folder.exists():
            shutil.rmtree(wf_clean_folder)
        if report_clean_folder.exists():
            shutil.rmtree(report_clean_folder)

        clean_sorting = clean_sorting.save(folder=sorting_clean_folder)  # To NAS

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
        si.compute_unit_locations(we_clean, load_if_exists=True, **unit_location_params)

        print("compute correlograms")
        si.compute_correlograms(we_clean, load_if_exists=True, **correlogram_params)

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

        # Add potential labels based on metrics
        csv_metrics_path = report_clean_folder / "quality metrics.csv"
        df = pd.read_csv(csv_metrics_path, index_col=0)
        our_query = f"snr < {classification_params['snr']} & isi_violations_ratio > {classification_params['isi_violations_ratio']}"
        multi_units = df.query(our_query).index
        df["unit_type"] = "single"
        df.loc[multi_units, "unit_type"] = "multi"
        df.to_csv(csv_metrics_path, index=True)


def compare_sorter_cleaned(working_folder, sorting_clean_folder):
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
    sortings = []
    for sorter_name, _ in sorters.items():
        sorting_clean_folder = Path(
            str(sorting_clean_folder).replace("None", sorter_name)
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


def compute_pulse_alignement(spikeglx_folder, working_folder, time_range=None):
    # Get recordings with fixed depth
    rec_nidq = read_rec(
        spikeglx_folder,
        "nidq",
        time_range,
        depth_range=None,
        load_sync_channel=True,
    )
    rec_ap = read_rec(
        spikeglx_folder,
        "imec0.ap",
        time_range,
        depth_range=None,
        load_sync_channel=True,
    )

    print(rec_nidq)
    print(rec_ap)

    # Read NIDQ stream
    pulse_nidq = rec_nidq.get_traces(channel_ids=["nidq#XA1"])
    pulse_nidq = pulse_nidq[:, 0]
    thresh_nidq = (np.max(pulse_nidq) + np.min(pulse_nidq)) / 2

    times_nidq = rec_nidq.get_times()
    pulse_ind_nidq = np.flatnonzero(
        (pulse_nidq[:-1] <= thresh_nidq) & (pulse_nidq[1:] > thresh_nidq)
    )  # identifies the beggining of the pulse
    pulse_time_nidq = times_nidq[pulse_ind_nidq]

    # plt.figure()
    # plt.plot(pulse_time_nidq)
    # plt.show()

    assert np.all(
        np.diff(pulse_time_nidq) > 0.98
    )  # to check if there are no artifacts that could affect the alignment
    assert np.all(
        np.diff(pulse_time_nidq) < 1.02
    )  # to check if there are no artifacts that could affect the alignment

    times_ap = rec_ap.get_times()  # in seconds
    pulse_ap = rec_ap.get_traces(channel_ids=[rec_ap.channel_ids[-1]])
    pulse_ap = pulse_ap[:, 0]

    # Define a threshold
    thresh_ap = 30.0  # there was a weird peak so we couldn't use min max
    pulse_ind_ap = np.flatnonzero(
        (pulse_ap[:-1] <= thresh_ap) & (pulse_ap[1:] > thresh_ap)
    )  # identifies the beggining of the pulse
    pulse_time_ap = times_ap[pulse_ind_ap]

    print("Checking assertions")
    assert np.all(
        np.diff(pulse_time_ap) > 0.98
    )  # to check if there are no artifacts that could affect the alignment
    assert np.all(
        np.diff(pulse_time_ap) < 1.02
    )  # to check if there are no artifacts that could affect the alignment

    print("Computing Linear Regression")
    assert (
        pulse_time_ap.size == pulse_time_nidq.size
    ), f"The two pulse pulse_time_ap:{pulse_time_ap.size} pulse_time_nidq:{pulse_time_nidq.size}"

    a, b, r, tt, stderr = linregress(pulse_time_ap, pulse_time_nidq)
    # times_ap_corrected = times_ap * a + b

    print("regression imec.ap->nidq", "a", a, "b", b, "stderr", stderr)
    assert np.abs(1 - a) < 0.0001, "Very strange slope"
    assert np.abs(b) < 0.5, "intercept (delta) very strange"
    assert stderr < 1e-5, "sterr (tolerance) very strange"

    rec_name = spikeglx_folder.stem
    implant_name = spikeglx_folder.parents[1].stem

    print(rec_name, implant_name)

    synchro_folder_nas = get_synchro_file(
        implant_name, rec_name, time_range, time_stamp
    ).parent

    print(synchro_folder_nas)

    synchro_folder_cache = working_folder / "synchro"
    synchro_folder_cache.mkdir(exist_ok=True)
    synchro_folder_nas.mkdir(exist_ok=True)

    print('Saving')
    np.save(synchro_folder_cache / "pulse_time_nidq.npy", pulse_time_nidq)
    np.save(synchro_folder_cache / "pulse_time_ap.npy", pulse_time_ap)

    # Save info for recording
    synchro_dict = {"a": a, "b": b, "stderr": stderr}

    with open(
        synchro_folder_cache / "synchro_imecap_corr_on_nidq.json", "w"
    ) as outfile:
        json.dump(synchro_dict, outfile, indent=4)

    with open(synchro_folder_nas / "synchro_imecap_corr_on_nidq.json", "w") as outfile:
        json.dump(synchro_dict, outfile, indent=4)


#################################
########### Run Batch ###########
#################################
def run_all(
    pre_check=False,
    sorting=True,
    postproc=True,
    compare_sorters=False,
    compute_alignment=False,
    time_stamp="default",
):
    for (
        implant_name,
        rec_name,
        time_range,
        depth_range,
        drift_correction,
    ) in recording_list:
        spikeglx_folder = get_spikeglx_folder(implant_name, rec_name)
        cache_working_folder = get_working_folder(
            implant_name, rec_name, time_range, depth_range, time_stamp
        )
        NAS_sorting_folder = get_sorting_folder(
            implant_name, rec_name, time_range, depth_range, time_stamp, "None"
        )

        print(f"Data is coming from {spikeglx_folder}")
        print(f"Cache is saved at {cache_working_folder}")

        if any([pre_check, sorting, postproc]):
            # Get relevant recording
            rec_preprocess = get_preprocess_recording(
                spikeglx_folder,
                cache_working_folder,
                time_range,
                depth_range,
                stream_id="imec0.ap",
                load_sync_channel=False
            )

            if pre_check:
                # Run pre-sorting checks
                run_pre_sorting_checks(rec_preprocess, cache_working_folder)

            if sorting:
                # Run sorting pipeline
                run_sorting_pipeline(rec_preprocess, cache_working_folder, drift_correction)

            if postproc:
                # Run postprocessing
                run_postprocessing_sorting(
                    rec_preprocess,
                    cache_working_folder,
                    NAS_sorting_folder,
                    drift_correction,
                )
                print(f"Final sorting is saved at {NAS_sorting_folder}")

        if compare_sorters:
            # Compare sorters
            compare_sorter_cleaned(cache_working_folder, NAS_sorting_folder)

        if compute_alignment:
            # Compute pulse alignement
            compute_pulse_alignement(
                spikeglx_folder,
                cache_working_folder,
                time_range=time_range
            )


if __name__ == "__main__":
    pre_check = False
    sorting = True
    postproc = True
    compare_sorters = False
    compute_alignment = True
    time_stamp = "default"

    run_all(
        pre_check=pre_check,
        sorting=sorting,
        postproc=postproc,
        compare_sorters=compare_sorters,
        compute_alignment=compute_alignment,
        time_stamp=time_stamp,
    )