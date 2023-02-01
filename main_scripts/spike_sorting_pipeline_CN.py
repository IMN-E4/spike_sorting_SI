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


from probeinterface import get_probe
import probeinterface as pi
import networkx as nx
import pandas as pd

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from params_CN import *
from experimental_sorting import run_experimental_sorting
from recording_list_CN import recording_list
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
    rec_preproc = si.common_reference(rec, reference="local", local_radius=(50, 100)) ## global for cambridge probe, move to params (make conditions in params!)
    return rec_preproc


def get_workdir_folder(
    oe_folder,
    time_range,
    depth_range,
    load_sync_channel=False,
    time_stamp="default",
):
    """Create working directory

    Parameters
    ----------
    oe_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    time_stamp: str
        time stamp on folder. default = current month

    Returns
    -------
    rec: spikeinterface object
        recording

    working_folder: Path
        working folder
    """
    rec = si.read_openephys(
        oe_folder
    )
    print(rec)

    # Add probe here
    probe = get_probe('cambridgeneurotech', probe_type)
    probe.wiring_to_device(amp_type)
    rec = rec.set_probe(probe, group_mode='by_shank')
    print(rec.get_property('group'))

    fs = rec.get_sampling_frequency()

    session_name = oe_folder.stem
    bird_name = oe_folder.parents[0].stem

    print(session_name)
    print(bird_name)

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing    ### fix hierarchy of folders, discuss with anindita.
    if time_range is None:  
        working_folder = (
            base_sorting_cache_folder
            / bird_name
            / session_name
            / f"{time_stamp}-{session_name}-full"
        )

    else:
        time_range = tuple(float(e) for e in time_range)

        frame_range = (int(t * fs) for t in time_range)
        rec = rec.frame_slice(*frame_range)

        working_folder = (
            base_sorting_cache_folder
            / bird_name
            / session_name
            / f"{time_stamp}-{session_name}-{int(time_range[0])}to{int(time_range[1])}"
        )
        print(working_folder)

    working_folder.mkdir(exist_ok=True, parents=True)
    print(working_folder)

    if depth_range is not None and not load_sync_channel:
        print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
        yloc = rec.get_channel_locations()[:, 1]
        keep = (yloc >= depth_range[0]) & (yloc <= depth_range[1])
        keep_chan_ids = rec.channel_ids[keep]
        rec = rec.channel_slice(channel_ids=keep_chan_ids)
    else:
        print(f"Using all channels")

    return rec, working_folder


########### Preprocess & Checks ###########
def get_preprocess_recording(
    oe_folder, time_range=None, depth_range=None, time_stamp="default"
):
    """Get preprocessed recording

    Parameters
    ----------
    oe_folder: Path
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
    rec, working_folder = get_workdir_folder(
        oe_folder,
        time_range=time_range,
        depth_range=depth_range,
        time_stamp=time_stamp,
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

    return rec_preprocess, working_folder


def run_pre_sorting_checks(
    oe_folder, time_range=None, depth_range=None, time_stamp="default"
):
    """Apply pre-sorting checks

    Parameters
    ----------
    oe_folder: Path
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

    # Get recording and working dir
    rec_preprocess, working_folder = get_preprocess_recording(
        oe_folder,
        time_range=time_range,
        depth_range=depth_range,
        time_stamp=time_stamp,
    )

    splitted_rec = rec_preprocess.split_by(property='group')
    print(splitted_rec)

    for group, rec_one_shank in splitted_rec.items():
        # Load/compute noise levels
        noise_file = working_folder / f"noise_levels_{group}.npy"
        if noise_file.exists():
            noise_levels = np.load(noise_file)
        else:
            noise_levels = si.get_noise_levels(rec_one_shank, return_scaled=False)
            np.save(noise_file, noise_levels)

        # Load/compute peaks
        peaks_file = working_folder / f"peaks_{group}.npy"
        if peaks_file.exists():
            peaks = np.load(peaks_file)
        else:
            peaks = detect_peaks(
                rec_one_shank,
                noise_levels=noise_levels,
                **peak_detection_params,
                **job_kwargs,
            )
            np.save(peaks_file, peaks)

        # Load/compute peak locations
        location_file = working_folder / f"peak_locations_{group}.npy"
        if location_file.exists():
            peak_locations = np.load(location_file)
        else:
            peak_locations = localize_peaks(rec_one_shank, peaks, **peak_location_params, **job_kwargs)
            np.save(location_file, peak_locations)
        
        # compute the motion
        
        motion_file0 = working_folder / f"motion_{group}.npy"
        motion_file1 = working_folder / f"motion_temporal_bins_{group}.npy"
        motion_file2 = working_folder / f"motion_spatial_bins_{group}.npy"

        if motion_file0.exists():
            motion = np.load(motion_file0)
            temporal_bins = np.load(motion_file1)
            spatial_bins = np.load(motion_file2)
        else:
            motion, temporal_bins, spatial_bins = estimate_motion(rec_one_shank, peaks, peak_locations, **motion_estimation_params)
            np.save(motion_file0, motion)
            np.save(motion_file1, temporal_bins)
            np.save(motion_file2, spatial_bins)


        # Save plots
        name = Path(oe_folder).stem
        figure_folder = working_folder / f"figures_{group}"
        figure_folder.mkdir(exist_ok=True, parents=True)

        plot_drift(peaks, rec_one_shank, peak_locations, name, figure_folder,
                motion=motion, temporal_bins=temporal_bins, spatial_bins=spatial_bins)
        plot_peaks_axis(rec_one_shank, peak_locations, name, figure_folder)
        plot_peaks_activity(peaks, rec_one_shank, peak_locations, name, figure_folder)
        plot_noise(
            rec_one_shank,
            figure_folder,
            with_contact_color=False,
            with_interpolated_map=True,
        )

        fig, ax = plt.subplots()
        ax.plot(temporal_bins, motion)
        fig.savefig(figure_folder / 'motions.png')



########### Run sorting ###########
def run_sorting_pipeline(
    oe_folder, time_range=None, depth_range=None, time_stamp="default"
):
    """Run sorting with different sorters and params

    Parameters
    ----------
    oe_folder: Path
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
        oe_folder,
        time_range=time_range,
        depth_range=depth_range,
        time_stamp=time_stamp,
    )

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
    oe_folder, time_range=None, depth_range=None, time_stamp="default"
):
    """Run post-processing on different sorters and params
    The idea is to have bad units removed according to metrics, and run auto-merging of units.

    Parameters
    ----------
    oe_folder: Path
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
        oe_folder,
        time_range=time_range,
        depth_range=depth_range,
        time_stamp=time_stamp,
    )
    name = working_folder.stem
    bird_name = oe_folder.parents[1].stem

    def merging_unit(potential_pair_merges, sorting):
        graph = nx.Graph()
        for u1, u2 in potential_pair_merges:
            graph.add_edge(u1,u2)

        final_merges = list(nx.components.connected_components(graph))
        final_merges =[list(i) for i in final_merges] # just to convert to list of lists

        for some_units in final_merges:
            print(some_units)
            sorting = si.MergeUnitsSorting(sorting, some_units)
        
        return sorting

    for sorter_name, params in sorters.items():
        # Read existing waveforms
        wf_folder = working_folder / f"waveforms_{sorter_name}"
        we = si.WaveformExtractor.load_from_folder(wf_folder)
        print(we)

             
        # sorting_no_dup = si.remove_redundant_units(we, remove_strategy="minimum_shift") # DELETE?
        
        # Collect metrics and clean sorting
        metrics = si.compute_quality_metrics(we, load_if_exists=True)
        our_query = f"snr < {cleaning_params['snr_threshold']} | firing_rate < {cleaning_params['firing_rate']}"
        remove_unit_ids = metrics.query(our_query).index

        clean_sorting = we.sorting.remove_units(remove_unit_ids)

        if clean_sorting.unit_ids.size == 0:
            print("no units to work on")
            continue
        
        # First round of merges (very strict - mostly for single units)
        first_potential_pair_merges = si.get_potential_auto_merge(we, **first_merge_params) # list ofpairs
        clean_sorting = merging_unit(first_potential_pair_merges, clean_sorting)
        wf_temp = working_folder / f"waveforms_temp_{sorter_name}"
        we_clean_first = si.extract_waveforms(
            rec_preprocess,
            clean_sorting,
            folder=wf_temp,
            load_if_exists=True,
            **waveform_params,
            **job_kwargs,
        )
        shutil.rmtree(wf_temp)

        # Second round of merges (less strict - mostly for multi units)
        potential_pair_merges = si.get_potential_auto_merge(we_clean_first, **second_merge_params)
        clean_sorting = merging_unit(potential_pair_merges, clean_sorting)        

        sorting_clean_folder = (
            base_input_folder / bird_name / "Sortings_clean" / name / sorter_name
        )

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

        # Add potential labels based on metrics
        csv_metrics_path = report_clean_folder/'quality metrics.csv'
        df = pd.read_csv(csv_metrics_path, index_col=0)
        our_query = f"snr < {classification_params['snr']} & isi_violations_ratio > {classification_params['isi_violations_ratio']}"
        multi_units = df.query(our_query).index
        df['unit_type'] = 'single'
        df.loc[multi_units, 'unit_type'] = 'multi'
        df.to_csv(csv_metrics_path, index=True)




def compare_sorter_cleaned(oe_folder, time_range=None, time_stamp="default"):
    """Comparison between sorters

    Parameters
    ----------
    oe_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    Returns
    -------
    Agreement matrix.

    """
    rec_preprocess, working_folder = get_preprocess_recording(
        oe_folder, time_range=time_range
    )
    name = working_folder.stem
    bird_name = oe_folder.parents[1].stem

    sortings = []
    for sorter_name, params in sorters.items():
        sorting_clean_folder = (
            base_input_folder / bird_name / "Sortings_clean" / name / sorter_name
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



def test_path(oe_folder, time_range=None, depth_range=None, time_stamp="default"):
    rec_preprocess, working_folder = get_preprocess_recording(
        oe_folder,
        time_range=time_range,
        depth_range=depth_range,
        time_stamp=time_stamp,
    )

    print(working_folder)
    working_folder.mkdir()

#################################
########### Run Batch ###########
#################################


def run_all(
    pre_check=True,
    sorting=True,
    postproc=True,
    compare_sorters=True,
    time_stamp="default",
):
    for bird_name, session_name, time_range, depth_range in recording_list:
        oe_folder = base_input_folder / bird_name / session_name

        print(oe_folder)

        if pre_check:
            # Run pre-sorting checks
            run_pre_sorting_checks(
                oe_folder,
                time_range=time_range,
                depth_range=depth_range,
                time_stamp=time_stamp,
            )

        if sorting:
            # Run sorting pipeline
            run_sorting_pipeline(
                oe_folder,
                time_range=time_range,
                depth_range=depth_range,
                time_stamp=time_stamp,
            )

        if postproc:
            # Run postprocessing
            run_postprocessing_sorting(
                oe_folder,
                time_range=time_range,
                depth_range=depth_range,
                time_stamp=time_stamp,
            )

        if compare_sorters:
            # Compare sorters
            compare_sorter_cleaned(
                oe_folder, time_range=time_range, time_stamp=time_stamp
            )


if __name__ == "__main__":
    pre_check = False
    sorting = True
    postproc = True
    compare_sorters = False
    time_stamp = "default"

    run_all(
        pre_check=pre_check,
        sorting=sorting,
        postproc=postproc,
        compare_sorters=compare_sorters,
        time_stamp=time_stamp,
    )
