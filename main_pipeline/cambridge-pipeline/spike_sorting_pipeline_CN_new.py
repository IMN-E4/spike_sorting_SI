#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sorting pipeline for Cambridge Neurotech data.

It allows user to run several sorters with several params.

Preprocessed recording and waveforms are saved in local cache, final clean sorting
is saved in NAS.
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/09/1"
__status__ = "Production"


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports
import shutil
from pathlib import Path

# Third party imports
import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import (
    localize_peaks,
)
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
import networkx as nx
import pandas as pd

# Internal imports
from params_CN import *
from recording_list_CN import recording_list
from myfigures import *
from path_handling import (
    concatenate_openephys_folder_path,
    concatenate_working_folder_path,
    concatenate_clean_sorting_path,
)
from utils import *


########### Preprocess & Checks ###########
def get_preprocess_recording(
    openephys_folder, working_folder, time_range=None, depth_range=None
):
    """Get preprocessed recording

    Parameters
    ----------
    openephys_folder: Path
        path to spikeglx folder

    working_folder: Path
        working folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    Returns
    -------
    with_probe_rec: spikeinterface object
        preprocessed recording with probe attached

    """
    assert isinstance(
        openephys_folder, Path
    ), f"openephys_folder must be type Path not {type(openephys_folder)}"
    assert isinstance(
        working_folder, Path
    ), f"working_folder must be Path not {type(working_folder)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), "time_range must be type tuple, list or None"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), "depth_range must be type tuple, list or None"



    # Preprocessing
    preprocess_folder = working_folder / "preprocess_recording"
    if preprocess_folder.exists():
        print("Already preprocessed")
        rec_preprocess = si.load_extractor(preprocess_folder)
        # This should be  unecessary!!!
        # rec_preprocess.annotate(is_filtered=True)
        print('ici')
        print(rec_preprocess)
        print(rec_preprocess.is_filtered())


    elif (working_folder / "preprocess.json").exists():
        rec_preprocess = si.load_extractor(working_folder / "preprocess.json")
        rec_preprocess = rec_preprocess.save(
            format="binary", folder=preprocess_folder, **job_kwargs
        )
    else:
        print("Run/save preprocessing")
        rec = read_rec(
            openephys_folder,
            time_range,
            depth_range,
        )
        # attach probe
        with_probe_rec = add_probe_to_rec(rec)
        print(with_probe_rec)
                
        rec_preprocess = apply_preprocess(with_probe_rec)
        rec_preprocess.dump_to_json(working_folder / "preprocess.json")
        rec_preprocess = rec_preprocess.save(
            format="binary", folder=preprocess_folder, **job_kwargs
        )
    

    print(rec_preprocess)

    # probe_group = pi.ProbeGroup()
    # probe_group.add_probe(rec_preprocess.get_probe())
    # write_prb(working_folder / "arch.prb", probe_group)  # for lussac

    return rec_preprocess


def run_pre_sorting_checks(rec_preprocess, working_folder):
    """Apply pre-sorting checks

    Parameters
    ----------
    rec_preprocess: spikeinterface BinaryFolderRecording or si.ChannelSliceRecording
        recording object

    working_folder: path
        path to working folder

    Returns
    -------
    This function will result in plots from plot_drift,  plot_peaks_axis,
    plot_peaks_activity, and plot_noise.

    """
    assert isinstance(
        rec_preprocess, (si.BinaryFolderRecording, si.ChannelSliceRecording)
    ), f"rec_preprocess must be type spikeinterface BinaryFolderRecording or si.ChannelSliceRecording not {type(rec_preprocess)}"
    assert isinstance(
        working_folder, Path
    ), f"working_folder must be Path not {type(working_folder)}"

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
    rec_preprocess: si.BinaryFolderRecording or si.ChannelSliceRecording
        recording object

    working_folder: path
        path to working folder

    drift_correction: boolean
        to correct for drift or not

    Returns
    -------
    This function will result in sorting and waveform folders.

    """
    # assert isinstance(
    #     rec_preprocess, (si.BinaryFolderRecording, si.ChannelSliceRecording)
    # ), f"rec_preprocess must be type spikeinterface BinaryFolderRecording or si.ChannelSliceRecording not {type(rec_preprocess)}"
    assert isinstance(
        working_folder, Path
    ), f"working_folder must be Path not {type(working_folder)}"
    assert isinstance(
        drift_correction, bool
    ), f"drift_correction must be boolean not {type(drift_correction)}"

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
                format="numpy_folder", folder=working_folder / f"sorting_{sorter_name}"
            )

    # Extract waveforms and compute some metrics
    for sorter_name, params in sorters.items():
        sorting_folder = working_folder / f"sorting_{sorter_name}"
        sorting = si.load_extractor(sorting_folder)

        print('la')
        print(rec_preprocess)
        print(rec_preprocess.is_filtered())


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

    rec_preprocess: spikeinterface obj
        recording object

    working_folder: path
        path to working folder

    sorting_clean_folder: Path
        path to sorting clean folder

    drift_correction: boolean
        to correct for drift or not

    Returns
    -------
    This function will result in a clean sorting, waveform, and report folders.

    """
    assert isinstance(
        rec_preprocess, si.BinaryFolderRecording
    ), f"rec_preprocess must be type spikeinterface BinaryFolderRecording not {type(rec_preprocess)}"
    assert isinstance(
        working_folder, Path
    ), f"working_folder must be Path not {type(working_folder)}"
    assert isinstance(
        sorting_clean_folder, Path
    ), f"sorting_clean_folder must be Path not {type(sorting_clean_folder)}"
    assert isinstance(
        drift_correction, bool
    ), f"drift_correction must be boolean not {type(drift_correction)}"

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

        si.compute_template_metrics(we_clean)

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

        # Add potential labels based on metrics
        csv_metrics_path = report_clean_folder / "quality metrics.csv"
        df = pd.read_csv(csv_metrics_path, index_col=0)
        our_query = f"snr < {classification_params['snr']} & isi_violations_ratio > {classification_params['isi_violations_ratio']}"
        multi_units = df.query(our_query).index
        df["unit_type"] = "single"
        df.loc[multi_units, "unit_type"] = "multi"
        df.to_csv(csv_metrics_path, index=True)


########### Sorting comparison ###########
def compare_sorter_cleaned(working_folder, sorting_clean_folder):
    """Comparison between sorters

    Parameters
    ----------
    working_folder: path
        path to working folder

    sorting_clean_folder: Path
        path to sorting clean folder

    Returns
    -------
    Agreement matrix.

    """
    assert isinstance(
        working_folder, Path
    ), f"working_folder must be Path not {type(working_folder)}"
    assert isinstance(
        sorting_clean_folder, Path
    ), f"sorting_clean_folder must be Path not {type(sorting_clean_folder)}"

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
                **sorting_comparison_params,
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
    pre_check=True,
    sorting=True,
    postproc=True,
    compare_sorters=False,
    compute_alignment=False,
    time_stamp="default",
):
    """Overarching function to run pipeline

    Parameters
    ----------
    pre_check: bool
        to run or not prechecks

    sorting: path
        to run or not sorting

    postproc: None | list | tuple
        to run or not postprocessing

    compare_sorters: bool
        to run or not sorting comparison

    compute_alignment: bool
        to run or not alignment computation

    time_stamp: str
        time stamp to be used in folder name


    """
    assert isinstance(
        pre_check, bool
    ), f"pre_check must be boolean not {type(pre_check)}"
    assert isinstance(sorting, bool), f"sorting must be boolean not {type(sorting)}"
    assert isinstance(postproc, bool), f"postproc must be boolean not {type(postproc)}"
    assert isinstance(
        compare_sorters, bool
    ), f"compare_sorters must be boolean not {type(compare_sorters)}"
    assert isinstance(
        compute_alignment, bool
    ), f"compute_alignment must be boolean not {type(compute_alignment)}"
    assert isinstance(time_stamp, str), f"time_stamp must be str not {type(time_stamp)}"

    # Starts loop over recordings in recording list
    for (
        implant_name,
        rec_name,
        time_range,
        depth_range,
        drift_correction,
    ) in recording_list:
        openephys_folder = concatenate_openephys_folder_path(implant_name, rec_name)
        cache_working_folder = concatenate_working_folder_path(
            implant_name, rec_name, time_range, depth_range, time_stamp
        )
        NAS_sorting_folder = concatenate_clean_sorting_path(
            implant_name, rec_name, time_range, depth_range, time_stamp, ""
        )

        print(f"Data is coming from {openephys_folder}")
        print(f"Cache is saved at {cache_working_folder}")

        if any([pre_check, sorting, postproc]):
            # Get relevant recording
            rec_preprocess = get_preprocess_recording(
                openephys_folder, cache_working_folder, time_range, depth_range
            )
            print('yepyep')
            print(rec_preprocess)
            print(rec_preprocess.is_filtered())


            if pre_check:
                # Run pre-sorting checks
                run_pre_sorting_checks(rec_preprocess, cache_working_folder)

            if sorting:
                # Run sorting pipeline
                run_sorting_pipeline(
                    rec_preprocess, cache_working_folder, drift_correction
                )

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


if __name__ == "__main__":
    pre_check = True
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
        time_stamp=time_stamp,
    )