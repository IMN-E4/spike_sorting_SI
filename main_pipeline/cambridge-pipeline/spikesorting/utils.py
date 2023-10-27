#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utils for Cambridge Neurotech spike sorting pipeline.
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2022/06/1"
__status__ = "Production"


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports
from pathlib import Path
from shutil import copy

# Third party imports
import spikeinterface.full as si
import numpy as np
from probeinterface import get_probe

# Internal imports
from params_CN import preprocessing_params, probe_type, amp_type


####### Functions
def correct_drift(rec, working_folder):
    """Apply drift correction

    Parameters
    ----------
    rec: spikeinterface BinaryFolderRecording
        recording to apply preprocessing on.

    working_folder: Path
        working folder

    Returns
    -------
    recording_corrected: spikeinterface object
        drift-corrected rec
    """
    assert isinstance(working_folder, Path), "working_folder must be Path"
    assert isinstance(
        rec, si.OpenEphysBinaryRecordingExtractor
    ), "rec must be type spikeinterface OpenEphysBinaryRecordingExtractor"

    motion_file0 = working_folder / "motion.npy"
    motion_file1 = working_folder / "motion_temporal_bins.npy"
    motion_file2 = working_folder / "motion_spatial_bins.npy"

    if motion_file0.exists():
        motion = np.load(motion_file0)
        temporal_bins = np.load(motion_file1)
        spatial_bins = np.load(motion_file2)
    else:
        raise "drift params not computer! run pre sorting checks first!"

    recording_corrected = si.correct_motion(  ### is this valid?
        rec, motion, temporal_bins, spatial_bins
    )
    return recording_corrected

def add_probe_to_rec(rec):
    """Apply probe configuration to recording

    Parameters
    ----------
    rec: spikeinterface OpenEphysBinaryRecordingExtractor or FrameSliceRecording
        recording to apply preprocessing on.

    Returns
    -------
    with_probe_rec: spikeinterface object
        rec with probe attached
    """
    assert isinstance(
        rec, (si.OpenEphysBinaryRecordingExtractor, si.FrameSliceRecording)
    ), f"rec must be type spikeinterface OpenEphysBinaryRecordingExtractor or FrameSliceRecording not {type(rec)}"


    # Add probe here
    probe = get_probe('cambridgeneurotech', probe_type)
    probe.wiring_to_device(amp_type)
    with_probe_rec = rec.set_probe(probe, group_mode='by_shank')
    # print(with_probe_rec.get_property('group'))

    return with_probe_rec


def apply_preprocess(rec):
    """Apply lazy preprocessing chain.

    Parameters
    ----------
    rec: spikeinterface OpenEphysBinaryRecordingExtractor or FrameSliceRecording or si.ChannelSliceRecording
        recording to apply preprocessing on.

    Returns
    -------
    rec_preproc: spikeinterface object
        preprocessed rec
    """
    assert isinstance(
        rec, (si.OpenEphysBinaryRecordingExtractor, si.FrameSliceRecording, si.ChannelSliceRecording)
    ), f"rec must be type spikeinterface OpenEphysBinaryRecordingExtractor or FrameSliceRecording or si.ChannelSliceRecording not {type(rec)}"

    print(rec.get_probe())
    if preprocessing_params["order_by_depth"]:
        rec = si.depth_order(rec)

    # Bandpass filter
    rec_filter = si.bandpass_filter(rec, **preprocessing_params["bandpass_filter"])
    
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec_filter,  **preprocessing_params["bad_channels"])

    rec_preproc = rec_filter.remove_channels(bad_channel_ids)

    # Common referencing
    rec_preproc = si.common_reference(rec_preproc, **preprocessing_params["common_reference"])
    # rec_preproc.annotate(is_filtered=True)

    return rec_preproc


def slice_rec_time(rec, time_range):
    """Slice recording in time

    Parameters
    ----------
    rec: si.ChannelSliceRecording
        recording to apply preprocessing on

    time_range: None | list | tuple
        beginning and end time to slice recording

    Returns
    -------
    sliced_rec: spikeinterface object
        time-sliced recording
    """
    assert isinstance(
        rec, si.ChannelSliceRecording
    ), f"rec must be type spikeinterface ChannelSliceRecording not {type(rec)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), f"time_range must be type tuple, list or None not {type(time_range)}"

    fs = rec.get_sampling_frequency()

    # Time slicing
    print(f"Time slicing between {time_range[0]} and {time_range[1]}")
    time_range = tuple(float(e) for e in time_range)
    frame_range = (int(t * fs) for t in time_range)
    sliced_rec = rec.frame_slice(*frame_range)
    return sliced_rec


def slice_rec_depth(rec, depth_range):
    """Slice recording in depth

    Parameters
    ----------
    rec: spikeinterface ChannelSliceRecording or FrameSliceRecording
        recording to apply preprocessing on

    depth_range: None | list | tuple
        beginning and end depth to slice recording

    Returns
    -------
    sliced_rec: spikeinterface object
        depth-sliced recording
    """
    assert isinstance(
        rec, (si.ChannelSliceRecording, si.FrameSliceRecording)
    ), f"rec must be type spikeinterface ChannelSliceRecording or FrameSliceRecording not {type(rec)}"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), f"depth_range must be type tuple, list or None not {type(depth_range)}"

    print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
    yloc = rec.get_channel_locations()[:, 1]
    keep = (yloc >= depth_range[0]) & (yloc <= depth_range[1])
    keep_chan_ids = rec.channel_ids[keep]
    sliced_rec = rec.channel_slice(channel_ids=keep_chan_ids)
    return sliced_rec


def read_rec(openephys_folder, experiment_number, time_range, depth_range):
    """Read recording

    Parameters
    ----------
    openephys_folder: path
        path to data

    time_range: tuple
        beginning and end time to slice recording

    depth_range: tuple
        beginning and end depth to slice recording


    Returns
    -------
    with_probe_rec: spikeinterface object
        time/depth corrected rec with probe attached
    """
    assert isinstance(openephys_folder, Path), f"openephys_folder must be Path not {type(openephys_folder)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), f"depth_range must be type tuple, list or None not {type(depth_range)}"

    rec = si.read_openephys(openephys_folder, block_index=experiment_number-1)

    # attach probe
    with_probe_rec = add_probe_to_rec(rec)

    if time_range is not None:
        with_probe_rec = slice_rec_time(with_probe_rec, time_range)

    if depth_range is not None:
        with_probe_rec = slice_rec_depth(with_probe_rec, depth_range)

    return with_probe_rec

def propagate_params(original_file_path, output_path):

    assert isinstance(original_file_path, Path), f'original_file_path must be Path not {type(original_file_path)}'
    assert isinstance(output_path, Path), f'output_path must be Path not {type(output_path)}'
    assert original_file_path.exists(), 'revise path of original file'

    output_file_path = output_path / original_file_path.parts[-1]

    copy(original_file_path.as_posix(), output_file_path.as_posix())


