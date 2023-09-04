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
        rec, si.SpikeGLXRecordingExtractor
    ), "rec must be type spikeinterface SpikeGLXRecordingExtractor"

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
    rec: spikeinterface SpikeGLXRecordingExtractor or FrameSliceRecording
        recording to apply preprocessing on.

    Returns
    -------
    with_probe_rec: spikeinterface object
        rec with probe attached
    """
    assert isinstance(
        rec, (si.SpikeGLXRecordingExtractor, si.FrameSliceRecording)
    ), f"rec must be type spikeinterface SpikeGLXRecordingExtractor or FrameSliceRecording not {type(rec)}"


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
    rec: spikeinterface SpikeGLXRecordingExtractor or FrameSliceRecording
        recording to apply preprocessing on.

    Returns
    -------
    rec_preproc: spikeinterface object
        preprocessed rec
    """
    assert isinstance(
        rec, (si.SpikeGLXRecordingExtractor, si.FrameSliceRecording)
    ), f"rec must be type spikeinterface SpikeGLXRecordingExtractor or FrameSliceRecording not {type(rec)}"

    # Bandpass filter
    rec = si.bandpass_filter(
        rec,
        freq_min=preprocessing_params["highpass"],
        freq_max=preprocessing_params["lowpass"],
    )

    # Common referencing
    rec_preproc = si.common_reference(
        rec,
        reference=preprocessing_params["reference"],
        local_radius=preprocessing_params["local_radius"],
    )
    return rec_preproc


def slice_rec_time(rec, time_range):
    """Slice recording in time

    Parameters
    ----------
    rec: si.SpikeGLXRecordingExtractor
        recording to apply preprocessing on

    time_range: None | list | tuple
        beginning and end time to slice recording

    Returns
    -------
    sliced_rec: spikeinterface object
        time-sliced recording
    """
    assert isinstance(
        rec, si.SpikeGLXRecordingExtractor
    ), f"rec must be type spikeinterface BinaryFolderRecording not {type(rec)}"
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
    rec: spikeinterface SpikeGLXRecordingExtractor
        recording to apply preprocessing on

    depth_range: None | list | tuple
        beginning and end depth to slice recording

    Returns
    -------
    sliced_rec: spikeinterface object
        depth-sliced recording
    """
    assert isinstance(
        rec, si.SpikeGLXRecordingExtractor
    ), f"rec must be type spikeinterface SpikeGLXRecordingExtractor not {type(rec)}"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), f"depth_range must be type tuple, list or None not {type(depth_range)}"

    print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
    yloc = rec.get_channel_locations()[:, 1]
    keep = (yloc >= depth_range[0]) & (yloc <= depth_range[1])
    keep_chan_ids = rec.channel_ids[keep]
    sliced_rec = rec.channel_slice(channel_ids=keep_chan_ids)
    return sliced_rec


def read_rec(openephys_folder, time_range, depth_range):
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
    rec: spikeinterface object
        time/depth corrected rec
    """
    assert isinstance(openephys_folder, Path), f"openephys_folder must be Path not {type(openephys_folder)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), f"depth_range must be type tuple, list or None not {type(depth_range)}"

    rec = si.openephys(openephys_folder)

    if time_range is not None:
        rec = slice_rec_time(rec, time_range)

    elif depth_range is not None:
        rec = slice_rec_depth(rec, depth_range)

    return rec