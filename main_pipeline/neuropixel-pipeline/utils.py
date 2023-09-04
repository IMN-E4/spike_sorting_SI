#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utils for spike sorting pipeline.
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2022/06/1"
__status__ = (
    "Production"
)


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports

# Third party imports
import spikeinterface.full as si
import numpy as np

# Internal imports
from params_NP import preprocessing_params


def slice_rec_time(rec, time_range):
    """Slice recording in time

    Parameters
    ----------
    rec: spikeinterface object
        recording to apply preprocessing on

    time_range: tuple
        beginning and end time to slice recording

    Returns
    -------
    sliced_rec: spikeinterface object
        time-sliced recording
    """

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
    rec: spikeinterface object
        recording to apply preprocessing on

    depth_range: tuple
        beginning and end depth to slice recording

    Returns
    -------
    sliced_rec: spikeinterface object
        depth-sliced recording
    """

    # Channel Slicing
    print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
    yloc = rec.get_channel_locations()[:, 1]
    keep = (yloc >= depth_range[0]) & (yloc <= depth_range[1])
    keep_chan_ids = rec.channel_ids[keep]
    sliced_rec = rec.channel_slice(channel_ids=keep_chan_ids)
    return sliced_rec


def read_rec(
    spikeglx_folder, stream_id, time_range, depth_range, load_sync_channel=False
):
    """Read recording

    Parameters
    ----------
    spikeglx_folder: path
        path to data

    stream_id: str
        data stream id

    time_range: tuple
        beginning and end time to slice recording

    depth_range: tuple
        beginning and end depth to slice recording

    load_sync_channel: bool
        load or not synchronization channel (TTL)


    Returns
    -------
    rec: spikeinterface object
        time/depth corrected rec
    """
    rec = si.read_spikeglx(
        spikeglx_folder, stream_id=stream_id, load_sync_channel=load_sync_channel
    )

    if time_range is not None:
        rec = slice_rec_time(rec, time_range)

    elif depth_range is not None:
        rec = slice_rec_depth(rec, depth_range)

    return rec


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


def correct_drift(rec, working_folder):
    """Apply drift correction

    Parameters
    ----------
    rec: spikeinterface object
        recording to apply preprocessing on.

    working_folder: path
        working folder


    Returns
    -------
    recording_corrected: spikeinterface object
        drift-corrected rec
    """
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