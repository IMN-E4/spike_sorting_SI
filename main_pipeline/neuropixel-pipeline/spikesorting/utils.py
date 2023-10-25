#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utils for spike sorting pipeline.
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

# Internal imports
from params_NP import preprocessing_params


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
    spikeglx_folder: Path
        path to data

    stream_id: str
        data stream id

    time_range: None | list | tuple
        beginning and end time to slice recording

    depth_range: None | list | tuple
        beginning and end depth to slice recording

    load_sync_channel: bool
        load or not synchronization channel (TTL)


    Returns
    -------
    rec: spikeinterface object
        time/depth corrected rec
    """
    assert isinstance(
        spikeglx_folder, Path
    ), f"spikeglx_folder must be Path not {type(spikeglx_folder)}"
    assert isinstance(stream_id, str), f"stream_id must be str not {type(stream_id)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), f"depth_range must be type tuple, list or None not {type(depth_range)}"
    assert isinstance(
        load_sync_channel, bool
    ), f"load_sync_channel must be boolean not {type(load_sync_channel)}"

    rec = si.read_spikeglx(
        spikeglx_folder, stream_id=stream_id, load_sync_channel=load_sync_channel
    )

    if time_range is not None:
        rec = slice_rec_time(rec, time_range)

    if depth_range is not None:
        rec = slice_rec_depth(rec, depth_range)

    return rec


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
        rec, (si.SpikeGLXRecordingExtractor, si.FrameSliceRecording, si.ChannelSliceRecording)
    ), f"rec must be type spikeinterface SpikeGLXRecordingExtractor or FrameSliceRecording or ChannelSliceRecording not {type(rec)}"

    # Phase shift correction
    rec = si.phase_shift(rec)

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
    rec: spikeinterface BinaryFolderRecording
        recording to apply preprocessing on.

    working_folder: Path
        working folder

    Returns
    -------
    recording_corrected: spikeinterface object
        drift-corrected rec
    """
    assert isinstance(
        working_folder, Path
    ), f"working_folder must be Path not {type(working_folder)}"
    assert isinstance(
        rec, si.SpikeGLXRecordingExtractor
    ), f"rec must be type spikeinterface SpikeGLXRecordingExtractor not {type(rec)}"

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

def propagate_params(original_file_path, output_path):

    assert isinstance(original_file_path, Path), f'original_file_path must be Path not {type(original_file_path)}'
    assert isinstance(output_path, Path), f'output_path must be Path not {type(output_path)}'
    assert original_file_path.exists(), 'revise path of original file'

    output_file_path = output_path / original_file_path.parts[-1]

    copy(original_file_path.as_posix(), output_file_path.as_posix())