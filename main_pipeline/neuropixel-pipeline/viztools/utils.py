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
import glob

# Third party imports
import spikeinterface.full as si
import pandas as pd

# Internal imports


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
        rec, (si.SpikeGLXRecordingExtractor, si.FrameSliceRecording)
    ), f"rec must be type spikeinterface SpikeGLXRecordingExtractor or FrameSliceRecording not {type(rec)}"
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


def identify_time_and_depth_range(sorting_folder):
    """Identify time and depth ranges from sorting foldername

    Parameters
    ----------
    sorting_folder: str or Path
        sorting folder

    Returns
    -------
    rec_name_sorting: str
        recording name

    time_stamp: str
        time stamp

    depth_range: tuple[int, int]
        depth range

    time_range: tuple[int, int]
        time range
    """
    assert isinstance(
        sorting_folder, (str, Path)
    ), f"sorting_folder must be type str or Path not {type(sorting_folder)}"

    sorting_folder = Path(sorting_folder)
    depth_range = None
    time_range = None

    split_parts = sorting_folder.parts[1:]

    for part in split_parts:
        if "depth" in part:
            depth = part.split("_")
            depth_beg = int(depth[1])
            depth_end = int(depth[3])
            depth_range = (depth_beg, depth_end)

        if ("Rec_" in part) or ("session" in part):
            print('here')
            split = part.split("-")
            time_stamp = "-".join(split[:2])
            rec_name_sorting = split[2]
            time_range_tmp = split[3]

            if time_range_tmp != "full":
                times = time_range_tmp.split("to")
                time_beg = int(times[0])
                time_end = int(times[1])
                time_range = (time_beg, time_end)

    return rec_name_sorting, time_stamp, depth_range, time_range


def find_data_in_nas(root_to_data, target_folder):
    """Find database in NAS

    Parameters
    ----------
    root_to_data: str or Path
        root folder where data is
    
    target_folder: str
        target NP folder

    Returns
    -------
    df: pandas DataFrame
        database dataframe
    """
    
    assert isinstance(
        root_to_data, (str, Path)
    ), f"root_to_data must be type str or Path not {type(root_to_data)}"
    
    target_path = Path(root_to_data) / target_folder
    keyword = "*ap.meta"
    df = pd.DataFrame(
        columns=[
            "root",
            "rec_system",
            "brain_area",
            "implant_name",
            "intermediate",
            "rec_name",
        ]
    )

    glob_path = target_path / f"**/{keyword}"
    all_files = glob.glob(glob_path.as_posix(), recursive=True)

    ## Populate dataframe
    for file in all_files:
        split_parts = Path(file).parts[1:-1]
        df.loc[len(df)] = split_parts

    return df

