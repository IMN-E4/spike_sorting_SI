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
import glob

# Third party imports
import spikeinterface.full as si
from probeinterface import get_probe
import pandas as pd

# Internal imports
from params_viz import probe_type, amp_type


####### Functions
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

def find_data_in_nas(root_to_data="/nas"):
    """Find database in NAS

    Parameters
    ----------
    root_to_data: str or Path
        root folder where data is

    Returns
    -------
    df: pandas DataFrame
        database dataframe
    """
    
    assert isinstance(
        root_to_data, (str, Path)
    ), f"root_to_data must be type str or Path not {type(root_to_data)}"
    
    target_path = Path(root_to_data) / "Cambridge_Recordings" / "**" / "Recordings"
    keyword = "experiment*"
    df = pd.DataFrame(
        columns=[
            "root",
            "rec_system",
            "brain_area",
            "implant_name",
            "intermediate",
            "rec_name",
            "node",
            "experiment"
        ]
    )

    glob_path = target_path / f"**/{keyword}"
    all_files = glob.glob(glob_path.as_posix(), recursive=True)

    ## Populate dataframe
    for file in all_files:
        split_parts = Path(file).parts[1:]
        df.loc[len(df)] = split_parts

    return df
