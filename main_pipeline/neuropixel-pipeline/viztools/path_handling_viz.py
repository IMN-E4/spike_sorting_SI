#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Path handling for Neuropixel sorting pipeline.

"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/05/1"
__status__ = "Production"


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports
from datetime import datetime

# Third party imports
import spikeinterface.full as si
import pandas as pd

# Internal imports
from params_viz import base_folder


def concatenate_spikeglx_folder_path(brain_area, implant_name, rec_name):
    """Concatenates the spikeglx directory path

    Parameters
    ----------
    brain_area: str
        brain area

    implant_name: str
        implant name

    rec_name: str
        recording name

    Returns
    -------
    spikeglx_folder: Path
        spikeglx folder
    """
    assert isinstance(
        implant_name, str
    ), f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"

    spikeglx_folder = base_folder / brain_area / implant_name / "Recordings" / rec_name

    return spikeglx_folder


def concatenate_synchro_file_path(
    brain_area, implant_name, rec_name, time_range, time_stamp
):
    """Concatenates path to stream synchronization file

    Parameters
    ----------
    brain_area: str
        brain area

    implant_name: str
        implant name

    rec_name: str
        recording name

    time_range: None | list | tuple
        time range to slice recording

    time_stamp: str
        time stamp on folder. default = current month


    Returns
    -------
    synchro_file: Path
        path synchronization json
    """
    assert isinstance(
        implant_name, str
    ), f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(
        time_stamp, str
    ), f"time_stamp must be type str not {type(time_stamp)}"

    base_input_folder = base_folder / brain_area

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        name = f"{time_stamp}-{rec_name}-full"

        synchro_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / "synchro"
        )

    else:
        time_range = tuple(float(e) for e in time_range)
        name = f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
        synchro_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / "synchro"
        )

    synchro_file = synchro_folder / "synchro_imecap_corr_on_nidq.json"

    return synchro_file

def concatenate_available_sorting_paths(brain_area, implant_name, rec_name):
    """Concatenates the sorting paths

    Parameters
    ----------
    brain_area: str
        brain area
    
    implant_name: str
        implant name

    rec_name: str
        recording name

    Returns
    -------
    sorting_folders: list of paths
        available sortings for a recording
    """
    assert isinstance(
        implant_name, str
    ), f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"

    from pathlib import Path
    base_folder = Path("/nas/Neuropixel_Recordings/")

    main_path = base_folder / brain_area / implant_name / "Sortings_clean"

    sorting_folders = list(main_path.glob(f"*-{rec_name}-*/**/sorting_cached.npz"))

    sorting_folders = [folder.parent for folder in sorting_folders]

    return sorting_folders