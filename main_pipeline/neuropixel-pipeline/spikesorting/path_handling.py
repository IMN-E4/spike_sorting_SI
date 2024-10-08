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


# Internal imports
from params_NP import base_input_folder, base_sorting_cache_folder


def concatenate_spikeglx_folder_path(implant_name, rec_name):
    """Concatenates the spikeglx directory path

    Parameters
    ----------
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

    spikeglx_folder = base_input_folder / implant_name / "Recordings" / rec_name

    return spikeglx_folder


def concatenate_working_folder_path(
    implant_name,
    rec_name,
    time_range,
    depth_range,
    time_stamp="default",
):
    """Concatenates the working directory path

    Parameters
    ----------
    implant_name: str
        implant name

    rec_name: str
        recording name

    time_range: None | list | tuple
        time range to slice recording

    depth_range: None | list | tuple
        depth range to slice recording

    time_stamp: str
        time stamp on folder. default = current month

    Returns
    -------
    working_folder: Path
        working folder
    """
    assert isinstance(
        implant_name, str
    ), f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), f"depth_range must be type tuple, list or None not {type(depth_range)}"
    assert isinstance(
        time_stamp, str
    ), f"time_stamp must be type str not {type(time_stamp)}"

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{rec_name}-full"
        )

    else:
        time_range = tuple(float(e) for e in time_range)

        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
        )

    if depth_range is not None:
        print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
        working_folder = working_folder / f"depth_{depth_range[0]}_to_{depth_range[1]}"
    else:
        print(f"Using all channels")

    return working_folder


def concatenate_clean_sorting_path_in_NAS(
    implant_name, rec_name, time_range, depth_range, time_stamp, sorter_name
):
    """Concatenates the CLEAN spike sorting folder path IN NAS

    Parameters
    ----------
    implant_name: str
        implant name

    rec_name: str
        recording name

    time_range: None | list | tuple
        time range to slice recording

    depth_range: None | list | tuple
        depth range to slice recording

    time_stamp: str
        time stamp on folder. default = current month

    sorter_name: str
        sorter name

    Returns
    -------
    sorting_clean_folder_in_NAS: Path
        path for saving clean sorting
    """

    assert isinstance(
        implant_name, str
    ), f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"
    assert isinstance(
        time_range, (tuple, list, type(None))
    ), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(
        depth_range, (tuple, list, type(None))
    ), f"depth_range must be type tuple, list or None not {type(depth_range)}"
    assert isinstance(
        time_stamp, str
    ), f"time_stamp must be type str not {type(time_stamp)}"
    assert isinstance(
        sorter_name, str
    ), f"sorter_name must be type str not {type(sorter_name)}"

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        name = f"{time_stamp}-{rec_name}-full"

        sorting_clean_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name
        )

    else:
        time_range = tuple(float(e) for e in time_range)
        name = f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
        sorting_clean_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name
        )

    if depth_range is not None:
        print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
        sorting_clean_folder = (
            sorting_clean_folder / f"depth_{depth_range[0]}_to_{depth_range[1]}"
        )
    else:
        print(f"Using all channels")

    sorting_clean_folder_in_NAS = sorting_clean_folder / sorter_name

    return sorting_clean_folder_in_NAS


def concatenate_synchro_file_path(implant_name, rec_name, time_range, time_stamp):
    """Concatenates path to stream synchronization file

    Parameters
    ----------
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
