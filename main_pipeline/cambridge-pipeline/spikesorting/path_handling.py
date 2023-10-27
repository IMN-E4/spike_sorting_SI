#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""# Internal imports
Path handling for Cambridge Neurotech sorting pipeline.

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
from datetime import datetime

# Third party imports
import spikeinterface.full as si

# Internal imports
from params_CN import base_input_folder, base_sorting_cache_folder


def concatenate_openephys_folder_path(implant_name, rec_name, node_number):
    """Concatenates the spikeglx directory path

    Parameters
    ----------
    implant_name: str
        implant name

    rec_name: str
        recording name

    node_number: int
        node number

    Returns
    -------
    openephys_folder: Path
        spikeglx folder
    """
    assert isinstance(implant_name, str),  f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"
    assert isinstance(node_number, int),  f"node_name must be type int not {type(node_number)}"
    
    openephys_folder = base_input_folder / implant_name / "Recordings" / rec_name / f"Record Node {node_number}"
    
    return openephys_folder


def concatenate_working_folder_path(
    implant_name,
    rec_name,
    node_number, 
    experiment_number,
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
    
    node_number: int
        node number

    experiment_number: int
        experiment number

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
    assert isinstance(implant_name, str),  f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"
    assert isinstance(node_number, int),  f"node_name must be type int not {type(node_number)}"
    assert isinstance(experiment_number, int), f"experiment_number must be type int not {type(experiment_number)}"
    assert isinstance(time_range, (tuple, list, type(None))), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(depth_range, (tuple, list, type(None))), f"depth_range must be type tuple, list or None not {type(depth_range)}"
    assert isinstance(time_stamp, str), f"time_stamp must be type str not {type(time_stamp)}"

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{rec_name}-full"
            / f"RecordNode{node_number}"
            / f"experiment{experiment_number}"
        )

    else:
        time_range = tuple(float(e) for e in time_range)

        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
            / f"RecordNode{node_number}"
            / f"experiment{experiment_number}"
        )

    if depth_range is not None:
        print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
        working_folder = working_folder / f"depth_{depth_range[0]}_to_{depth_range[1]}"
    else:
        print(f"Using all channels")

    return working_folder


def concatenate_clean_sorting_path(
    implant_name, rec_name, node_number, experiment_number, time_range, depth_range, time_stamp, sorter_name
):
    """Concatenates the CLEAN spike sorting folder path

    Parameters
    ----------
    implant_name: str
        implant name

    rec_name: str
        recording name
    
    node_number: int
        node number

    experiment_number: int
        experiment number

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
    sorting_clean_folder: Path
        path for saving clean sorting
    """
    assert isinstance(implant_name, str),  f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"
    assert isinstance(node_number, int),  f"node_name must be type int not {type(node_number)}"
    assert isinstance(experiment_number, int), f"experiment_number must be type int not {type(experiment_number)}"
    assert isinstance(time_range, (tuple, list, type(None))), f"time_range must be type tuple, list or None not {type(time_range)}"
    assert isinstance(depth_range, (tuple, list, type(None))), f"depth_range must be type tuple, list or None not {type(depth_range)}"
    assert isinstance(time_stamp, str), f"time_stamp must be type str not {type(time_stamp)}"
    assert isinstance(sorter_name, str), f"sorter_name must be type str not {type(sorter_name)}"

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        name = f"{time_stamp}-{rec_name}-full"

        sorting_clean_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / f"RecordNode{node_number}" / f"experiment{experiment_number}"
        )

    else:
        time_range = tuple(float(e) for e in time_range)
        name = f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
        sorting_clean_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / f"RecordNode{node_number}" / f"experiment{experiment_number}"
        )

    if depth_range is not None:
        print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
        sorting_clean_folder = (
            sorting_clean_folder / f"depth_{depth_range[0]}_to_{depth_range[1]}"
        )
    else:
        print(f"Using all channels")

    sorting_clean_folder = sorting_clean_folder / sorter_name

    return sorting_clean_folder