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


# Third party imports
from params_viz import base_folder

# Internal imports


def concatenate_available_sorting_paths(brain_area, implant_name, rec_name, node, experiment):
    """Concatenates the sorting paths

    Parameters
    ----------
    brain_area: str
        brain area

    implant_name: str
        implant name

    rec_name: str
        recording name

    node: str
        node name
    
    experiment:
        experiment number

    Returns
    -------
    sorting_folders: list of paths
        available sortings for a recording
    """
    assert isinstance(
        implant_name, str
    ), f"implant_name must be type str not {type(implant_name)}"
    assert isinstance(rec_name, str), f"rec_name must be type str not {type(rec_name)}"

    main_path = base_folder / brain_area / implant_name / "Sortings_clean"

    sorting_folders = list(main_path.glob(f"*-{rec_name}-*/{node.replace(' ', '')}/{experiment}/**/spikes.npy"))

    sorting_folders = [folder.parent for folder in sorting_folders]

    return sorting_folders