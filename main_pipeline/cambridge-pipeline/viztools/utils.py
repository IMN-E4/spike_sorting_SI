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