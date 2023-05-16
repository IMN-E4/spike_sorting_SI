#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to visualize processed data.
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2022/06/1"  ### Date it was created
__status__ = (
    "Production"  ### Production = still being developed. Else: Concluded/Finished.
)

####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
import json
from pathlib import Path

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import numpy as np
import ephyviewer
import matplotlib.pyplot as plt
import spikeinterface.full as si
from ephyviewer.myqt import QT
from ephyviewer.tools import ParamDialog

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from path_handling import (
    get_sorting_folder,
    get_spikeglx_folder,
    get_working_folder,
    get_synchro_file,
)
from utils import *


def open_ephyviewer_mainwindow(
    key_tuple,
    sorter_name,
    raw_ap=False,
    raw_lfp=False,
    mic=True,
    filtered_ap=False,
    filtered_lfp=False,
    sorting_panel=False,
    lf_filt_range=None
):  
    
    def slice_rec(rec, time_range=None, depth_range=None):
        if time_range is not None:
            rec = slice_rec_time(rec, time_range)
        
        if depth_range is not None:
            rec = slice_rec_depth(rec, depth_range)

        return rec

    #Unpacking
    implant_name, rec_name, time_range, depth_range, _, time_stamp = key_tuple

    spikeglx_folder = get_spikeglx_folder(implant_name, rec_name)

    sorting_folder = get_sorting_folder(
        implant_name, rec_name, time_range, depth_range, time_stamp, sorter_name
    )
    working_dir = get_working_folder(
        implant_name, rec_name, time_range, depth_range, time_stamp
    )

    synchro_file = get_synchro_file(implant_name, rec_name, time_range, time_stamp)
    with open(synchro_file) as f:
        synchro = json.load(f)

    # App and viewer objects
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

    ### Sources
    if raw_ap:
        recording_spike = si.read_spikeglx(
            spikeglx_folder, stream_id="imec0.ap"
        )  # ap file

        recording_spike = slice_rec(recording_spike, time_range, depth_range)

        sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=recording_spike
        )
        sig_source0.sample_rate = sig_source0.sample_rate / synchro["a"]
        sig_source0._t_start = sig_source0._t_start + synchro["b"]

        view0 = ephyviewer.TraceViewer(
            source=sig_source0, name="ap"
        )  # Trace of ap signal

        view0.params["scale_mode"] = "same_for_all"
        for c in range(recording_spike.get_num_channels()):
            if c % 50 == 0:
                visible = True
            else:
                visible = False
            view0.by_channel_params[f"ch{c}", "visible"] = visible

        win.add_view(view0)

    if filtered_ap:
        print(working_dir)
        recording_spike_filt = si.load_extractor(
            working_dir / "preprocess_recording"
        )  # ap file
        sig_source1 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=recording_spike_filt
        )
        sig_source1.sample_rate = sig_source1.sample_rate / synchro["a"]
        sig_source1._t_start = sig_source1._t_start + synchro["b"]

        view1 = ephyviewer.TraceViewer(
            source=sig_source1, name="ap_filtered"
        )  # Trace of ap signal

        view1.params["scale_mode"] = "same_for_all"
        for c in range(recording_spike_filt.get_num_channels()):
            if c % 50 == 0:
                visible = True
            else:
                visible = False
            view1.by_channel_params[f"ch{c}", "visible"] = visible

        win.add_view(view1)

    if raw_lfp:
        recording_lf = si.read_spikeglx(spikeglx_folder, stream_id="imec0.lf")  # lfp
        
        recording_lf = slice_rec(recording_lf, time_range, depth_range)

        sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=recording_lf
        )  # lfp
        sig_source2.sample_rate = sig_source2.sample_rate / synchro["a"]
        sig_source2._t_start = sig_source2._t_start + synchro["b"]

        view2 = ephyviewer.TraceViewer(
            source=sig_source2, name="signals lf"
        )  # Trace of LFP
        win.add_view(view2)

        # time-freq
        view2_tf = ephyviewer.TimeFreqViewer(
            source=sig_source2, name="timefreq"
        )  # Timefreq view of LFP
        win.add_view(view2_tf)

        view2.params["scale_mode"] = "same_for_all"
        for c in range(recording_lf.get_num_channels()):
            if c % 50 == 0:
                visible = True
            else:
                visible = False
            view2.by_channel_params[f"ch{c}", "visible"] = visible

    if mic:
        recording_nidq = si.read_spikeglx(
            spikeglx_folder, stream_id="nidq"
        )  # microphone

        recording_nidq = slice_rec(recording_nidq, time_range, None)

        sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=recording_nidq
        )  # microphone
        view3 = ephyviewer.SpectrogramViewer(
            source=sig_source3, name="signals nidq"
        )  # Trace of Microphone
        win.add_view(view3)

        view3.params["colormap"] = "inferno"
        view3.params.child("scalogram")["overlapratio"] = 0.2
        view3.params.child("scalogram")["binsize"] = 0.02

        for c in range(recording_nidq.get_num_channels()):
            if c == 0:
                view3.by_channel_params[f"ch{c}", "visible"] = True
                view3.by_channel_params[f"ch{c}", "clim_min"] = -30
                view3.by_channel_params[f"ch{c}", "clim_max"] = 60
            else:
                view3.by_channel_params[f"ch{c}", "visible"] = False

    if filtered_lfp:
        recording_lf_filt = si.read_spikeglx(spikeglx_folder, stream_id="imec0.lf")  # lfp
        
        recording_lf_filt = slice_rec(recording_lf_filt, time_range, depth_range)
        
        recording_f = si.bandpass_filter(recording_lf_filt, lf_filt_range[0], lf_filt_range[1])
        sig_source4 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=recording_f
        )  # filtered LFP trace
        sig_source4.sample_rate = sig_source4.sample_rate / synchro["a"]
        sig_source4._t_start = sig_source4._t_start + synchro["b"]
        view4 = ephyviewer.TraceViewer(
            source=sig_source4, name="signals filtered lfp"
        )  # Trace of LFP filtered
        win.add_view(view4)

    if sorting_panel:
        sorting = si.read_npz_sorting(sorting_folder / "sorting_cached.npz")
        t_start = synchro["b"]
        sr = recording_spike.get_sampling_frequency() / synchro["a"]
        all_spikes = []
        for unit_id in sorting.unit_ids:
            spike_times = sorting.get_unit_spike_train(unit_id=unit_id) / sr + t_start
            all_spikes.append({"time": spike_times, "name": f"Unit#{unit_id}"})
        spike_source = ephyviewer.InMemorySpikeSource(all_spikes)

        view4 = ephyviewer.SpikeTrainViewer(
            source=spike_source, name="sorting"
        )  # spiking traces
        win.add_view(view4)

    # Display
    win.show()
    app.exec_()


def select_folder_and_open():
    app = ephyviewer.mkQApp()
    dia = QT.QFileDialog(
        fileMode=QT.QFileDialog.Directory, acceptMode=QT.QFileDialog.AcceptOpen
    )
    dia.setViewMode(QT.QFileDialog.Detail)
    if dia.exec_():
        folder_names = dia.selectedFiles()
        folder_name = folder_names[0]
    else:
        return

    params = [
        {"name": "mic", "type": "bool", "value": True},
        {"name": "raw_ap", "type": "bool", "value": False},
        {"name": "filtered_ap", "type": "bool", "value": False},
        {"name": "raw_lfp", "type": "bool", "value": False},
        {"name": "filtered_lfp", "type": "bool", "value": False},
    ]
    dia = ParamDialog(params, title="Select streams")
    if dia.exec_():
        kwargs_streams = dia.get()

    else:
        return

    print(kwargs_streams)
    print(folder_name)

    open_ephyviewer_mainwindow(folder_name, **kwargs_streams)


def test_open_data():
    key_tuple = (
        "Anesth_21_01_2023",
        "Rec_21_01_2023_3_g0",
        [1500, 2670],
        [3200, 3840],
        False,
        "2023-01",
    )
    sorter_name = "kilosort2_5"
    open_ephyviewer_mainwindow(
        key_tuple,
        sorter_name,
        raw_ap=True,
        raw_lfp=False,
        mic=True,
        filtered_ap=True,
        filtered_lfp=False,
        sorting_panel=True,
        lf_filt_range=None
    )


if __name__ == "__main__":
    # select_folder_and_open()
    test_open_data()
