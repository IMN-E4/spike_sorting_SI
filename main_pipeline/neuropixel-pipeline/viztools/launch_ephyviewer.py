#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the script to launch ephyviewer.

"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/10/1"  ### Date it was created
__status__ = (
    "Production"  ### Production = still being developed. Else: Concluded/Finished.
)


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
from pathlib import Path

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import spikeinterface.full as si
import ephyviewer
from ephyviewer.myqt import QT

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from utils import *
from params_viz import *
from path_handling_viz import *


def open_my_viewer(
    brain_area,
    implant_name,
    rec_name,
    ap_recording=True,
    mic_spectrogram=True,
    lf_recording=False,
    viz_sorting=False,
    load_sync_channel=False,
    parent=None,
):
    def slice_rec(rec, time_range=None, depth_range=None):
        if time_range is not None:
            rec = slice_rec_time(rec, time_range)

        if depth_range is not None:
            rec = slice_rec_depth(rec, depth_range)

        return rec

    # Find folders
    spikeglx_folder = concatenate_spikeglx_folder_path(brain_area, implant_name, rec_name)

    ## App and viewer objects
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True, parent=parent)

    ### Sources
    if ap_recording:
        recording_spike = si.read_spikeglx(
            spikeglx_folder, stream_id="imec0.ap", load_sync_channel=load_sync_channel
        )
        # recording_spike = slice_rec(recording_spike, time_range, depth_range)
        recording_spike = si.depth_order(recording_spike, flip=True)
        
        sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=recording_spike
        )

        # sig_source0.sample_rate = sig_source0.sample_rate / synchro["a"]
        # sig_source0._t_start = sig_source0._t_start + synchro["b"]

        view0 = ephyviewer.TraceViewer(source=sig_source0, name="ap_raw")
        view0.params["scale_mode"] = "same_for_all"
        for c in range(recording_spike.get_num_channels()):
            if c == 1:
                visible = True
            else:
                visible = False
            view0.by_channel_params[f"ch{c}", "visible"] = visible
        win.add_view(view0)

    if lf_recording:
        recording_lf = si.read_spikeglx(
            spikeglx_folder, stream_id="imec0.lf", load_sync_channel=load_sync_channel
        )

        recording_lf = si.depth_order(recording_lf, flip=True)
        # recording_lf = slice_rec(recording_lf, time_range, depth_range)

        sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf)
        # sig_source2.sample_rate = sig_source2.sample_rate / synchro["a"]
        # sig_source2._t_start = sig_source2._t_start + synchro["b"]

        view2 = ephyviewer.TraceViewer(source=sig_source2, name="signals lf")
        win.add_view(view2)

        # time-freq
        view2_tf = ephyviewer.TimeFreqViewer(
            source=sig_source2, name="timefreq"
        )  # Timefreq view of LFP

        view2.params["scale_mode"] = "same_for_all"
        for c in range(recording_lf.get_num_channels()):
            if c == 1:
                visible = True
            else:
                visible = False
            view2.by_channel_params[f"ch{c}", "visible"] = visible
        win.add_view(view2_tf)

    if mic_spectrogram:
        recording_nidq = si.read_spikeglx(
            spikeglx_folder, stream_id="nidq"
        )

        # recording_nidq = slice_rec(recording_nidq, time_range, None)

        sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=recording_nidq
        )
        view_raw_mic = ephyviewer.TraceViewer(source=sig_source3, name="raw mic")
        view3 = ephyviewer.SpectrogramViewer(
            source=sig_source3, name="signals nidq"
        )

        view3.params["colormap"] = "inferno"
        view3.params.child("scalogram")["overlapratio"] = 0.2
        view3.params.child("scalogram")["binsize"] = 0.02
        for c in range(recording_nidq.get_num_channels()):
            if c == 0:
                visible = True
            else:
                visible = False
            view3.by_channel_params[f"ch{c}", "visible"] = visible
        win.add_view(view_raw_mic)
        win.add_view(view3, tabify_with="raw mic")

    if viz_sorting:
        dia = QT.QFileDialog(
            fileMode=QT.QFileDialog.Directory, acceptMode=QT.QFileDialog.AcceptOpen
        )
        dia.setViewMode(QT.QFileDialog.Detail)
        if dia.exec_():
            folder_names = dia.selectedFiles()
            sorting_folder = folder_names[0]
        else:
            return

        sorting_data = si.load_extractor(sorting_folder)
    #     sorting_folder = concatenate_clean_sorting_path(
    #     implant_name, rec_name, time_range, depth_range, time_stamp, sorter_name
    # ) 
        # t_start = synchro["b"]
        # sr = sorting_data.get_sampling_frequency() / synchro["a"]
        all_spikes = []
        for unit_id in sorting_data.unit_ids:
            spike_times = sorting_data.get_unit_spike_train(unit_id=unit_id) #/ sr + t_start
            all_spikes.append({"time": spike_times, "name": f"Unit#{unit_id}"})
        spike_source = ephyviewer.InMemorySpikeSource(all_spikes)

        view6 = ephyviewer.SpikeTrainViewer(
            source=spike_source, name="sorting"
        )  # spiking traces

        win.add_view(view6)

    return win


# if __name__ == "__main__":
#     implant_name = "Test_Data_troubleshoot"
#     rec_name = "2023-08-23_15-56-05"
#     app = ephyviewer.mkQApp()

#     win = open_my_viewer(implant_name, rec_name)
#     win.show()
#     app.exec_()
