#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the script to launch ephyviewer.

"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2021/10/1"  ### Date it was created
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
from utils import add_probe_to_rec
from params_viz import *


def open_my_viewer(brain_area, bird_name, session_name, node, experiment, parent=None):
    data_folder = (
        base_folder / f"{brain_area}/{bird_name}/Recordings/{session_name}/{node}/"
    )
    experiment = experiment.strip("experiment")
    experiment = int(experiment) - 1

    print(data_folder)

    ## App and viewer objects

    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True, parent=parent)

    ### Sources
    if raw_recording:
        recording = si.read_openephys(data_folder, block_index=experiment)
        if order_by_depth:
            recording = add_probe_to_rec(recording)
            recording = si.depth_order(recording, flip=True)
        sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording)
        view0 = ephyviewer.TraceViewer(source=sig_source0, name="recording traces")
        view0.params["scale_mode"] = "by_channel"
        view0.params["display_labels"] = True
        view0.auto_scale()
        win.add_view(view0)

    if mic_spectrogram:
        recording = si.read_openephys(data_folder, bloc=experiment)
        mic = recording.channel_slice(channel_ids=["ADC1"])
        sig_source1 = ephyviewer.SpikeInterfaceRecordingSource(recording=mic)

        view1 = ephyviewer.SpectrogramViewer(source=sig_source1, name="mic spectrogram")
        win.add_view(view1)

    if bandpassed_recording:
        recording = si.read_openephys(data_folder, block_index=experiment)
        if order_by_depth:
            recording = add_probe_to_rec(recording)
            recording = si.depth_order(recording, flip=True)
        filtered_recording = si.bandpass_filter(
            recording, bandpass_range[0], bandpass_range[1]
        )
        sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(
            recording=filtered_recording
        )
        view2 = ephyviewer.TraceViewer(source=sig_source2, name="signals bandpassed")

        view2.params["scale_mode"] = "by_channel"
        view2.params["display_labels"] = True
        view2.auto_scale()

        win.add_view(view2)

        view3 = ephyviewer.TimeFreqViewer(
            source=sig_source2, name="timefreq sig bandpassed"
        )
        win.add_view(view3)

    if sorting:
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
        all_spikes = []
        for unit_id in sorting_data.unit_ids:
            spike_times_s = sorting_data.get_unit_spike_train(
                unit_id=unit_id, return_times=True
            )
            all_spikes.append({"time": spike_times_s, "name": f"Unit#{unit_id}"})

        spike_source = ephyviewer.InMemorySpikeSource(all_spikes)

        view4 = ephyviewer.SpikeTrainViewer(
            source=spike_source, name="sorting"
        )  # spiking traces
        win.add_view(view4)

    return win


# if __name__ == "__main__":
#     bird_name = "Test_Data_troubleshoot"
#     session_name = "2023-08-23_15-56-05"
#     app = ephyviewer.mkQApp()

#     win = open_my_viewer(bird_name, session_name)
#     win.show()
#     app.exec_()