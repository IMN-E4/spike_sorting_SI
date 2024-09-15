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

###########
# To do   #
###########
## Add camera!

####################
# Libraries        #
####################

# Standard imports
import json
from pathlib import PosixPath

# Third party imports
import spikeinterface.full as si
import ephyviewer

# Internal imports
from utils import *
from path_handling_viz import *
from params_viz import min_freq_envelope, max_freq_envelope, song_channel_name, envelope_kernel_size
from song_envelope import make_song_envelope


def open_my_viewer(
    brain_area,
    implant_name,
    rec_name,
    ap_recording=True,
    mic_spectrogram=True,
    lf_recording=False,
    viz_sorting=False,
    camera=False,
    align_streams=False,
    load_sync_channel=False,
    order_by_depth=True,
    parent=None,
):
    assert (
        load_sync_channel and order_by_depth
    ) is False, "It is not possible to have load_sync_channel and order_by_depth as True at the same time!"

    # Find folders
    spikeglx_folder = concatenate_spikeglx_folder_path(
        brain_area, implant_name, rec_name
    )

    time_range = None
    depth_range = None

    ## App and viewer objects
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True, parent=parent)

    ### Sources
    if type(viz_sorting) == PosixPath:
        sorting_folder = viz_sorting

        # Identify if recording was sliced in time and/or depth
        (
            rec_name_sorting,
            time_stamp,
            depth_range,
            time_range,
        ) = identify_time_and_depth_range(sorting_folder)

        assert rec_name_sorting == rec_name, "rec names dont match!"

        # Collect alignment information [if available]
        if align_streams:
            print("Aligning streams!")
            synchro_file = concatenate_synchro_file_path(
                brain_area,
                implant_name,
                rec_name,
                time_range=time_range,
                time_stamp=time_stamp,
            )
            assert synchro_file.exists(), "couldn't find synchro file in folder"

            with open(synchro_file) as f:
                synchro = json.load(f)

        # Read sorting data
        sorting_data = si.load_extractor(sorting_folder)
        recording_spike = read_rec(
            spikeglx_folder,
            stream_id="imec0.ap",
            time_range=time_range,
            depth_range=depth_range,
            load_sync_channel=load_sync_channel,
        )

        # Create spike trains
        if align_streams:
            sr = recording_spike.get_sampling_frequency() / synchro["a"]

        all_spikes = []
        order_units = sorting_data.unit_ids
        if order_by_depth:
            order_units = order_units[::-1]
            print('BEWARE: Units sorting by depth assumes that sorting object already contains data sorted by depth! Double check in unit list.csv')

        for unit_id in order_units:
            if align_streams:
                spike_times = (
                    sorting_data.get_unit_spike_train(unit_id=unit_id) / sr
                    + synchro["b"]
                )

            else:
                spike_times = sorting_data.get_unit_spike_train(
                    unit_id=unit_id, return_times=True
                )

            all_spikes.append({"time": spike_times, "name": f"Unit#{unit_id}"})
        spike_source = ephyviewer.InMemorySpikeSource(all_spikes)

        spike_train_view = ephyviewer.SpikeTrainViewer(
            source=spike_source, name="sorting"
        )

        win.add_view(spike_train_view)

    if ap_recording:
        if viz_sorting == False:
            # Read recording
            recording_spike = read_rec(
                spikeglx_folder,
                stream_id="imec0.ap",
                time_range=time_range,
                depth_range=depth_range,
                load_sync_channel=load_sync_channel,
            )

        # Order recording by depth [probe tip at bottom]
        if order_by_depth:
            recording_spike = si.depth_order(recording_spike, flip=True)

        # Create ephyviewer source
        source_ap = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_spike)

        # Align traces if possible
        if align_streams:
            source_ap.sample_rate = source_ap.sample_rate / synchro["a"]
            source_ap._t_start = source_ap._t_start + synchro["b"]

        # Create view and display only one channel to make things faster
        ap_trace_view = ephyviewer.TraceViewer(source=source_ap, name="ap_raw")
        ap_trace_view.params["scale_mode"] = "same_for_all"
        for c in range(recording_spike.get_num_channels()):
            if c == 1:
                visible = True
            else:
                visible = False
            ap_trace_view.by_channel_params[f"ch{c}", "visible"] = visible

        win.add_view(ap_trace_view)

    if lf_recording:
        # Read recording
        recording_lf = read_rec(
            spikeglx_folder,
            stream_id="imec0.lf",
            time_range=time_range,
            depth_range=depth_range,
            load_sync_channel=load_sync_channel,
        )

        # Order recording by depth [probe tip at bottom]
        if order_by_depth:
            recording_lf = si.depth_order(recording_lf, flip=True)

        # Create ephyviewer source
        source_lf = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf)

        # Align traces if possible
        if align_streams:
            source_lf.sample_rate = source_lf.sample_rate / synchro["a"]
            source_lf._t_start = source_lf._t_start + synchro["b"]

        # Create view and display only one channel to make things faster
        lf_trace_view = ephyviewer.TraceViewer(source=source_lf, name="signals lf")
        lf_trace_view.params["scale_mode"] = "same_for_all"

        # Plot time-freq
        lf_tf_view = ephyviewer.TimeFreqViewer(source=source_lf, name="time-freq")

        for c in range(recording_lf.get_num_channels()):
            if c == 1:
                visible = True
            else:
                visible = False
            lf_trace_view.by_channel_params[f"ch{c}", "visible"] = visible
            lf_tf_view.by_channel_params[f"ch{c}", "visible"] = visible

        win.add_view(lf_tf_view)
        win.add_view(lf_trace_view, tabify_with="time-freq")

    if mic_spectrogram:
        # Read recording
        recording_nidq = read_rec(
            spikeglx_folder,
            stream_id="nidq",
            time_range=time_range,
            depth_range=None,
            load_sync_channel=load_sync_channel,
        )

        # Create ephyviewer source
        source_mic = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_nidq)

        # Create view and display only one channel to make things faster
        view_raw_mic = ephyviewer.TraceViewer(source=source_mic, name="raw mic")

        # Create spectrogram
        view_spec_mic = ephyviewer.SpectrogramViewer(
            source=source_mic, name="signals nidq"
        )
        view_spec_mic.params["colormap"] = "inferno"
        view_spec_mic.params.child("scalogram")["overlapratio"] = 0.2
        view_spec_mic.params.child("scalogram")["binsize"] = 0.02

        # Create smoothed amplitude view (important for song sorting)
        song_envelope = make_song_envelope(recording_nidq, channel_ids=[song_channel_name], kernel_size=envelope_kernel_size, min_freq_envelope=min_freq_envelope, max_freq_envelope=max_freq_envelope)

        source_smoothed_mic = ephyviewer.SpikeInterfaceRecordingSource(song_envelope)

        view_smoothed_mic = ephyviewer.TraceViewer(
            source=source_smoothed_mic, name="smoothed mic"
        )

        for c in range(recording_nidq.get_num_channels()):
            if c == 0:
                visible = True
            else:
                visible = False
            view_spec_mic.by_channel_params[f"ch{c}", "visible"] = visible
            view_raw_mic.by_channel_params[f"ch{c}", "visible"] = visible

        win.add_view(view_raw_mic)
        win.add_view(view_spec_mic, tabify_with="raw mic")
        win.add_view(view_smoothed_mic, tabify_with="raw mic")
    
    if camera:
        # if the camera data is a sequence of photos, you have to convert to video.mp4 first. Check "create_video_from_photos.py"!
        assert (spikeglx_folder/'video.mp4').exists(), 'video.mp4 not found!!!'       
        video_source = ephyviewer.MultiVideoFileSource([spikeglx_folder/'video.mp4'])
        view_camera = ephyviewer.VideoViewer(source=video_source, name='video')
        win.add_view(view_camera)

    return win