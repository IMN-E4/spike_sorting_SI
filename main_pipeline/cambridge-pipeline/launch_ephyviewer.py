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
# Updates          #
####################
#1. Do we have to do any probe attaching here?

####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
from pathlib import Path

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import spikeinterface.full as si
import ephyviewer

# Internal imports ### (Put here imports that are related to internal codes from the lab)
from utils import add_probe_to_rec


# Params
block_index = 0
bandpass_range = (1,300)
order_by_depth = True  # probe tip to probe upper part

# Choose sources:
raw_recording = True
mic_spectrogram = False
bandpassed_recording = False
sorting = False

# Paths
base_folder = Path('/nas/Cambridge_Recordings/Test_Data_troubleshoot/')
data_folder = base_folder / '2023-08-23_15-56-05/Record Node 101/'

print(data_folder)

# sorting_sub_path = 'sorting_20220420/full/filter+cmr_radius/tridesclous/custom_tdc_1/'
#
# ## Folders
# sorting_folder = data_folder / sorting_sub_path
# print(sorting_folder)


## App and viewer objects
app = ephyviewer.mkQApp()
win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

### Sources
if raw_recording:   
    recording = si.read_openephys(data_folder, block_index=block_index) 
    if order_by_depth:
        recording = add_probe_to_rec(recording)
        recording = si.depth_order(recording)
    sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording)
    view0 = ephyviewer.TraceViewer(source=sig_source0, name='recording traces')
    win.add_view(view0)
    
if mic_spectrogram:
    recording = si.read_openephys(data_folder, block_index=block_index)
    mic = recording.channel_slice(channel_ids=['ADC1'])
    sig_source1 = ephyviewer.SpikeInterfaceRecordingSource(recording=mic) 

    view1 = ephyviewer.SpectrogramViewer(source=sig_source1, name='mic spectrogram')
    win.add_view(view1)


if bandpassed_recording:
    recording = si.read_openephys(data_folder, block_index=block_index)
    if order_by_depth:
        recording = add_probe_to_rec(recording)
        recording = si.depth_order(recording)
    filtered_recording = si.bandpass_filter(recording, bandpass_range[0], bandpass_range[1])
    sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=filtered_recording)
    view2 = ephyviewer.TraceViewer(source=sig_source2, name='signals bandpassed')
    win.add_view(view2)

    view3 = ephyviewer.TimeFreqViewer(source=sig_source2, name='timefreq sig bandpassed')
    win.add_view(view3)

# if sorting:
#     # choose which by commenting!
#     sorting = si.TridesclousSortingExtractor(sorting_folder)    # TridesclousSortingExtractor
#     #sorting = si.SpykingCircusSortingExtractor(folder) # SpykingCircusSortingExtractor
#     print(sorting)

#     spike_source = ephyviewer.SpikeInterfaceSortingSource(sorting)

#     view4 = ephyviewer.SpikeTrainViewer(source=spike_source, name='sorting') # spiking traces
#     win.add_view(view4)    


## Display only 2 channels
#for channel in range(recording_spike.get_num_channels()):
   #for view in (view0, view2):
        #if channel <3:
            #view.by_channel_params[f'ch{channel}', 'visible'] = True
        #else:
            #view.by_channel_params[f'ch{channel}', 'visible'] = False


# Display
win.show()
app.exec_()

