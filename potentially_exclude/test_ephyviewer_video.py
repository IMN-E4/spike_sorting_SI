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
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
from pathlib import Path

# Third party imports ### (Put here third-party libraries e.g. pandas, numpy)
import spikeinterface.full as si
import ephyviewer
import numpy as np

# Internal imports ### (Put here imports that are related to internal codes from the lab)


# Choose sources:
ap = True
raw_lfp = False
mic = False
filtered_lfp = False
sorting = False
camera = False

# Paths
base_folder = Path('/nas/Neuropixel2_Recordings/Pilot_Data_XLMAN/Imp_20240410/Recordings/')
data_folder = base_folder / '20240410_LMANmidshanksXextshanks_e1_g0'
camera_file =  'video.mp4'


print(data_folder)

# sorting_sub_path = 'sorting_20220420/full/filter+cmr_radius/tridesclous/custom_tdc_1/'
#
# ## Folders
# sorting_folder = data_folder / sorting_sub_path
# print(sorting_folder)


# App and viewer objects
app = ephyviewer.mkQApp()
win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

### Sources
if ap:   
    recording_spike = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.ap') # ap file
    recording_spike = recording_spike.channel_slice(channel_ids=['imec0.ap#AP191', 'imec0.ap#AP190', 'imec0.ap#AP189'])
    
    recording_spike = si.bandpass_filter(recording_spike, freq_min=300, freq_max=6000)

    sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_spike) # spike trains
    view0 = ephyviewer.TraceViewer(source=sig_source0, name='ap') # Trace of ap signal
    win.add_view(view0)
    
if raw_lfp:
    recording_lf = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.lf') # lfp
    sig_source1 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # lfp

    view1 = ephyviewer.TraceViewer(source=sig_source1, name='signals lf') # Trace of LFP
    win.add_view(view1)
    
    # time-freq
    view1_tf = ephyviewer.TimeFreqViewer(source=sig_source1, name='timefreq') # Timefreq view of LFP
    win.add_view(view1_tf)

if mic:
    recording_nidq = si.SpikeGLXRecordingExtractor(data_folder, stream_id='nidq') # microphone
    sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_nidq) # microphone
    view2 = ephyviewer.SpectrogramViewer(source=sig_source2, name='signals nidq') # Trace of Microphone
    win.add_view(view2)

if camera:
    n_secs_bet_photos = 5   
    video_source = ephyviewer.MultiVideoFileSource([data_folder/camera_file])
    # print(video_source.nb_frames)
    video_times = np.arange(video_source.nb_frames[0])*n_secs_bet_photos
    view3 = ephyviewer.VideoViewer.from_filenames([data_folder/camera_file], video_times=[video_times], name='video')
    win.add_view(view3)

# if filtered_lfp:
#     recording_lf = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.lf') # lfp
#     recording_f = si.bandpass_filter(recording_lf, 50, 180)
#     sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # filtered LFP trace
#     view3 = ephyviewer.TraceViewer(source=sig_source3, name='signals flfp') # Trace of LFP filtered
#     win.add_view(view3)
   

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

