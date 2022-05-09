import ephyviewer

import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np



# Paths
base_folder = Path('/home/admin/smb4k/NAS5802A5.LOCAL/Public/Neuropixel_Recordings/Impl_07_03_2022/Recordings/')
data_folder = base_folder / 'Rec_5_11_03_2022_g0'

print(data_folder)

## sorting_folder = Path('/media/storage/spikesorting_output/sorting_pipeline_out_29092021_try/')
#sorting_sub_path = 'sorting_20220420/full/filter+cmr_radius/tridesclous/custom_tdc_1/'
## Folders
#folder = data_folder / sorting_sub_path
#print(folder)


# Main recording files
recording_spike = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.ap') # ap file
#print(recording)

# recording_lf = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.lf') # lfp
# # #print(recording)

recording_nidq = si.SpikeGLXRecordingExtractor(data_folder, stream_id='nidq') # microphone
print(recording_nidq.get_sampling_frequency())
#print(recording_nidq)


# # Probe
# probe = read_spikeglx(data_folder / 'raw_awake_01_g0_t0.imec0.ap.meta')
# # probe = recording_spike.get_probe()
# #print(probe)
# recording_spike = recording_spike.set_probe(probe)
# #print(recording)
# #exit()


# # Sorting
#sorting = si.SpykingCircusSortingExtractor(folder) # SpykingCircusSortingExtractor
#sorting = si.TridesclousSortingExtractor(folder)    # TridesclousSortingExtractor
##print(sorting)


## # Spike source
#spike_source = ephyviewer.SpikeInterfaceSortingSource(sorting)


# # Filtering
# recording_f = si.bandpass_filter(recording_lf, 50, 180)


# # # Sources
sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_spike) # original ap
# sig_source1 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # lfp
sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_nidq) # microphone
# sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_f) # filtered LFP trace


# App and viewer objects
app = ephyviewer.mkQApp()
win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

view0 = ephyviewer.TraceViewer(source=sig_source0, name='signals') # Trace of ap signal
win.add_view(view0)

# view1 = ephyviewer.TraceViewer(source=sig_source1, name='signals lf') # Trace of LFP
# win.add_view(view1)

# view1_tf = ephyviewer.TimeFreqViewer(source=sig_source1, name='timefreq') # Timefreq view of LFP
# win.add_view(view1_tf)

view2 = ephyviewer.SpectrogramViewer(source=sig_source2, name='signals nidq') # Trace of Microphone
win.add_view(view2)

# view2_raw = ephyviewer.TraceViewer(source=sig_source2, name='signals ndq_raw') # Trace of Microphone
# win.add_view(view2_raw)

# view3 = ephyviewer.TraceViewer(source=sig_source3, name='signals F') # Trace of LFP filtered
# win.add_view(view3)

#view4 = ephyviewer.SpikeTrainViewer(source=spike_source) # spiking traces
#win.add_view(view4)


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

