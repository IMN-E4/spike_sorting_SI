import ephyviewer
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np



# Choose sources:
ap = False
raw_lfp = False
mic = True
filtered_lfp = False
sorting = False

# Paths
base_folder = Path('/home/analysis_user/smb4k/NAS5802A5.LOCAL/Public/Neuropixel_Recordings/AreaX-LMAN/Imp_10_11_2021/Recordings/')
data_folder = base_folder / 'Rec_2_19_11_2021_g0'

print(data_folder)

sorting_sub_path = 'sorting_20220420/full/filter+cmr_radius/tridesclous/custom_tdc_1/'

## Folders
sorting_folder = data_folder / sorting_sub_path
print(sorting_folder)


# App and viewer objects
app = ephyviewer.mkQApp()
win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

### Sources
if ap:   
    recording_spike = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.ap') # ap file
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
    print(recording_nidq.get_sampling_frequency())
    print(recording_nidq)
    sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_nidq) # microphone
    view2 = ephyviewer.SpectrogramViewer(source=sig_source2, name='signals nidq') # Trace of Microphone
    win.add_view(view2)

if filtered_lfp:
    recording_lf = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.lf') # lfp
    recording_f = si.bandpass_filter(recording_lf, 50, 180)
    sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # filtered LFP trace
    view3 = ephyviewer.TraceViewer(source=sig_source3, name='signals flfp') # Trace of LFP filtered
    win.add_view(view3)
   

if sorting:
    # choose which by commenting!
    sorting = si.TridesclousSortingExtractor(sorting_folder)    # TridesclousSortingExtractor
    #sorting = si.SpykingCircusSortingExtractor(folder) # SpykingCircusSortingExtractor
    print(sorting)

    spike_source = ephyviewer.SpikeInterfaceSortingSource(sorting)

    view4 = ephyviewer.SpikeTrainViewer(source=spike_source, name='sorting') # spiking traces
    win.add_view(view4)    


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

