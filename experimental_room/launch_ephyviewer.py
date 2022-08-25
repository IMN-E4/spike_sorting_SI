import ephyviewer
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np


from ephyviewer.myqt import QT


def open_ephyviewer_mainwindow(spikeglx_folder):

    # Choose sources:
    ap = False
    raw_lfp = False
    mic = True
    filtered_lfp = False
    sorting = False

    print(spikeglx_folder)

    # App and viewer objects
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

    ### Sources
    if ap:   
        recording_spike = si.SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='imec0.ap') # ap file

        #~ locations = recording_spike.get_channel_locations()
        #~ print(locations)
        #~ order = np.argsort(locations[:, 1])
        #~ chan_ids_ordered = recording_spike.channel_ids[order]
        #~ print(order)
        #~ print(chan_ids_ordered)
        #~ recording_spike_ordered = recording_spike.channel_slice(chan_ids_ordered)        
        
        sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_spike)
        view0 = ephyviewer.TraceViewer(source=sig_source0, name='ap') # Trace of ap signal
        
        
        view0.params['scale_mode'] = 'same_for_all'
        for c in range(recording_spike.get_num_channels()):
            if c % 50 == 0:
                visible = True
            else:
                visible = False
            view0.by_channel_params[f'ch{c}', 'visible'] = visible
        
        win.add_view(view0)
            

    if raw_lfp:
        recording_lf = si.SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='imec0.lf') # lfp
        sig_source1 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # lfp

        view1 = ephyviewer.TraceViewer(source=sig_source1, name='signals lf') # Trace of LFP
        win.add_view(view1)
        
        # time-freq
        view1_tf = ephyviewer.TimeFreqViewer(source=sig_source1, name='timefreq') # Timefreq view of LFP
        win.add_view(view1_tf)

    if mic:
        recording_nidq = si.SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='nidq') # microphone
        #~ print(recording_nidq.get_sampling_frequency())
        #~ print(recording_nidq)
        sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_nidq) # microphone
        view2 = ephyviewer.SpectrogramViewer(source=sig_source2, name='signals nidq') # Trace of Microphone
        win.add_view(view2)
        
        view2.params['colormap'] = 'inferno'
        
        for c in range(recording_nidq.get_num_channels()):
            if c == 0:
                view2.by_channel_params[f'ch{c}', 'visible'] = True
            else:
                view2.by_channel_params[f'ch{c}', 'visible'] = False

    if filtered_lfp:
        recording_lf = si.SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='imec0.lf') # lfp
        recording_f = si.bandpass_filter(recording_lf, 50, 180)
        sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # filtered LFP trace
        view3 = ephyviewer.TraceViewer(source=sig_source3, name='signals flfp') # Trace of LFP filtered
        win.add_view(view3)
       
    #~ win.auto_scale()

    # Display
    win.show()
    app.exec_()



def select_folder_and_open():
    app = ephyviewer.mkQApp()
    dia = QT.QFileDialog(fileMode=QT.QFileDialog.Directory, acceptMode=QT.QFileDialog.AcceptOpen)
    dia.setViewMode(QT.QFileDialog.Detail)
    if dia.exec_():
        folder_names = dia.selectedFiles()
        folder_name = folder_names[0]
    else:
        return
    
    print(folder_name)
    
    open_ephyviewer_mainwindow(folder_name)


if __name__ == '__main__':
    spikeglx_folder = '/home/samuel/DataSpikeSorting/eduarda/raw_files/Imp_16_08_2022/Recordings/Rec_18_08_2022_g0/'
    open_ephyviewer_mainwindow(spikeglx_folder)
    
    #~ select_folder_and_open()
    
    