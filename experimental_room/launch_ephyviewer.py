import ephyviewer
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np
from ephyviewer.myqt import QT
from ephyviewer.tools import ParamDialog


def open_ephyviewer_mainwindow(spikeglx_folder,     
    ap = False,
    raw_lfp = False,
    mic = True,
    filtered_lfp = False,):

    # App and viewer objects
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

    ### Sources
    if ap:   
        recording_spike = si.SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='imec0.ap') # ap file        
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
        
        
        view1.params['scale_mode'] = 'same_for_all'
        for c in range(recording_lf.get_num_channels()):
            if c % 50 == 0:
                visible = True
            else:
                visible = False
            view1.by_channel_params[f'ch{c}', 'visible'] = visible

    if mic:
        recording_nidq = si.SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='nidq') # microphone
        sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_nidq) # microphone
        view2 = ephyviewer.SpectrogramViewer(source=sig_source2, name='signals nidq') # Trace of Microphone
        win.add_view(view2)
        
        view2.params['colormap'] = 'inferno'
        view2.params.scalogram['overlapratio'] = 0.2
        view2.params.scalogram['binsize'] = 0.02
        
        
        for c in range(recording_nidq.get_num_channels()):
            if c == 0:
                view2.by_channel_params[f'ch{c}', 'visible'] = True
                view2.by_channel_params[f'ch{c}', 'clim_min'] = -30
                view2.by_channel_params[f'ch{c}', 'clim_max'] = 60
            else:
                view2.by_channel_params[f'ch{c}', 'visible'] = False
      
    if filtered_lfp:
        recording_lf = si.SpikeGLXRecordingExtractor(spikeglx_folder, stream_id='imec0.lf') # lfp
        recording_f = si.bandpass_filter(recording_lf, 50, 180)
        sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # filtered LFP trace
        view3 = ephyviewer.TraceViewer(source=sig_source3, name='signals flfp') # Trace of LFP filtered
        win.add_view(view3)
       
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
    
    


    params = [
        {'name': 'mic', 'type': 'bool', 'value': True},
        {'name': 'ap', 'type': 'bool', 'value': False},
        {'name': 'raw_lfp', 'type': 'bool', 'value': False},
        {'name': 'filtered_lfp', 'type': 'bool', 'value': False},
    ]
    dia = ParamDialog(params, title='Select streams')
    if dia.exec_():
        kwargs_streams = dia.get()
        
    else:
        return
    
    print(kwargs_streams)
    print(folder_name)
    
    
    open_ephyviewer_mainwindow(folder_name, **kwargs_streams)


if __name__ == '__main__':
    select_folder_and_open()
    
    