import ephyviewer
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np
from ephyviewer.myqt import QT
from ephyviewer.tools import ParamDialog
import json
from path_handling import get_sorting_folder, get_spikeglx_folder, get_working_folder, get_synchro_file


def open_ephyviewer_mainwindow(key_tuple,
                               sorter_name,     
                                ap = False,
                                raw_lfp = False,
                                mic = True,
                                filtered_lfp = False,
                                sorting_panel=False):
    
    implant_name, rec_name, time_range, depth_range, _, time_stamp = key_tuple
    spikeglx_folder = get_spikeglx_folder(implant_name, rec_name)
    sorting_folder = get_sorting_folder(implant_name, rec_name, time_range, depth_range, time_stamp, sorter_name)
    working_dir = get_working_folder(spikeglx_folder,
                                     rec_name,
                                     time_range,
                                     depth_range,
                                     time_stamp)

    synchro_file = get_synchro_file(implant_name, rec_name, time_range, time_stamp)
    with open(synchro_file) as f:
        synchro = json.load(f)

    # App and viewer objects
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

    ### Sources
    if ap:   
        recording_spike = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap') # ap file        
        sig_source0 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_spike)
        sig_source0.sample_rate = sig_source0.sample_rate / synchro['a']
        sig_source0._t_start = sig_source0._t_start + synchro['b']
        

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
        recording_lf = si.read_spikeglx(spikeglx_folder, stream_id='imec0.lf') # lfp
        sig_source1 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_lf) # lfp
        sig_source1.sample_rate = sig_source1.sample_rate / synchro['a']
        sig_source1._t_start = sig_source1._t_start + synchro['b']

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
        recording_nidq = si.read_spikeglx(spikeglx_folder, stream_id='nidq') # microphone
        sig_source2 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_nidq) # microphone
        view2 = ephyviewer.SpectrogramViewer(source=sig_source2, name='signals nidq') # Trace of Microphone
        win.add_view(view2)
        
        view2.params['colormap'] = 'inferno'
        view2.params.child('scalogram')['overlapratio'] = 0.2
        view2.params.child('scalogram')['binsize'] = 0.02
        
        
        for c in range(recording_nidq.get_num_channels()):
            if c == 0:
                view2.by_channel_params[f'ch{c}', 'visible'] = True
                view2.by_channel_params[f'ch{c}', 'clim_min'] = -30
                view2.by_channel_params[f'ch{c}', 'clim_max'] = 60
            else:
                view2.by_channel_params[f'ch{c}', 'visible'] = False
      
    if filtered_lfp:
        recording_lf = si.read_spikeglx(spikeglx_folder, stream_id='imec0.lf') # lfp
        recording_f = si.bandpass_filter(recording_lf, 50, 180)
        sig_source3 = ephyviewer.SpikeInterfaceRecordingSource(recording=recording_f) # filtered LFP trace
        sig_source3.sample_rate = sig_source3.sample_rate / synchro['a']
        sig_source3._t_start = sig_source3._t_start + synchro['b']
        view3 = ephyviewer.TraceViewer(source=sig_source3, name='signals filtered lfp') # Trace of LFP filtered
        win.add_view(view3)
       
    if sorting_panel:
        sorting = si.read_npz_sorting(sorting_folder / 'sorting_cached.npz')

        t_start = synchro['b']
        sr = recording_spike.get_sampling_frequency() / synchro['a']
        all_spikes = []
        for unit_id in sorting.unit_ids:
            spike_times = sorting.get_unit_spike_train(unit_id=unit_id) / sr + t_start
            all_spikes.append({ 'time':spike_times, 'name':f'Unit#{unit_id}' })
        spike_source = ephyviewer.InMemorySpikeSource(all_spikes)

        view4 = ephyviewer.SpikeTrainViewer(source=spike_source, name='sorting') # spiking traces
        win.add_view(view4)    

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

def test_open_data():
    key_tuple = ('Anesth_21_01_2023', 'Rec_21_01_2023_1_g0', None, None, False, "2023-01")
    sorter_name = 'kilosort2_5'
    open_ephyviewer_mainwindow(key_tuple, 
                               sorter_name,
                               ap=True,
                               raw_lfp=False,
                               mic=True,
                               filtered_lfp=False,
                               sorting_panel=True                               
                               )

if __name__ == '__main__':
    # select_folder_and_open()
    test_open_data()
    
    
