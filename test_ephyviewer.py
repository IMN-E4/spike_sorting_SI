import ephyviewer

import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np

base_folder = Path('/home/arthur/Documents/SpikeSorting/Test_20210518/') 

data_folder = base_folder / 'raw_awake'


recording_spike = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.ap')
#~ print(recording)

#~ recording_lf = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.lf')
#~ print(recording)

#~ recording_nidq = si.SpikeGLXRecordingExtractor(data_folder, stream_id='nidq')
#~ print(recording_nidq)


#~ probe = read_spikeglx(data_folder / 'raw_awake_01_g0_t0.imec0.ap.meta')
#~ print(probe)
#~ recording = recording.set_probe(probe)
#~ print(recording)

#~ exit()

recording_f = si.bandpass_filter(recording_spike)


sig_source0 = ephyviewer.FromSpikeinterfaceRecordingSource(recording=recording_spike)
sig_source4 = ephyviewer.FromSpikeinterfaceRecordingSource(recording=recording_f)
#~ sig_source1 = ephyviewer.FromSpikeinterfaceRecordingSource(recording=recording_lf)
#~ sig_source2 = ephyviewer.FromSpikeinterfaceRecordingSource(recording=recording_nidq)

app = ephyviewer.mkQApp()
win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

view0 = ephyviewer.TraceViewer(source=sig_source0, name='signals')
win.add_view(view0)

view5 = ephyviewer.TraceViewer(source=sig_source4, name='signals F')
win.add_view(view5 )




#~ view1 = ephyviewer.TraceViewer(source=sig_source1, name='signals lf')
#~ win.add_view(view1)

#~ view2 = ephyviewer.TraceViewer(source=sig_source2, name='signals nidq')
#~ win.add_view(view2)

#~ view4 = ephyviewer.TimeFreqViewer(source=sig_source1, name='timefreq')
#~ win.add_view(view4)


for c in range(recording_spike.get_num_channels()):
   for view in (view0, view5):
        if c <3:
            view.by_channel_params[f'ch{c}', 'visible'] = True
        else:
            view.by_channel_params[f'ch{c}', 'visible'] = False




win.show()
app.exec_()

