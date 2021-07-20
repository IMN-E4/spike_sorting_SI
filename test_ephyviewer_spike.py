import ephyviewer

import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np


out_path = Path('/media/storage/spikesorting_output/sorting_pieline_out/')
    
    
sub_path = 'sing/filter+cmr_radius/spykingcircus/custum_sc_1/'

folder = out_path / sub_path
print(folder)

sorting_sc = si.SpykingCircusSortingExtractor(folder)
print(sorting_sc)

spike_source = ephyviewer.FromSpikeinterfaceSorintgSource(sorting_sc)


app = ephyviewer.mkQApp()
win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

view1 = ephyviewer.SpikeTrainViewer(source=spike_source)


#put this veiwer in the main window
win.add_view(view1)



win.show()
app.exec_()

