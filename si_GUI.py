import spikeinterface.full as si
# recording = si.read_XXXX('/path/to/my/recording')
# recording_filtered = si.bandpass_filter(recording)
# sorting = si.run_sorter('YYYYY', recording_filtered)
waveform_forlder = '/media/storage/spikesorting_output/sorting_pipeline_out_29092021_try/rest/filter+cmr_radius/tridesclous/custom_tdc_1_waveforms/'
# we = si.extract_waveforms(
#     recording_filtered, sorting, waveform_folder,
#     max_spikes_per_unit=500,
#     ms_before=1.5, ms_after=2.5,
#     n_jobs=10, total_memory='500M',
#     progress_bar=True,
# )
# and optionally compute principal component
# pc = compute_principal_components(we,
#     n_components=5,
#     mode='by_channel_local',
#     whiten=True)

import spikeinterface_gui
# This cerate a Qt app
app = spikeinterface_gui.mkQApp() 
# reload the waveform folder
we = si.WaveformExtractor.load_from_folder(waveform_forlder)
# create the mainwindow and show
win = spikeinterface_gui.MainWindow(we, verbose=True)
win.show()
# run the main Qt6 loop
app.exec_()
