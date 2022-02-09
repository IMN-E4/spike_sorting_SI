# spike_sorting_with_samuel

## How to use the spike sorting pipeline

    1. Check for drift using check_drift.ipynb (corrections not available yet) -- we are computing the peaks in advance using compute_peaks.py [This possibly will be updated to motion notebooks]

    2. Check peaks and noise in probe to have an idea (noise_peaks_probe.py)

    3. Use the spike_sorting_pipeline.py to run the spike sorting in different files/sorting params.
        3.1. Arthur requested that we create also a 'dirty sorter' which gets 'multiunits'. e.g. one channel == one unit. This is part of the sorting pipeline.

    4. We will update the jupyters SCvsTDC.ipynb for a further comparison/visual inspection between different sorters.

    Additional: script compute_tridesclous_several_times.py was used to check TDC stability. You can use it to check sorting stability again.


## How to use EphyViewer
    1. You can use test_ephyviewer*.py to play with it. We are still building the standard set of panels we would like to have, but with this tool you can code/adapt to different needs.


## How to use SpikeInterface GUI
    1. Use script si_GUI.py
