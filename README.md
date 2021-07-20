# spike_sorting_with_samuel

## How to use the spike sorting pipeline

    1. Check for drift using check_drift.ipynb (corrections not available yet) -- we are computing the peaks in advance using compute_peaks.py

    2. Use the spike_sorting_pipeline.py to run the spike sorting in different files/sorting params

    3. We will update the jupyters SCvsTDC.ipynb for a further comparison/visual inspection between different sorters.

    Additional: script compute_tridesclous_several_times.py was used to check TDC stability. You can use it to check sorting stability again.






## How to use EphyViewer
    1. You can use test_ephyviewer*.py to play with it. We are still building the standard set of panels we would like to have, but with this tool you can code/adapt to different needs.
