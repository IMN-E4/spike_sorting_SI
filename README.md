# Authors: Eduarda Centeno and Samuel Garcia
General support: contact Sam for support

## How to use the spike sorting pipeline and related visualization tools

**Key point: it's important to know that the tool expects the data to be organized and named in a specific way (check data management plan in NAS)! Pay attention: there are hard-coded paths in the different script!**

    
We have pipelines for two systems in main_pipeline
        
### Neuropixel / SpikeGLX
- **Spikes Sorting**:
The main script here is spike_sorting_pipeline_NP.py. This file depends on the adjacent Python files:

        params_NP: here we define all the parameters for preprocessing, sorting, and postprocessing. This includs paths to input and ouput data!
        recording_list_NP: here we define the recordings to be processed
        myfigures: here we have supporting functions to create probe vs peaks/noise plots in the preprocessing step
        path_handling: here we keep all functions necessary for manipulating and concatenating paths
        utils: here we have functions to manipulate the recording object (read/slice/apply preprocess)
    
    Before running this script, you have to select with booleans which steps of the chain you want to run. For new sortings, we recommend:
        pre_check = True
        sorting = True
        postproc = True
        compute_alignment = True
        time_stamp = "default"
        
    *Important point 1*: you can also consider running only the pre_check to first visualize the data drift and amount of peaks to decide if it's worth proceeding with the sorting at all.
    
    *Important point 2*: light clean sorting will by default will be saved in NAS, while heavy/complete cache in local disk (necessary for using SpikeInterface GUI). You can define this in params_NP.py.

- **Data visualization**:
 The main script here is mainwindow.py. This file depends on the adjacent Python files:

         launch_ephyviewer: here we have a chain of EphyViewer viewers for our data types (lf, ap, nidq)
         params_viz: this file contais the base_folder for data fetching
         path_handling_viz: here we keep all functions necessary for manipulating and concatenating paths
         utils: here we have functions to manipulate the recording object (read/slice) and find data in NAS
         song_envelope.py: class to create envelope of microphone data.


    We also have a NP_VIZ.sh file to work with an executable and facilitate user experience.


### Cambridge Neurotech / OpenEphys
Sam and Eduarda worked temporarily to create a sorting pipeline for Cambridge Neurotech data; however, developments stopped due to discontinuation of data collection with this setup. 