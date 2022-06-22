"""
This run a full spike sorting pipeline to explore:
  * diffrents sorter
  * diffrents params
  * diffrents pre processing pipeline

This also compute peaks.

This is the OLD pipeline started summer 2021

Author: Samuel, Eduarda
"""

# Packages
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.testing import verbose
import spikeinterface.full as si
# from probeinterface import read_spikeglx
import probeinterface as pi
from probeinterface.plotting import plot_probe
import numpy as np
import os
import glob
from spikeinterface.sortingcomponents.peak_detection import detect_peaks



# Paths
base_input_folder = Path('/home/admin/smb4k/NAS5802A5.LOCAL/Public/Neuropixel_Recordings/Impl_07_03_2022/Recordings/')

job_kwargs = dict(n_jobs=20,
                        chunk_memory='200M',
                        progress_bar=True,
                        )

def find_paths(main_dir, bird, **kwargs):
    """ Flexible way to find files in subdirectories based on keywords
    Parameters
    ----------
    main_dir: str
        Give the main directory where the subjects' folders are stored

    subject: str
        Give the name of the recording to be analyzed

    **kwargs: str
        Give keywords that will be used in the filtering of paths

    Examples
    -------
    Ex.1
    find_paths(main_dir='/home/arthur/Documents/SpikeSorting/',
               bird='Test_20210518/',
               key1='small')
    Returns
    -------
    updatedfilter: list
        List with path strings

    """

    # Check if arguments are in the correct type
    assert isinstance(main_dir, str), 'Argument must be str'
    assert isinstance(bird, str), 'Argument must be str'

    # Filtering step based on keywords
    firstfilter = glob.glob(main_dir + '/' + bird + '/**/*.imec0.ap.bin',
                            recursive=True)

    updatedfilter = firstfilter

    for _, value in kwargs.items():
        # Update list accoring to key value
        updatedfilter = list(filter(lambda path: value in path, updatedfilter))

    final_paths = [Path(path).parent for path in updatedfilter]

    return final_paths



# Get recordings
def get_recordings(path):
    """ Function to get the recordings and set the probe from base_input_folder
    
    """

    # recording = si.SpikeGLXRecordingExtractor(path, stream_id='imec0.ap')
    recording = si.read_spikeglx(path, stream_id='imec0.ap')
    print(recording)
    probe = recording.get_probe()
    print(probe)

    # meta_file = glob.glob(path.as_posix() + '/*.ap.meta')[0]
    # probe = pi.read_spikeglx(meta_file)
    # recording = recording.set_probe(probe)
    
    fs = recording.get_sampling_frequency()
    
    recordings = {}
    

    # ## You have to select the part of the recording by commenting the rest

    # whole recording
    #recordings['full'] = recording
    #recordings['full'].set_probe(probe)

    # # Chunk
    t0, t1 = (40000., 50000.) # Time in seconds
    frame0, frame1 = int(t0*fs), int(t1*fs)
    recordings['40000to50000'] = recording.frame_slice(frame0, frame1)
    recordings['40000to50000'].set_probe(probe) # there was an in place here that I removed
    

    return recordings

####
# Possible Preprocessings
####

def apply_preprocessing_1(rec):
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    return rec

def apply_preprocessing_2(rec):
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    rec = si.common_reference(rec, reference='local', local_radius=(50, 100))
    return rec


## You have to select the preprocessing you want by uncommenting
pre_processings = {
    # 'filter' : apply_preprocessing_1,
    'filter+cmr_radius' : apply_preprocessing_2,
}

## You have to select the sorters you want by uncommenting
sorter_names = [
    'tridesclous',
    # 'spykingcircus',
    # 'yass'
]


## You have to select the sorting params you want by uncommenting
sorters_params = {}

sorters_params['tridesclous'] = {
#     #~ 'default' : {},
    
    'custom_tdc_1' :  {
        'freq_min': 300.,
        'freq_max': 6000.,
        'detect_threshold' : 5,
        'common_ref_removal': False,
        'nested_params' : {
            'chunksize' : 30000,
            'preprocessor' : {'engine': 'opencl'},
            # 'preprocessor' : {'engine': 'numpy'},
            'peak_detector': {'adjacency_radius_um': 100},
            'clean_peaks': {'alien_value_threshold': 100.},
            'peak_sampler': {
                'mode': 'rand_by_channel',
                'nb_max_by_channel': 2000,
            }
        }
    }

}

# sorters_params['spykingcircus'] = {
#     #~ 'default' : {},
#     'custom_sc_1': {'detect_sign': -1,
#              'adjacency_radius': 100,
#              'detect_threshold': 6,
#              'template_width_ms': 3,
#              'filter': True,
#              'merge_spikes': True,
#              'auto_merge': 0.75,
#              'num_workers': None,
#              'whitening_max_elts': 1000,
#              'clustering_max_elts': 10000
#     }
# }

# sorters_params['yass'] = {
#     # 'default' : {},
#     'custom_yass_1': {'dtype': 'int16',
#                     'freq_min': 300,
#                     'freq_max': 0.3,
#                     'neural_nets_path': None,
#                     'multi_processing': 1,
#                     'n_processors': 20,
#                     'n_gpu_processors': 1,
#                     'n_sec_chunk': 10,
#                     'n_sec_chunk_gpu_detect': 0.5,
#                     'n_sec_chunk_gpu_deconv': 5,
#                     'gpu_id': 0,
#                     'generate_phy': 0,
#                     'phy_percent_spikes': 0.05,
#                     'spatial_radius': 70,
#                     'spike_size_ms': 5,
#                     'clustering_chunk': [0, 300],
#                     'update_templates': 0,
#                     'neuron_discover': 0,
#                     'template_update_time': 300,
#                     'total_memory': '500M',
#                     'n_jobs_bin': 15}
# }


# ## You have to select the respective docker images you want by uncommenting
# docker_images = {}
# docker_images['tridesclous'] = 'spikeinterface/tridesclous-base:1.6.4'
# # docker_images['spykingcircus'] = 'spikeinterface/spyking-circus-base:1.0.7'
# docker_images['yass'] = 'spikeinterface/yass-base:2.0.0'

## Main function to run sorters in a nested way
def run_all_sorters(path):
    """
    This run all in nested loop:
      * recording
      
      
    """
    recordings = get_recordings(path)
    
    for rec_name, rec in recordings.items():
        print(rec_name, rec)
        for preprocess_name, func in pre_processings.items():
            print('  ', preprocess_name)
            
            rec_processed = func(rec)
            rec_preproc_folder = out_path / rec_name / preprocess_name / 'preproc_rec'
            rec_saved = rec_processed.save(folder=rec_preproc_folder, **job_kwargs)

            for sorter_name in sorter_names:
                print('    ', sorter_name)

                for param_name, sparams in sorters_params[sorter_name].items():
                    print('      ', param_name)

                    # docker_image = docker_images[sorter_name]
                    # print(docker_image)

                    output_folder = out_path / rec_name / preprocess_name / sorter_name / param_name
                    # os.makedirs(output_folder, exist_ok=True) ## put this because docker usage throws path error
                    print(output_folder)
                    
                    ## Uncomment this to skip folders that already exist
                    # if output_folder.is_dir():
                    #     print('Already exists, so skip it', output_folder)
                    #     continue
                    

                    
                    # No docker
                    print('Starting to run sorter without docker')
                    sorting = si.run_sorter(sorter_name, rec_saved,
                        output_folder=output_folder, verbose=True, 
                        raise_error=True,
                        **sparams)

                    # With docker
                    # print('Starting to run sorter with docker')
                    # sorting = si.run_sorter_container(sorter_name, recording=rec_processed, mode='docker', 
                    #                         container_image=docker_image, verbose=True, raise_error=True, **sparams)

                    print(sorting.get_num_units())

def run_all_post_processing(path):
    recordings = get_recordings(path)
    
    for rec_name, rec in recordings.items():
        #~ print(rec_name, rec)
        
        for preprocess_name, func in pre_processings.items():
            #~ print('  ', preprocess_name)
            
            rec_preproc_folder = out_path / rec_name / preprocess_name / 'preproc_rec'
            rec_processed = si.load_extractor(rec_preproc_folder)

            for sorter_name in sorter_names:
                #~ print('    ', sorter_name)
                
                for param_name, sparams in sorters_params[sorter_name].items():
                    #~ print('      ', param_name)
                    
                    output_folder = out_path / rec_name / preprocess_name / sorter_name / param_name
                    print(output_folder)
                    
                    if sorter_name == 'tridesclous':
                        sorting = si.TridesclousSortingExtractor(output_folder)
                    elif sorter_name == 'spykingcircus':
                        sorting = si.SpykingCircusSortingExtractor(output_folder)
                    elif sorter_name == 'yass':
                        sorting = si.YassSortingExtractor(output_folder) 

                    print(sorting)

                    # extractor waveforms

                    print(rec_processed.get_probe())
                    # si.plot_probe_map(rec_processed, channel_ids=rec_processed.channel_ids[0:4])
                    wf_folder = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_waveforms')
                    # if wf_folder.is_dir():
                    #     print('Already exists', wf_folder)
                    #     continue
                    
                    print('Starting to run extract_waveforms')
                    we = si.extract_waveforms(rec_processed, sorting, folder=wf_folder,
                                load_if_exists=True, ms_before=1., ms_after=2.,
                                max_spikes_per_unit=500,
                                chunk_size=30000, n_jobs=30, relative_path=True, progress_bar=True)

                    if sorting.unit_ids.size == 0:
                        continue
                    
                    print('Starting to run compute_spike_amplitudes')
                    si.compute_spike_amplitudes(we, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=30,
                    return_scaled=True, load_if_exists=True) 

                    # compute PCs
                    print('Starting to run compute_principal_components and calculate_pc_metrics')
                    pc = si.compute_principal_components(we, load_if_exists=True,
                                n_components=3, mode='by_channel_local')

                    # pc_metrics = si.calculate_pc_metrics(pc, metric_names=['nearest_neighbor'])
                    
                    # compute metrics
                    print('Starting to run compute_quality_metrics')
                    metrics_list = ['snr', 'isi_violation', 'num_spikes', 'firing_rate', 'presence_ratio']
                    print(metrics_list)
                    metrics = si.compute_quality_metrics(we, load_if_exists=False, metric_names=metrics_list)

                    # https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

                    # https://spikeinterface.readthedocs.io/en/latest/modules/toolkit/plot_4_curation.html
                    
                    # export report
                    print('Starting to run export_report')
                    report_folder = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_report')
                    si.export_report(we, report_folder, remove_if_exists=True,
                            chunk_size=30000, n_jobs=30, progress_bar=True, metrics=metrics, verbose=True)
                    
                    # save to npz
                    output_sorting_folder = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_sorting_uncurated')
                    if output_sorting_folder.exists():
                        continue
                    print('Starting to run save_to_folder')
                    sorting.save_to_folder(output_sorting_folder)




def open_one_sorting():
    sub_path = '/media/storage/spikesorting_output/sorting_pipeline_out_29092021_try/rest/filter/tridesclous/custom_tdc_1/'
    
    folder = out_path / sub_path
    print(folder)
     
    sorting_tdc = si.TridesclousSortingExtractor(folder)
    print(sorting_tdc.get_num_units())

    # sorting_sc = si.SpykingCircusSortingExtractor(folder)
    # print(sorting_sc.get_num_units())
    

def check_probe_channels(rec):
    probe = rec.get_probe()
    plot_probe(probe, show_channel_on_click=True)
    plt.show()

def dirty_sorter(path, path_to_peaks=None, save=True):
    recordings = get_recordings(path)
    
    for rec_name, rec in recordings.items():
        print(rec_name, rec)
        for preprocess_name, func in pre_processings.items():
            print('  ', preprocess_name)
                
            output_folder = out_path / rec_name / preprocess_name
            print(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            
            if path_to_peaks==None:

                job_kwargs = dict(n_jobs=30,
                                  chunk_memory='10M',
                                  progress_bar=True,
                                )
                rec_preprocessed = func(rec)
                noise_levels_scaled = si.get_noise_levels(rec_preprocessed, return_scaled=True)
                peaks = detect_peaks(rec_preprocessed, method='locally_exclusive', local_radius_um=50,
                        peak_sign='neg', detect_threshold=5, n_shifts=5,
                        noise_levels=noise_levels_scaled, **job_kwargs)
                np.save(output_folder / 'peaks.npy', peaks)
                print('peaks saved at', output_folder)
            else: 
                peaks = np.load(Path(path_to_peaks) / 'peaks.npy')
            
            sorting = si.NumpySorting.from_times_labels(peaks['sample_ind'], peaks['channel_ind'], rec_preprocessed.get_sampling_frequency()) 
            print(sorting)

            if save==True:
                sorting.save(folder = output_folder / 'multiunit')
                print('sorting saved at', output_folder)


if __name__ == '__main__':

    for path in find_paths(main_dir=base_input_folder.as_posix(), bird='Rec_7_13_03_2022_g0/'):
        print('Working on: ', path)
        global out_path
        out_path = Path(path.parent / 'sorting_20220504')
        print(out_path)

        # get_recordings(path)

        run_all_sorters(path)
        
        run_all_post_processing(path)
        # dirty_sorter(path, path_to_peaks=None,  save=True)
        
    # open_one_sorting()
        
