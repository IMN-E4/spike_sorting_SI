"""
This run a full spike sorting pipeline to explore:
  * diffrents sorter
  * diffrents params
  * diffrents pre processing pipeline

This also compute peaks.


Author: Samuel, Eduarda
"""

import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np




base_input_folder = Path('/home/arthur/Documents/SpikeSorting/Test_20210518/') 


out_path = Path('/media/storage/spikesorting_output/sorting_pieline_out/')



def get_recordings():
    

    data_folder = base_input_folder / 'raw_awake'

    out_path = Path('/media/storage/spikesorting_output/REST/')

    recording = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.ap')
    probe = read_spikeglx(data_folder / 'raw_awake_01_g0_t0.imec0.ap.meta')
    recording = recording.set_probe(probe)
    
    
    fs = recording.get_sampling_frequency()
    
    
    recordings = {}
    
    # full
    #~ full_recording = recording
    #~ recordings['full'] = recording
    
    # rest 
    #~ t0, t1 = (2000., 2500.) # Time in seconds
    #~ frame0, frame1 = int(t0*fs), int(t1*fs)
    #~ recordings['rest'] = recording.frame_slice(frame0, frame1)
    #~ recordings['rest'].set_probe(probe, in_place=True) #  this is due to a SI bug will be removed
    
    t0, t1 = 1000, 1100
    frame0, frame1 = int(t0*fs), int(t1*fs)    
    recordings['sing'] = recording.frame_slice(frame0, frame1)
    recordings['sing'].set_probe(probe, in_place=True)  #  this is due to a SI bug will be removed
    

    return recordings

####
# several pre processing
####

def apply_preprocessing_1(rec):
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    return rec

def apply_preprocessing_2(rec):
    rec = si.bandpass_filter(rec, freq_min=300, freq_max=6000)
    rec = si.common_reference(rec, reference='local', local_radius=(50, 100))
    return rec
    

pre_processings = {
    #~ 'filter' : apply_preprocessing_1,
    'filter+cmr_radius' : apply_preprocessing_2,
}


sorter_names = [
    #~ 'tridesclous',
    'spykingcircus'

]

sorters_params = {}

sorters_params['tridesclous'] = {
    #~ 'default' : {},
    
    'custum_tdc_1' :  {
        'freq_min': 300.,
        'freq_max': 6000.,
        'detect_threshold' : 5,
        'common_ref_removal': False,
        'nested_params' : {
            'peak_detector': {'adjacency_radius_um': 100},
            'clean_peaks': {'alien_value_threshold': 100.},
            'peak_sampler': {
                'mode': 'rand_by_channel',
                'nb_max_by_channel': 2000,
            }
        }
    }

}

sorters_params['spykingcircus'] = {
    #~ 'default' : {},
    'custum_sc_1': {'detect_sign': -1,
             'adjacency_radius': 100,
             'detect_threshold': 6,
             'template_width_ms': 3,
             'filter': True,
             'merge_spikes': True,
             'auto_merge': 0.75,
             'num_workers': None,
             'whitening_max_elts': 1000,
             'clustering_max_elts': 10000
    }
}




def run_all_sorters():
    """
    This run all in nested loop:
      * recording
      
      
    """
    recordings = get_recordings()
    
    for rec_name, rec in recordings.items():
        #~ print(rec_name, rec)
        
        for preprocess_name, func in pre_processings.items():
            #~ print('  ', preprocess_name)
            
            for sorter_name in sorter_names:
                #~ print('    ', sorter_name)
                
                for param_name, sparams in sorters_params[sorter_name].items():
                    #~ print('      ', param_name)
                    
                    output_folder = out_path / rec_name / preprocess_name / sorter_name / param_name
                    #~ print(output_folder)
                    
                    if output_folder.is_dir():
                        print('Already exists, so skip it', output_folder)
                        continue
                    
                    rec_processed = func(rec)
                    si.run_sorter(sorter_name, rec_processed,
                            output_folder=output_folder,
                            verbose=True,
                            **sparams)

def run_all_post_processing():
    recordings = get_recordings()
    
    for rec_name, rec in recordings.items():
        #~ print(rec_name, rec)
        
        for preprocess_name, func in pre_processings.items():
            #~ print('  ', preprocess_name)
            
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
                    elif sorter_name == 'kilosort2':
                        sorting = si.KiloSortSortingExtractor(output_folder)

                    print(sorting)

                    # extactor waveforms
                    rec_processed = func(rec)
                    wf_folder = output_folder = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_waveforms')
                    if wf_folder.is_dir():
                        print('Already exists', wf_folder)
                        continue

                    we = si.extract_waveforms(rec_processed, sorting, folder=wf_folder,
                                load_if_exists=False, ms_before=1., ms_after=2.,
                                max_spikes_per_unit=500,
                                chunk_size=30000, n_jobs=6, progress_bar=True)
                    
                    # compute PCs
                    # pc = si.compute_principal_components(we, load_if_exists=True,
                    #             n_components=3, mode='by_channel_local')
                    pc = None
                    # compte metrics
                    metrics = si.compute_quality_metrics(we, waveform_principal_component=pc,
                            metric_names=['snr', 'isi_violation', ])
                    print(metrics)

                    # https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

                    # https://spikeinterface.readthedocs.io/en/latest/modules/toolkit/plot_4_curation.html


                    metric_file_path = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_metrics.csv')
                    metrics.to_csv(metric_file_path)
                    
                    # export report
                    report_folder = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_report')
                    si.export_report(we, report_folder, remove_if_exists=True,
                            chunk_size=30000, n_jobs=6, progress_bar=True)
                    
                    # export to phy
                    phy_folder = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_phy_export')
                    si.export_to_phy(rec_processed, sorting, phy_folder,
                                waveform_extractor=we,
                                compute_pc_features=False, compute_amplitudes=True,
                                remove_if_exists=True, chunk_size=30000, n_jobs=6, progress_bar=True)

                    # save to npz
                    output_sorting_folder = out_path / rec_name / preprocess_name / sorter_name / (param_name + '_sorting_uncurated')
                    sorting.save(output_sorting_folder)




def open_one_sorting():
    sub_path = 'sing/filter+cmr_radius/spykingcircus/custum_sc_1/'
    
    folder = out_path / sub_path
    print(folder)
    
    
    sorting_sc = si.SpykingCircusSortingExtractor(folder)
    print(sorting_sc)
    




if __name__ == '__main__':
    
    #~ run_all_sorters()
    
    run_all_post_processing()
    
    # open_one_sorting()
    