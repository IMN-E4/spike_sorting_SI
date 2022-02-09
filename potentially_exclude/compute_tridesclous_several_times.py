import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
from probeinterface import read_spikeglx
from probeinterface.plotting import plot_probe
import numpy as np

base_folder = Path('/home/arthur/Documents/SpikeSorting/Test_20210518/') 

data_folder = base_folder / 'raw_awake'

out_path = Path('/media/storage/spikesorting_output/tridesclous_several_times/')


def get_recording():
    

    data_folder = base_folder / 'raw_awake'

    out_path = Path('/media/storage/spikesorting_output/REST/')

    recording = si.SpikeGLXRecordingExtractor(data_folder, stream_id='imec0.ap')
    #print(recording)

    probe = read_spikeglx(data_folder / 'raw_awake_01_g0_t0.imec0.ap.meta')
    #print(probe)
    
    # print(recording)

    fs = recording.get_sampling_frequency()
    # t0, t1 = (2000., 2500.) # Time in seconds
    t0, t1 = 1000, 1100
    frame0, frame1 = int(t0*fs), int(t1*fs)


    recording = recording.frame_slice(frame0, frame1)

    recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)

    recording = recording.set_probe(probe)

    recording = si.common_reference(recording, reference='local', local_radius=(50, 100))

    

    


    name = f'filtered common_ref local singing window_{t0}_{t1} run_5'

    return recording, name


def run_one_tridesclous(rec, output_folder):
    tdc_params = {
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

    sorting = si.run_sorter('tridesclous', rec,
                       output_folder=output_folder,
                        verbose=True,
                        raise_error=True,
                        **tdc_params)

    # sc_params = {'detect_sign': -1,
    #             'adjacency_radius': 100,
    #             'detect_threshold': 6,
    #             'template_width_ms': 3,
    #             'filter': True,
    #             'merge_spikes': True,
    #             'auto_merge': 0.75,
    #             'num_workers': None,
    #             'whitening_max_elts': 1000,
    #             'clustering_max_elts': 10000}
    # sorting_SC = si.run_spykingcircus(rest_rec, output_folder=out_path / 'output_spykingcircus_new', **sc_params)


def compare_2_by_2():
    folder0 = 'output_tridesclous_filtered singing window_1000_1100 run_1'
    folder1 = 'output_tridesclous_filtered common_ref local singing window_1000_1100 run_5'

    sorting0 = si.TridesclousSortingExtractor(out_path / folder0 / 'tridesclous')
    sorting1 = si.TridesclousSortingExtractor(out_path / folder1/ 'tridesclous')

    comp = si.compare_two_sorters(sorting0, sorting1)

    fig, ax = plt.subplots()
    si.plot_agreement_matrix(comp, ax=ax)
    plt.show()


def compute_all_metrics():

    for folder in out_path.iterdir():
        print(folder)
    
        sorting_folder = folder / 'tridesclous'
        wf_folder = folder / 'tridesclous_waveforms'
        sorting = si.TridesclousSortingExtractor(sorting_folder)
        recording, name = get_recording()

        we = si.extract_waveforms(recording, sorting, wf_folder,
            load_if_exists=True,
            ms_before=1, ms_after=2., max_spikes_per_unit=500,
            n_jobs=3, chunk_size=30000, progress_bar=True)
        print(we)

        pc = st.compute_principal_components(we, load_if_exists=True,
                    n_components=3, mode='by_channel_local')
        print(pc)        

        metrics = st.compute_quality_metrics(we, waveform_principal_component=pc)
        print(metrics)






if __name__ == '__main__':
    #rec, name = get_recording()
    #print(rec)

    #folder = out_path / ('output_tridesclous_' + name)
    # run_one_tridesclous(rec, folder)


    compare_2_by_2()

    # compute_all_metrics()
