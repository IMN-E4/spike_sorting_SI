from pathlib import Path


base_input_folder = Path(
    "/data2/MilesasData/dataset/"
)

output_path = Path("/data2/MilesasData/dataset/current-test_all/")

job_kwargs = {
    "n_jobs": 40,
    "chunk_duration": "1s",
    "progress_bar": True,
}

preproc_params = { #### check with same if this is ok
    " dtype":'float32',
    "freq_min": 300,
    "freq_max": 6000,
    "filter_order" : 2,
    "ftype" : 'bessel'
}


metrics_list = [
    "snr",
    "num_spikes",
    "firing_rate"
]

waveform_params = {
    "ms_before": 1.0,
    "ms_after": 2.0,
    "max_spikes_per_unit": 500,
    "use_relative_path": True,
}

peak_sign = "neg"


amplitude_params = {
    "return_scaled": False, # binary doesnt give the scale
    "peak_sign": peak_sign,
}



#### Sorter params!
tridesclous_params = {
    "freq_min": 300.0,
    "freq_max": 6000.0,
    "detect_threshold": 5,
    "common_ref_removal": False,
    "nested_params": {
        "chunksize": 30000,
        "preprocessor": {"engine": "numpy"},
        "peak_detector": {"adjacency_radius_um": 100},
        "clean_peaks": {"alien_value_threshold": 100.0},
        "peak_sampler": {
            "mode": "rand_by_channel",
            "nb_max_by_channel": 2000,
        },
    },
}

tridesclous_params_docker = {
    "docker_image" : "spikeinterface/tridesclous-base"
}

kilosort2_params = {
    "docker_image": "spikeinterface/kilosort2-compiled-base",
}

kilosort2_5_params = {
    "docker_image": "spikeinterface/kilosort2_5-compiled-base",
}

mountainsort4_params = {
    "docker_image": "spikeinterface/mountainsort4-base:latest",
}

spykingcircus2_params = {}

yass_params = {
    "docker_image": "spikeinterface/yass-base",
}

sorters = {
    "tridesclous": tridesclous_params,
    # 'kilosort2' : kilosort2_params,
    'kilosort2_5' : kilosort2_5_params,
    # 'experimental_sorter1': dict(delete_existing=True),
    # 'spykingcircus2' : spykingcircus2_params,
    # "yass" : yass_params
    "mountainsort4": mountainsort4_params
}
