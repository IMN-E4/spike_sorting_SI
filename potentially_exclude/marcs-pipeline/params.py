from pathlib import Path


base_input_folder = Path(
    "/data2/MilesasData/dataset/"
)

output_path = Path("/data2/MilesasData/dataset/run_050623")

job_kwargs = {
    "n_jobs": 25,
    "chunk_duration": "1s",
    "progress_bar": True,
}


cleaning_params = {
    "snr_threshold":2,
    "firing_rate":0.5
}

preproc_params = { 
    "dtype":'float32',
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

correlogram_params = {
    "window_ms":50.,
    "bin_ms":1.0
}



#### Sorter params!
tridesclous_params_default = {
    "detect_threshold": 4
}

tridesclous_params_docker = {
    "docker_image" : "spikeinterface/tridesclous-base"
}

mountainsort4_params = {
    "docker_image": "spikeinterface/mountainsort4-base:latest",
}

waveclus_params = {
    "docker_image": "spikeinterface/waveclus-compiled-base"
}

ironclust_params = {
    "docker_image": "spikeinterface/ironclust-compiled-base"
}




sorters = {
    "tridesclous": tridesclous_params_default,
    "waveclus": waveclus_params,
    "ironclust": ironclust_params,
    "mountainsort4": mountainsort4_params
}
