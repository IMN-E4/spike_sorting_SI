from pathlib import Path


base_input_folder = Path(
    "/home/analysis_user/smb4k/NAS5802A5.LOCAL/Public/Neuropixel_Recordings/AreaX-LMAN/"
)


# base_input_folder = Path('/media/analysis_user/LaCie/Data/Neuropixels')


base_sorting_cache_folder = Path("/data1/Neuropixel_recordings/")

# base_sorting_cache_folder = Path('/data1_test')


job_kwargs = {
    "n_jobs": 20,
    "chunk_duration": "1s",
    "progress_bar": True,
}

metrics_list = [
    "snr",
    "isi_violation",
    "num_spikes",
    "firing_rate",
    "presence_ratio",
]

waveform_params = {
    "ms_before": 1.0,
    "ms_after": 2.0,
    "max_spikes_per_unit": 500,
    "use_relative_path": True,
}

peak_sign = "neg"


amplitude_params = {
    "return_scaled": True,
    "peak_sign": peak_sign,
}


peak_detection_params = {
    "method": "locally_exclusive",
    "peak_sign": peak_sign,
    "detect_threshold": 5.0,
    "local_radius_um": 150,
}


peak_location_params = {
    "ms_before": 1,
    "ms_after": 1,
    "method": "monopolar_triangulation",
    "method_kwargs": dict(
        local_radius_um=150,
        max_distance_um=1000,
        optimizer="minimize_with_log_penality",
    ),
}

cleaning_params = {
    "snr_threshold": 5.0,
    "firing_rate": 1,
}


tridesclous_params = {
    "freq_min": 300.0,
    "freq_max": 6000.0,
    "detect_threshold": 5,
    "common_ref_removal": False,
    "nested_params": {
        "chunksize": 30000,
        "preprocessor": {"engine": "opencl"},
        # 'preprocessor' : {'engine': 'numpy'},
        "peak_detector": {"adjacency_radius_um": 100},
        "clean_peaks": {"alien_value_threshold": 100.0},
        "peak_sampler": {
            "mode": "rand_by_channel",
            "nb_max_by_channel": 2000,
        },
    },
}


kilosort2_params = {
    "docker_image": "spikeinterface/kilosort2-compiled-base",
}

kilosort2_5_params = {
    "docker_image": "spikeinterface/kilosort2_5-compiled-base",
}

spykingcircus2_params = {}

sorters = {
    "tridesclous": tridesclous_params,
    "kilosort2_5": kilosort2_5_params,
    # 'kilosort2' : kilosort2_params,
    # 'experimental_sorter1': dict(delete_existing=True),
    # 'spykingcircus2' : spykingcircus2_params,
}
