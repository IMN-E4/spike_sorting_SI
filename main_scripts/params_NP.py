from pathlib import Path


base_input_folder = Path(
    "/nas/Neuropixel_Recordings/Cerebellum/"
)

base_sorting_cache_folder = Path("/data2/Neuropixel_recordings/Cerebellum/")

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
    "sparse": True
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

unit_location_params = {
    "method":"monopolar_triangulation",
    "radius_um":150,
    "max_distance_um":1000,
    "optimizer":"minimize_with_log_penality",

}

correlogram_params = {
    "window_ms":50.0, 
    "bin_ms":1.0,
}

peak_location_params = {
    "ms_before": 1.,
    "ms_after": 1.,
    "method": "monopolar_triangulation",
    "local_radius_um":50.,
    "max_distance_um":100,
    "optimizer":"minimize_with_log_penality"
}

motion_estimation_params = dict(
    # historgram of raster
    bin_duration_s=10., bin_um=5., margin_um=0.,
    # non rigid
    rigid=False, win_shape='gaussian', win_step_um=150., win_sigma_um=450.,
    # clean : experimental we do not user
    post_clean=False, speed_threshold=30, sigma_smooth_s=None,
    # the method itself decentralised = paninski NY group 'iterative_template' : KS
    method='decentralized',
    ## decentralized
    pairwise_displacement_method='conv', max_displacement_um=500., weight_scale='linear',
    error_sigma=0.2,
    # conv_engine='numpy',
    conv_engine='torch', # pip install torch CPU or GPU
    torch_device=None, batch_size=1,
    corr_threshold=0,
    # time_horizon_s=None,
    time_horizon_s=400.,
    convergence_method='lsqr_robust',
    robust_regression_sigma=2, lsqr_robust_n_iter=20,
    ##


    progress_bar=True,
    verbose=True,

)

cleaning_params = {
    "snr_threshold": 4.0,
    "firing_rate": 0.5,
}


first_merge_params = {"steps":None,
        "maximum_distance_um":150,
        "corr_diff_thresh":0.16,
        "template_diff_thresh":0.25,
        "minimum_spikes":1000
}

second_merge_params = {"steps":['min_spike', 'unit_positions', 'template_similarity'],
    "template_diff_thresh": 0.25,
    "maximum_distance_um": 20

}

classification_params = {
    "snr": 8,
    "isi_violations_ratio": 0.5,
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

tridesclous_params_docker = {
    "docker_image" : "spikeinterface/tridesclous-base"
}

kilosort2_params = {
    "docker_image": "spikeinterface/kilosort2-compiled-base",
}

kilosort2_5_params = {
    "do_correction": False,
    "docker_image": "spikeinterface/kilosort2_5-compiled-base",
}

spykingcircus2_params = {}

yass_params = {
    "docker_image": "spikeinterface/yass-base",
}

sorters = {
    # "tridesclous": tridesclous_params_docker,
    # 'kilosort2' : kilosort2_params,
    'kilosort2_5' : kilosort2_5_params
    # 'experimental_sorter1': dict(delete_existing=True),
    # 'spykingcircus2' : spykingcircus2_params,
    # "yass" : yass_params
}
