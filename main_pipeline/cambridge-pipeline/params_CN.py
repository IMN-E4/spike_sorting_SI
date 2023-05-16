from pathlib import Path


base_input_folder = Path(
    "/data2/Anindita/Openephys/"
)

base_sorting_cache_folder = Path( "/data2/Anindita/Openephys_Sortings")

probe_type = 'ASSY-236-H5'
amp_type = 'cambridgeneurotech_mini-amp-64'


job_kwargs = {
    "n_jobs": 40,
    "chunk_duration": "1s",
    "progress_bar": True,
}

metrics_list = [
    "snr",
    "isi_violations",
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
    time_horizon_s=200.,
    convergence_method='lsqr_robust',
    robust_regression_sigma=2, lsqr_robust_n_iter=20,
    ##


    progress_bar=True,
    verbose=True,

)

cleaning_params = {
    "snr_threshold": 5.0,
    "firing_rate": 1,
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

kilosort2_5_params = {
    "do_correction": False,
    "docker_image": "spikeinterface/kilosort2_5-compiled-base",
}

sorters = {
    'kilosort2_5' : kilosort2_5_params
 }
