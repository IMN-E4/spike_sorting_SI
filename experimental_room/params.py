job_kwargs = dict(n_jobs=4 ,
                  chunk_duration='1s',
                  progress_bar=False,
                  verbose=True
                  )

peak_sign='neg'

peak_detection_params = dict(
    method='locally_exclusive',
    peak_sign=peak_sign,
    detect_threshold=5.,
    local_radius_um=150,
)


peak_location_params = dict(
    ms_before=1, 
    ms_after=1,
    method='monopolar_triangulation',
    method_kwargs= dict(local_radius_um=150, max_distance_um=1000, optimizer='minimize_with_log_penality'),
)
