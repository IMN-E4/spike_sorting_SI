from __future__ import annotations

from typing import Iterable, Union

import numpy as np
from scipy.signal import oaconvolve

from spikeinterface.core import BaseRecording, BaseRecordingSegment, get_chunk_with_margin
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.preprocessing import bandpass_filter, rectify, gaussian_filter, resample


class MeanShiftRecording(BasePreprocessor):
    """
    Class for performing a rectangle smoothing on a recording.

    This is done by a convolution with a rectangle kernel.

    Here, convolution is performed using oaconvolve from scipy.

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor to be filtered.

    Returns
    -------
    trace_conv: MeanShiftRecording
        The convolved recording extractor object.
    """

    name = "apply_mean_shift"

    def __init__(
        self, recording: BaseRecording, margin_ms = None
    ):
        BasePreprocessor.__init__(self, recording)
        self.annotate(is_filtered=True)

        for parent_segment in recording._recording_segments:
            self.add_recording_segment(MeanShiftRecordingSegment(parent_segment, margin_ms))

        self._kwargs = {"recording": recording, "margin_ms": margin_ms}


class MeanShiftRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self, parent_recording_segment: BaseRecordingSegment, margin_ms = None
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.margin_ms = margin_ms

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[Iterable, None] = None,
    ):
        traces, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            channel_indices,
            margin=self.margin_ms,
            add_zeros=True,         
            add_reflect_padding=True,
        )
        dtype = traces.dtype

        trace_conv = traces - np.mean(traces, axis=0)

        if right_margin > 0:
            return trace_conv[left_margin:-right_margin, :].astype(dtype)
        else:
            return trace_conv[left_margin:, :].astype(dtype)
        
        
apply_mean_shift = define_function_from_class(source_class=MeanShiftRecording, name="apply_mean_shift")


def make_ap_envelope(recording, ds_rate=1000, lowpass_freq=500, margin_ms=3000):
    """ Function to create an envelope from AP channels

        Arguments
        ---------

        recording: spikeinterface obj

        ds rate: int
            downsampled sampling frequency

        lowpass_freq: int
            lowpass frequency

        margin_ms: int
            size of margin for chunk computations (due to lazy mode)
    
    """

    # Zero center recording
    mean_shifted = apply_mean_shift(recording, margin_ms=margin_ms)

    # Rectify trace
    squared_song = rectify(mean_shifted)

    # Lowpass rectified trace
    lp_trace = gaussian_filter(squared_song, freq_min=None, freq_max=lowpass_freq)
        
    # Downsample
    ds_factor = int(recording.get_sampling_frequency() / ds_rate)
        
    samp_period = recording.get_sampling_frequency()/ds_factor

    ds = resample(lp_trace, samp_period, margin_ms=margin_ms)

    return ds
