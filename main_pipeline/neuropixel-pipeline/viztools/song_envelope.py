from __future__ import annotations

from typing import Iterable, Union

import numpy as np
from scipy.signal import oaconvolve

from spikeinterface.core import BaseRecording, BaseRecordingSegment, get_chunk_with_margin
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.preprocessing import bandpass_filter, rectify


class KernelSmoothRecording(BasePreprocessor):
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
    trace_conv: KernelSmoothRecording
        The convolved recording extractor object.
    """

    name = "apply_kernel_smooth"

    def __init__(
        self, recording: BaseRecording, smooth_kernel = None
    ):
        BasePreprocessor.__init__(self, recording)
        self.annotate(is_filtered=True)

        for parent_segment in recording._recording_segments:
            self.add_recording_segment(KernelSmoothRecordingSegment(parent_segment, smooth_kernel))

        self._kwargs = {"recording": recording, "smooth_kernel": smooth_kernel}


class KernelSmoothRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self, parent_recording_segment: BaseRecordingSegment, smooth_kernel = None
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.smooth_kernel = smooth_kernel

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
            margin=self.smooth_kernel.shape[0],
            add_zeros=True,         
            add_reflect_padding=True,
        )
        dtype = traces.dtype

        trace_conv = oaconvolve(traces, self.smooth_kernel[:,np.newaxis], mode="same", axes=0)

        if right_margin > 0:
            return trace_conv[left_margin:-right_margin, :].astype(dtype)
        else:
            return trace_conv[left_margin:, :].astype(dtype)
        
        


apply_kernel_smooth = define_function_from_class(source_class=KernelSmoothRecording, name="apply_kernel_smooth")


def make_song_envelope(recording, channel_ids, kernel_size=1000, min_freq_envelope=1000, max_freq_envelope=8000):

    recording = recording.channel_slice(channel_ids=channel_ids)

    smooth_kernel = np.ones(kernel_size)/kernel_size

    bp_fil_rec = bandpass_filter(
        recording, freq_min=min_freq_envelope, freq_max=max_freq_envelope
    )
    squared_song = rectify(bp_fil_rec)
    smoothed = apply_kernel_smooth(squared_song, smooth_kernel=smooth_kernel)

    return smoothed
