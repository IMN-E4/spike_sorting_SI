#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the pipeline for Marc/Milesa's data.
December 2022

Uses raw unsigned files and splits them into 32 channels (a sorting per channel).

preprocessed recording and waveforms are saved in local cache, final clean sorting
is saved in NAS.
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2022/12/07"  ### Date it was created
__status__ = (
    "Production"  ### Production = still being developed. Else: Concluded/Finished.
)


####################
# Review History   #
####################


####################
# Libraries        #
####################

# Standard imports  ### (Put here built-in libraries - https://docs.python.org/3/library/)
import shutil
from pathlib import Path

# Third party imports ###
import numpy as np
import spikeinterface.full as si
import probeinterface as pi

# Internal imports ###
from params import *


########### Functions ###########
def prepare_recs(
    raw_files,
    indx_to_slice=range(32),
    fs=20_000.0,
    dtype="uint16",
    n_channels=32,
    dist_contacts=40,
    gain_val=0.3815,
):
    """Create splitted recs with probe group

    Parameters
    ----------
    raw_file: .raw
        recording to work on.

    indx_to_slice: list or range
        which channels to use. default: 1 to 31

    fs: float
        sampling frequency

    dtype: string
        data dtype. default: 'uint16'

    n_channels: int
        number of channels in raw file. default: 33

    dist_contacts: int
        distance between contact in probe (we are creating a fake probe)

    Returns
    -------
    splitted_rec: spikeinterface object
        preprocessed rec
    """
    if dtype == "uint16":
        offset_to_uv = -gain_val * 2**15
    else:
        offset_to_uv = None

    # Read raw recording
    rec_raw = si.read_binary(
        raw_files,
        dtype=dtype,
        num_chan=n_channels,
        sampling_frequency=fs,
        gain_to_uV=gain_val,
        offset_to_uV=offset_to_uv,
    )
    rec_filter = si.bandpass_filter(rec_raw, **preproc_params)  # 0 center it
    rec_filter = si.zscore(rec_filter)  # to have all channels in same range
    print(rec_raw)

    # Create Probe group + set contacts and IDs
    probe_group = pi.ProbeGroup()
    for i in indx_to_slice:
        probe = pi.Probe()
        probe.set_contacts(positions=[[i * dist_contacts, 0]], shapes="circle")
        probe.set_contact_ids([f"elec_{i}"])
        probe.set_device_channel_indices([i])
        probe_group.add_probe(probe)

    rec_spike = rec_filter.set_probegroup(probe_group)
    splitted_rec = rec_spike.split_by("group")

    return splitted_rec


def run_spike_sorting(files):
    """Run sorting with different sorters and params [per electrode]

    Parameters
    ----------
    files: list
        list of raw file to work on

    Returns
    -------
    This function will result in sorting and waveform folders.

    """

    print("################ Runninng sorters! ################")
    working_folder = Path(output_path)
    print(working_folder)
    splitted_rec = prepare_recs(files)
    print(splitted_rec)
    for group, rec_one_channel in splitted_rec.items():
        # this make the recording virtually one segment
        rec_preprocess_one_seg = si.concatenate_recordings([rec_one_channel])
        tmp_path = output_path / "temp"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        rec_preprocess_one_seg_saved = rec_preprocess_one_seg.save(
            folder=tmp_path, n_jobs=10
        )  # Saving was necessary to correct it for sorters
        print(rec_preprocess_one_seg_saved)

        # Start sorting
        for sorter_name, params in sorters.items():
            sorting_folder = (
                working_folder / f"contact_{group}" / f"sorting_{sorter_name}"
            )
            print(sorting_folder)

            # Compute sorting
            if sorting_folder.exists():
                print(f"{sorter_name} already computed ")
                sorting = si.load_extractor(sorting_folder)
            else:
                try:
                    sorting_folder = (
                        working_folder
                        / f"contact_{group}"
                        / f"raw_sorting_{sorter_name}"
                    )
                    sorting_one_seg = si.run_sorter(
                        sorter_name,
                        rec_preprocess_one_seg_saved,
                        output_folder=sorting_folder,
                        delete_output_folder=True,
                        verbose=True,
                        raise_error=False,
                        **params,
                    )
                except:
                    sorting_one_seg = None

                if sorting_one_seg is None:
                    print(f"Sorter is failing on this {sorting_folder}")
                    shutil.rmtree(sorting_folder)
                    continue

                # If sorting has no units, continue. If it has, save it.
                if sorting_one_seg.unit_ids.size == 0:
                    print(f"no units to work on {sorting_folder}")
                    shutil.rmtree(sorting_folder)
                    continue

                # back to having a split sorting
                sorting = si.split_sorting(sorting_one_seg, rec_preprocess_one_seg)
                print(sorting)

            sorting_save_path = (
                working_folder / f"contact_{group}" / f"sorting_{sorter_name}"
            )

            if sorting_save_path.exists():
                print("sorting already saved")
            else:
                sorting = sorting.save(format="npz", folder=sorting_save_path)

            # Get WaveformExtractor
            wf_folder = working_folder / f"contact_{group}" / f"waveforms_{sorter_name}"
            we = si.extract_waveforms(
                rec_one_channel,
                sorting,
                folder=wf_folder,
                **waveform_params,
                **job_kwargs,
            )

            # Simple cleaning
            print("computing quality metrics")
            metrics = si.compute_quality_metrics(
                we, load_if_exists=False, metric_names=metrics_list
            )

            our_query = f"snr < {cleaning_params['snr_threshold']} | firing_rate < {cleaning_params['firing_rate']}"
            remove_unit_ids = metrics.query(our_query).index
            clean_sorting = we.sorting.remove_units(remove_unit_ids)
            print(clean_sorting)

            if clean_sorting.unit_ids.size == 0:
                print("no units to work on after cleaning")
                shutil.rmtree(wf_folder)
                shutil.rmtree(sorting_save_path)
                continue

            sorting_clean_folder = (
                working_folder / f"contact_{group}" / f"sorting_{sorter_name}_clean"
            )
            # Delete tree before recomputing
            if sorting_clean_folder.exists():
                print("remove exists clean", sorting_clean_folder)
                shutil.rmtree(sorting_clean_folder)

            else:
                clean_sorting = clean_sorting.save(
                    format="npz", folder=sorting_clean_folder
                )

            # Update Wf and create report with clean sorting
            wf_clean_folder = (
                working_folder / f"contact_{group}" / f"waveforms_{sorter_name}_clean"
            )
            we_clean = si.extract_waveforms(
                rec_one_channel,
                clean_sorting,
                folder=wf_clean_folder,
                **waveform_params,
                **job_kwargs,
            )

            print("computing quality metrics")
            si.compute_quality_metrics(
                we_clean, load_if_exists=False, metric_names=metrics_list
            )

            print("computing spike amplitudes")
            si.compute_spike_amplitudes(
                we_clean, load_if_exists=False, **amplitude_params, **job_kwargs
            )

            print("compute correlograms")
            si.compute_correlograms(
                we_clean, window_ms=50.0, bin_ms=1.0, load_if_exists=False
            )

            try:
                report_folder = (
                    working_folder / f"contact_{group}" / f"report_clean_{sorter_name}"
                )
                if report_folder.exists():
                    print("report already there for ", report_folder)
                else:
                    print("exporting report")
                    si.export_report(
                        we_clean,
                        report_folder,
                        remove_if_exists=False,
                        **job_kwargs,
                    )
            except:
                print(f"report failed for {report_folder}")


if __name__ == "__main__":
    files = list(base_input_folder.glob("*.raw"))
    file_num = [int(f.stem.split("_")[-1]) for f in files]
    order = np.argsort(file_num)
    files = [files[i] for i in order]
    print(files)
    run_spike_sorting(files[1:])
