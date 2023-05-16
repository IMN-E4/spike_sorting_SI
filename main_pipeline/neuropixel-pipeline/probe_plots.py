import spikeinterface.full as si
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from probeinterface.plotting import plot_probe


def plot_units_probe(we, peak_sign="neg", depth_axis=1):
    # visualize units on probe

    # Get probe
    probe = we.recording.get_probe()

    # Get unit IDs
    unit_ids = we.sorting.unit_ids

    # Get unit locations
    ulc = we.load_extension("unit_locations")
    unit_locations = ulc.get_data(outputs="by_unit")

    # Get amplitudes
    unit_amplitude = si.get_template_extremum_amplitude(we, peak_sign=peak_sign)
    unit_amplitude = np.abs([unit_amplitude[unit_id] for unit_id in unit_ids])

    # Get firing rate
    num_spikes = np.zeros(len(unit_ids))
    for i, unit_id in enumerate(unit_ids):
        for segment_index in range(we.sorting.get_num_segments()):
            st = we.sorting.get_unit_spike_train(
                unit_id=unit_id, segment_index=segment_index
            )
            num_spikes[i] += st.size
    size = num_spikes / max(num_spikes) * 120

    # Get probe parts
    contact_positions = we.recording.get_channel_locations()
    y_max = np.max(contact_positions[:, 1])
    y_min = np.min(contact_positions[:, 1]) - 100

    # Get X Y vals from unit locations
    xy_vals = [unit_locations[index] for index in unit_locations.keys()]
    x_vals = np.squeeze(xy_vals)[:, 0]
    y_vals = np.squeeze(xy_vals)[:, 1]

    # Plots
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
    plt.xlabel("x")
    plt.ylabel("depth")

    # Plot FR
    plot_probe(probe, ax=axs[0], title=False)
    plot_fr = axs[0].scatter(x_vals, y_vals, s=size, c=size, cmap="plasma")
    cbar_fr = fig.colorbar(plot_fr)
    cbar_fr.set_label(
        "Hz",
    )
    axs[0].set_aspect(0.3)

    # Plot amplitude
    plot_probe(probe, ax=axs[1], title=False)
    plot_amp = axs[1].scatter(
        x_vals, y_vals, s=unit_amplitude * 0.1, c=unit_amplitude, cmap="plasma"
    )
    cbar_amp = fig.colorbar(plot_amp)
    cbar_amp.set_label("Amplitude")
    axs[1].set_aspect(0.3)

    plt.ylim(y_min, y_max)
    plt.subplots_adjust(left=0.070, right=0.770, top=0.975, bottom=0.090)
    plt.show()


if __name__ == "__main__":
    we_path = Path(
        "/data1/Neuropixel_recordings/Imp_27_09_2022/sorting_cache/2022-10-Rec_28_09_2022_sleep_morning_g0-full/waveforms_clean_tridesclous/"
    )
    we = si.WaveformExtractor.load_from_folder(we_path)
    plot_units_probe(we)
