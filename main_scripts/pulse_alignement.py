import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
import numpy as np
import neo
import json

from params import *
from recording_list import recording_list
from spike_sorting_pipeline import get_workdir_folder

import scipy.stats


def compute_pulse_alignement(spikeglx_folder, time_range=None, channel_range=None):
    assert time_range is None

    # Read NIDQ stream
    rec_nidq, working_folder = get_workdir_folder(
        spikeglx_folder,
        time_range=time_range,
        channel_range=channel_range,
        stream_id='nidq',
    )
    pulse_nidq = rec_nidq.get_traces(channel_ids=['nidq#XA1'])
    pulse_nidq = pulse_nidq[:, 0]
    thresh_nidq = (np.max(pulse_nidq) +  np.min(pulse_nidq)) / 2

    times_nidq = rec_nidq.get_times()
    pulse_ind_nidq = np.flatnonzero((pulse_nidq[:-1]<=thresh_nidq) & (pulse_nidq[1:]>thresh_nidq)) # identifies the beggining of the pulse
    pulse_time_nidq = times_nidq[pulse_ind_nidq]

    assert np.all(np.diff(pulse_time_nidq)>0.99) # to check if there are no artifacts that could affect the alignment
    assert np.all(np.diff(pulse_time_nidq)<1.01) # to check if there are no artifacts that could affect the alignment

    # Read AP stream
    rec_ap, working_folder = get_workdir_folder(
        spikeglx_folder,
        time_range=time_range,
        channel_range=channel_range,
        stream_id='imec0.ap',
        load_sync_channel=True,
    )
    times_ap = rec_ap.get_times()
    pulse_ap = rec_ap.get_traces(channel_ids=[rec_ap.channel_ids[-1]])
    pulse_ap = pulse_ap[:, 0]


    # Define a threshold
    thresh_ap = 30. # there was a weird peak so we couldn't use min max
    pulse_ind_ap = np.flatnonzero((pulse_ap[:-1]<=thresh_ap) & (pulse_ap[1:]>thresh_ap)) # identifies the beggining of the pulse
    pulse_time_ap = times_ap[pulse_ind_ap]

    assert np.all(np.diff(pulse_time_ap)>0.99) # to check if there are no artifacts that could affect the alignment
    assert np.all(np.diff(pulse_time_ap)<1.01) # to check if there are no artifacts that could affect the alignment


    # Linear regression
    a, b, r, tt, stderr = scipy.stats.linregress(pulse_time_nidq, pulse_time_ap)
    times_nidq_corrected = times_nidq * a + b

    print('regression nidq->imec.ap:', 'a', a, 'b', b, 'stderr', stderr)
    assert np.abs(1 - a) < 0.0001, 'Very strange slope'
    assert np.abs(b) < 0.5, 'intercept (delta) very strange'
    assert stderr < 1e-5, 'sterr (tolerance) very strange'


    synchro_folder = working_folder / 'synchro'
    synchro_folder.mkdir(exist_ok=True)

    np.save(synchro_folder / 'pulse_time_nidq.npy', pulse_time_nidq)
    np.save(synchro_folder / 'pulse_time_ap.npy', pulse_time_ap)

    # Save info for recording
    synchro_dict = {'a':a,
                'b':b, 
                'stderr':stderr
                }

    with open(synchro_folder / 'synchro_nidq_on_imecap.json', 'w') as outfile:
        json.dump(synchro_dict, outfile, indent=4)




def test_compute_pulse_alignement():
    for implant_name, name, time_range, channel_range in recording_list:
        spikeglx_folder = (
            base_input_folder / implant_name / "Recordings" / name
        )
        compute_pulse_alignement(spikeglx_folder, time_range=time_range, channel_range=channel_range)


def test_plot_synchro():
    working_folder = Path('/data1/Neuropixel_recordings/Imp_27_09_2022/sorting_cache/2022-10-Rec_30_09_2022_g0-full')
    synchro_folder = working_folder / 'synchro'

    implant_name = 'Imp_27_09_2022'
    name = 'Rec_30_09_2022_g0'
    spikeglx_folder = (
        base_input_folder / implant_name / "Recordings" / name
    )
    time_range = None
    channel_range = [0,200]
    
    pulse_time_nidq = np.load(synchro_folder / 'pulse_time_nidq.npy')
    pulse_time_ap = np.load(synchro_folder / 'pulse_time_ap.npy')

    with open(synchro_folder / 'synchro_nidq_on_imecap.json', 'r') as outfile:
        synchro_dict = json.load(outfile)
    print(synchro_dict)
    a = synchro_dict['a']
    b = synchro_dict['b']

    # read again pulse
    rec_nidq, working_folder = get_workdir_folder(
        spikeglx_folder,
        time_range=time_range,
        channel_range=channel_range,
        stream_id='nidq',
    )
    pulse_nidq = rec_nidq.get_traces(channel_ids=['nidq#XA1'])
    times_nidq = rec_nidq.get_times()

    rec_ap, working_folder = get_workdir_folder(
        spikeglx_folder,
        time_range=time_range,
        channel_range=channel_range,
        stream_id='imec0.ap',
        load_sync_channel=True,
    )
    times_ap = rec_ap.get_times()
    pulse_ap = rec_ap.get_traces(channel_ids=[rec_ap.channel_ids[-1]])
    pulse_ap = pulse_ap[:, 0]

    times_nidq_corrected = times_nidq * a + b

    fig, ax = plt.subplots()
    ax.scatter(pulse_time_nidq, pulse_time_ap)
    ax.plot(pulse_time_nidq, pulse_time_nidq * a + b, color='r')
    plt.title('Visualize Slope')

    # 

    # ## Sanity Checks
    # # Before alignment
    fig, ax = plt.subplots()
    ax.plot(times_nidq[-1000000:], pulse_nidq[-1000000:], label='nidq')
    ax.plot(times_ap[-1000000:], pulse_ap[-1000000:]*50, color='r', label='ap')
    plt.legend()
    plt.title('Before Alignment')

    # Plot beggining
    # fig, ax = plt.subplots()
    # ax.plot(pulse_time_corrected_nidq[:1000000], pulse_nidq[:1000000])
    # ax.plot(times_ap[:1000000], pulse_ap[:1000000]*50, color='r')

    # # After alignment
    fig, ax = plt.subplots()
    ax.plot(times_nidq_corrected[-1000000:], pulse_nidq[-1000000:], label='nidq')
    ax.plot(times_ap[-1000000:], pulse_ap[-1000000:]*50, color='r', label='ap')
    plt.legend()
    plt.title('After Alignment')

    # Plot error distribution after alignment
    # diff = times_nidq_corrected[pulse_ind_nidq] - pulse_time_ap
    # plt.figure()
    # plt.hist(diff)
    # plt.title('error distribution after alignment')


    plt.show()



if __name__ == '__main__':
    # test_compute_pulse_alignement()

    test_plot_synchro()







# base_folder = Path('/data1/ArthursLab/RomansData/AreaXLMAN/bird1/')
# data_folder = base_folder / 'Rec_4_19_11_2021_g0'


# # NIDQ part
# rec_nidq = si.SpikeGLXRecordingExtractor(base_folder / data_folder, stream_id='nidq') # microphone
# print(rec_nidq)
# pulse_nidq = rec_nidq.get_traces(channel_ids=['nidq#XA1'])
# pulse_nidq = pulse_nidq[:, 0]
# print(pulse_nidq.shape)

# thresh_nidq = (np.max(pulse_nidq) +  np.min(pulse_nidq)) / 2
# print(thresh_nidq)

# times_nidq = rec_nidq.get_times()
# pulse_ind_nidq = np.flatnonzero((pulse_nidq[:-1]<=thresh_nidq) & (pulse_nidq[1:]>thresh_nidq)) # identifies the beggining of the pulse
# pulse_time_nidq = times_nidq[pulse_ind_nidq]

# # print(np.diff(pulse_time_nidq))

# assert np.all(np.diff(pulse_time_nidq)>0.99) # to check if there are no artifacts that could affect the alignment
# assert np.all(np.diff(pulse_time_nidq)<1.01) # to check if there are no artifacts that could affect the alignment


# # imec part (wasn't possible to get it direcly from SI, yet)
# rec_ap = si.SpikeGLXRecordingExtractor(base_folder / data_folder, stream_id='imec0.ap')
# times_ap = rec_ap.get_times()

# reader = neo.SpikeGLXIO(data_folder, load_sync_channel=True)

# stream_ind = np.flatnonzero(reader.header['signal_streams']['id'] == 'imec0.ap')
# stream_ind = stream_ind[0]
# pulse_ap = reader.get_analogsignal_chunk(i_start=None, i_stop=None, stream_index=stream_ind, channel_ids=['imec0.ap#SY0'])
# pulse_ap = pulse_ap[:, 0]


# # Define a threshold
# thresh_ap = 30. # there was a weird peak so we couldn't use min max
# pulse_ind_ap = np.flatnonzero((pulse_ap[:-1]<=thresh_ap) & (pulse_ap[1:]>thresh_ap)) # identifies the beggining of the pulse
# pulse_time_ap = times_ap[pulse_ind_ap]

# # print(pulse_time_ap)
# print(np.diff(pulse_time_ap))

# assert np.all(np.diff(pulse_time_ap)>0.99) # to check if there are no artifacts that could affect the alignment
# assert np.all(np.diff(pulse_time_ap)<1.01) # to check if there are no artifacts that could affect the alignment


# # Linear regression
# import scipy.stats
# a, b, r, tt, stderr = scipy.stats.linregress(pulse_time_nidq, pulse_time_ap)
# times_nidq_corrected = times_nidq * a + b

# print('regression nidq->imec.ap:', 'a', a, 'b', b, 'stderr', stderr)
# assert np.abs(1 - a) < 0.0001, 'Very strange slope'
# assert np.abs(b) < 0.5, 'intercept (delta) very strange'
# assert stderr < 1e-5, 'sterr (tolerance) very strange'

# # Visualize slope
# fig, ax = plt.subplots()
# ax.scatter(pulse_time_nidq, pulse_time_ap)
# ax.plot(pulse_time_nidq, pulse_time_nidq * a + b, color='r')
# plt.title('Visualize Slope')
# plt.show()


# ## Sanity Checks
# # Before alignment
# fig, ax = plt.subplots()
# ax.plot(times_nidq[-1000000:], pulse_nidq[-1000000:], label='nidq')
# ax.plot(times_ap[-1000000:], pulse_ap[-1000000:]*50, color='r', label='ap')
# plt.legend()
# plt.title('Before Alignment')

# # # Plot beggining
# # fig, ax = plt.subplots()
# # ax.plot(pulse_time_corrected_nidq[:1000000], pulse_nidq[:1000000])
# # ax.plot(times_ap[:1000000], pulse_ap[:1000000]*50, color='r')

# # After alignment
# fig, ax = plt.subplots()
# ax.plot(times_nidq_corrected[-1000000:], pulse_nidq[-1000000:], label='nidq')
# ax.plot(times_ap[-1000000:], pulse_ap[-1000000:]*50, color='r', label='ap')
# plt.legend()
# plt.title('After Alignment')
# plt.show()

# # Plot error distribution after alignment
# diff = times_nidq_corrected[pulse_ind_nidq] - pulse_time_ap
# plt.figure()
# plt.hist(diff)
# plt.title('error distribution after alignment')
# plt.show()


# # Save info for recording
# results = {'a':a,
#             'b':b, 
#             'stderr':stderr
#             }

# with open(data_folder / 'regression_nidq2imec.json', 'w') as outfile:
#     json.dump(results, outfile, indent=4)