import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.full as si
import numpy as np
import neo
import json

base_folder = Path('/data1/ArthursLab/RomansData/AreaXLMAN/bird1/')
data_folder = base_folder / 'Rec_4_19_11_2021_g0'


# NIDQ part
rec_nidq = si.SpikeGLXRecordingExtractor(base_folder / data_folder, stream_id='nidq') # microphone
print(rec_nidq)
pulse_nidq = rec_nidq.get_traces(channel_ids=['nidq#XA1'])
pulse_nidq = pulse_nidq[:, 0]
print(pulse_nidq.shape)

thresh_nidq = (np.max(pulse_nidq) +  np.min(pulse_nidq)) / 2
print(thresh_nidq)

times_nidq = rec_nidq.get_times()
pulse_ind_nidq = np.flatnonzero((pulse_nidq[:-1]<=thresh_nidq) & (pulse_nidq[1:]>thresh_nidq)) # identifies the beggining of the pulse
pulse_time_nidq = times_nidq[pulse_ind_nidq]
print(np.diff(pulse_time_nidq))


# imec part
rec_ap = si.SpikeGLXRecordingExtractor(base_folder / data_folder, stream_id='imec0.ap')
times_ap = rec_ap.get_times()

reader = neo.SpikeGLXIO(data_folder, load_sync_channel=True)

stream_ind = np.flatnonzero(reader.header['signal_streams']['id'] == 'imec0.ap')
stream_ind = stream_ind[0]
# print('stream_ind', stream_ind)

# print(reader)
pulse_ap = reader.get_analogsignal_chunk(i_start=None, i_stop=None, stream_index=stream_ind, channel_ids=['imec0.ap#SY0'])
pulse_ap = pulse_ap[:, 0]


# Define a threshold
thresh_ap = 30. # there was a weird peak so we couldn't use min max
pulse_ind_ap = np.flatnonzero((pulse_ap[:-1]<=thresh_ap) & (pulse_ap[1:]>thresh_ap)) # identifies the beggining of the pulse
pulse_time_ap = times_ap[pulse_ind_ap]
# print(pulse_time_ap)
print(np.diff(pulse_time_ap))

# print(pulse_time_ap.shape, pulse_time_nidq.shape)

# Linear regression
import scipy.stats
a, b, r, tt, stderr = scipy.stats.linregress(pulse_time_nidq, pulse_time_ap)
print('regression nidq->imec.ap:', 'a', a, 'b', b, 'stderr', stderr)
assert np.abs(1 - a) < 0.0001, 'Very strange slope'
assert np.abs(b) < 0.5, 'intersept (delta) very strange'
assert stderr < 1e-5, 'sterr (tolerance) very strange'

# Visualize slope
fig, ax = plt.subplots()
ax.scatter(pulse_time_nidq, pulse_time_ap)
ax.plot(pulse_time_nidq, pulse_time_nidq * a + b, color='r')
plt.show()


times_nidq_corrected = times_nidq * a + b

fig, axs = plt.subplots(nrows=2, sharex=True)
axs[0].plot(pulse_nidq[-1000000:], label='nidq')
axs[1].plot(pulse_ap[-1000000:]*50, color='r', label='ap')
plt.legend()
plt.show()


# Before alignment
# fig, ax = plt.subplots()
# ax.plot(times_nidq[-1000000:], pulse_nidq[-1000000:], label='nidq')
# ax.plot(times_ap[-1000000:], pulse_ap[-1000000:]*50, color='r', label='ap')
# plt.legend()


fig, ax = plt.subplots()
ax.plot(times_nidq, pulse_nidq, label='nidq')
ax.plot(times_ap, pulse_ap*50, color='r', label='ap')
plt.legend()

# # Plot beggining
# fig, ax = plt.subplots()
# ax.plot(pulse_time_corrected_nidq[:1000000], pulse_nidq[:1000000])
# ax.plot(times_ap[:1000000], pulse_ap[:1000000]*50, color='r')

# Plot end
fig, ax = plt.subplots()
ax.plot(times_nidq_corrected[-1000000:], pulse_nidq[-1000000:], label='nidq')
ax.plot(times_ap[-1000000:], pulse_ap[-1000000:]*50, color='r', label='ap')
plt.legend()
plt.show()



# Save info for recording
results = {'a':a,
            'b':b, 
            'stderr':stderr
            }

with open(data_folder / 'regression_nidq2imec.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)