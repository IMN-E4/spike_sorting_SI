# %%
%load_ext autoreload
%autoreload 2

# %%
import spikeinterface.full as si
import matplotlib.pyplot as plt

from pathlib import Path

import spikeinterface.extractors as se

import probeinterface as pi

import neo


# %%
base_folder = Path('/data2/MilesasData/bin_test_this_is_the_correct/')

# %%

# pure binary
filename = base_folder / '221114_Hav_0000_nothing.raw'

rec = si.read_binary(filename, dtype='uint16', num_chan=97, sampling_frequency=20_000.)
rec.channel_ids

rec



# %%
fig , ax = plt.subplots()
si.plot_timeseries(rec, time_range=(10, 12), channel_ids=[1], ax=ax)
ax.set_ylim(32000,34000)


# %%
probe_group = pi.ProbeGroup()
for i in range(1,33):
    print(i)
    probe = pi.Probe()
    probe.set_contacts(positions=[[i*40,0]], shapes='circle')
    probe.set_contact_ids([f'elec_{i}'])
    probe.set_device_channel_indices([i])
    probe_group.add_probe(probe)

# %%
pi.plotting.plot_probe_group(probe_group, with_contact_id=True)

# %%
rec_spike = rec.set_probegroup(probe_group)
rec_spike

# %%
rec_spike.channel_ids

# %%
splitted_rec = rec_spike.split_by('group')

# %%
for group,rec_one_channel in splitted_rec.items():
    print(rec_one_channel)

# %%
si.runsorter?

# %%
# binary + MCS header
filename = base_folder / '221114_Hav_0000_writeheader_signed16bit.raw'


rec = se.read_mcsraw(filename)
rec


# %%


# %%
# filename = base_folder / '221114_Hav_0000_writeheader.raw'
filename = base_folder / '221114_Hav_0000_writeheader_signed16bit.raw'

# rec = se.read_mcsraw(filename)
reader = neo.rawio.RawMCSRawIO(filename)
reader.parse_header()
reader

# %%
filename = base_folder / '221114_Hav_0000.h5'
rec = se.read_mcsh5(filename)
rec

# %%



