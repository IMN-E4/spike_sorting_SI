

import spikeinterface.full as si

json_file = '/home/analysis_user/smb4k/NAS5802A5.LOCAL/PublicNeuropixel_Recordings/Impl_07_03_2022/Recordings/Rec_7_13_03_2022_g0/sorting_20220505/40000to50000/filter+cmr_radius/tridesclous/custom_tdc_1_waveforms/recording.json'

wf_folder= '/home/analysis_user/smb4k/NAS5802A5.LOCAL/PublicNeuropixel_Recordings/Impl_07_03_2022/Recordings/Rec_7_13_03_2022_g0/sorting_20220505/40000to50000/filter+cmr_radius/tridesclous/custom_tdc_1_waveforms/'

# rec = si.load_extractor(json_file, base_folder=wf_folder)
# print(rec)

wf = si.WaveformExtractor.load_from_folder(wf_folder)
print(wf)


