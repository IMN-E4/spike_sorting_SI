import spikeinterface.full as si
from pathlib import Path
from params import *
import shutil
import glob
import os

job_kwargs_local = dict(n_jobs=4 ,
                  chunk_duration='1s',
                  progress_bar=False,
                  )

def correct_metadata(output_path_file, rec_name, rec_ap_new, rec_lf_new, rec_mic_new):
    recs = [rec_ap_new, rec_lf_new, rec_mic_new]
    suffixes = ['_t0.imec0.ap.meta', '_t0.imec0.lf.meta', '_t0.nidq.meta']
    
    for rec, suffix in zip(recs,suffixes):
        file_path = (output_path_file / (rec_name+suffix))
        meta = file_path.read_text().splitlines()
        file_size_bytes = meta[5].split('=')
        file_time_secs = meta[6].split('=')
        new_val_bytes = os.path.getsize(file_path.as_posix().replace('.meta', '.bin'))
        new_val_time = rec.get_total_duration()
        file_size_bytes[1] = new_val_bytes
        file_time_secs[1] = new_val_time
        meta[5] = '='.join(map(str, file_size_bytes))
        meta[6] = '='.join(map(str, file_time_secs))
        with open(file_path, 'w') as fp:
                for item in meta:
                    # write each item on a new line
                    fp.write("%s\n" % item)
    print('Done')  
            

def export_files(rec_ap, rec_lf, rec_mic, original_path, output_path, time_range):
    # Sampling rates
    fs_ap = rec_ap.get_sampling_frequency()
    fs_mic = rec_mic.get_sampling_frequency()
    fs_lf = rec_lf.get_sampling_frequency()
    
    # Time range
    t0 = time_range[0]
    t1 = time_range[1]
    
    # Create new recs
    rec_ap_new = rec_ap.frame_slice(start_frame=t0*fs_ap, end_frame=t1*fs_ap)
    rec_mic_new = rec_mic.frame_slice(start_frame=t0*fs_mic, end_frame=t1*fs_mic)
    rec_lf_new = rec_lf.frame_slice(start_frame=t0*fs_lf, end_frame=t1*fs_lf)
    
    # Correct Path
    rec_name = output_path.stem
    output_path_file = Path(output_path.as_posix() + f'_from_{t0}_to_{t1}s')
    
    shutil.rmtree(output_path_file, ignore_errors=True)
    output_path_file.mkdir()
    
    # Save new recs
    si.write_binary_recording(rec_ap_new, file_paths=[output_path_file / (rec_name+'_t0.imec0.ap.bin')], dtype='int16', add_file_extension=False,
                           verbose=False, byte_offset=0, auto_cast_uint=False,  **job_kwargs_local)
    si.write_binary_recording(rec_mic_new, file_paths=[output_path_file / (rec_name+'_t0.nidq.bin')], dtype='int16', add_file_extension=False,
                           verbose=False, byte_offset=0, auto_cast_uint=False, **job_kwargs_local)
    si.write_binary_recording(rec_lf_new, file_paths=[output_path_file / (rec_name+'_t0.imec0.lf.bin')], dtype='int16', add_file_extension=False,
                           verbose=False, byte_offset=0, auto_cast_uint=False, **job_kwargs_local)
    
    # Move metadata
    files_list = glob.glob(original_path.as_posix() + '/**/**.meta', recursive=True)
    for file in files_list:
        file = Path(file)
        shutil.copy(file, output_path_file/file.parts[-1])
        
    correct_metadata(output_path_file, rec_name, rec_ap_new, rec_lf_new, rec_mic_new)
    
    readme_path = output_path_file / 'readme.txt'
    with open(readme_path, 'w') as fp:
                   fp.write(f'This recording is not the original, but a chunked version from {t0} to {t1} seconds') 
    
    
    
    