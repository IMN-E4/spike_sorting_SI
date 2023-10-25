import spikeinterface.full as si
from pathlib import Path
from params import *
import shutil
import glob
import os

job_kwargs_local = dict(n_jobs=20,
                  chunk_duration='1s',
                  progress_bar=False,
                  )

def correct_metadata(output_path_file, 
                     rec_name, 
                     rec_ap_new,
                     rec_lf_new, 
                     rec_mic_new,
                     time_range):
    # Time range
    t0 = time_range[0]
    t1 = time_range[1]

    recs = list()
    suffixes = list()
    if rec_ap_new is not None:
        recs += [rec_ap_new]
        suffixes += ['_t0.imec0.ap.meta']

    if rec_mic_new is not None:
        recs += [rec_mic_new]
        suffixes += ['_t0.nidq.meta']

    if rec_lf_new is not None:
        recs += [rec_lf_new]
        suffixes += ['_t0.imec0.lf.meta']

    for rec, suffix in zip(recs,suffixes):
        file_path = (output_path_file / (rec_name+suffix))
        print(file_path)
        meta = file_path.read_text().splitlines()
        file_size_bytes = meta[5].split('=')
        file_time_secs = meta[6].split('=')
        new_f_name = (output_path_file / (rec_name+f'_from_{t0}_to_{t1}s_g0'+suffix))
        new_val_bytes = os.path.getsize(new_f_name.as_posix().replace('.meta', '.bin'))
        new_val_time = rec.get_total_duration()
        file_size_bytes[1] = new_val_bytes
        file_time_secs[1] = new_val_time
        meta[5] = '='.join(map(str, file_size_bytes))
        meta[6] = '='.join(map(str, file_time_secs))
        with open(file_path, 'w') as fp:
                for item in meta:
                    # write each item on a new line
                    fp.write("%s\n" % item)
        
        os.rename(file_path, new_f_name)

    print('Done')  
            

def export_files(rec_ap,
                 rec_lf, 
                 rec_mic, 
                 original_path, 
                 output_path, 
                 time_range):
    # Time range
    t0 = time_range[0]
    t1 = time_range[1]

    # Correct Path
    rec_name = output_path.stem
    output_path_file = Path(output_path.as_posix() + f'_from_{t0}_to_{t1}s')

    shutil.rmtree(output_path_file, ignore_errors=True)
    output_path_file.mkdir()


    rec_ap_new = None
    if rec_ap is not None:
        fs_ap = rec_ap.get_sampling_frequency()
        rec_ap_new = rec_ap.frame_slice(start_frame=t0*fs_ap, end_frame=t1*fs_ap)
        si.write_binary_recording(rec_ap_new, file_paths=[output_path_file / (rec_name+f'_from_{t0}_to_{t1}s'+'_g0_t0.imec0.ap.bin')], dtype='int16', add_file_extension=False,
                           verbose=False, byte_offset=0, auto_cast_uint=False,  **job_kwargs_local)

    rec_mic_new = None
    if rec_mic is not None:
        fs_mic = rec_mic.get_sampling_frequency()
        rec_mic_new = rec_mic.frame_slice(start_frame=t0*fs_mic, end_frame=t1*fs_mic)
        si.write_binary_recording(rec_mic_new, file_paths=[output_path_file / (rec_name+f'_from_{t0}_to_{t1}s'+'_g0_t0.nidq.bin')], dtype='int16', add_file_extension=False,
                           verbose=False, byte_offset=0, auto_cast_uint=False, **job_kwargs_local)

    rec_lf_new = None
    if rec_lf is not None:
        fs_lf = rec_lf.get_sampling_frequency()
        rec_lf_new = rec_lf.frame_slice(start_frame=t0*fs_lf, end_frame=t1*fs_lf)
        si.write_binary_recording(rec_lf_new, file_paths=[output_path_file / (rec_name+f'_from_{t0}_to_{t1}s'+'_g0_t0.imec0.lf.bin')], dtype='int16', add_file_extension=False,
                           verbose=False, byte_offset=0, auto_cast_uint=False, **job_kwargs_local)

    # Move metadata
    files_list = glob.glob(original_path.as_posix() + '/**/**.meta', recursive=True)
    for file in files_list:
        file = Path(file)
        shutil.copy(file, output_path_file/file.parts[-1])
        
    print ('Altering metadata')
    
    correct_metadata(output_path_file, 
                     rec_name, 
                     rec_ap_new,
                     rec_lf_new, 
                     rec_mic_new,
                     time_range)
    
    readme_path = output_path_file / 'readme.txt'
    with open(readme_path, 'w') as fp:
                   fp.write(f'This recording is not the original, but a chunked version from {t0} to {t1} seconds') 
    
    
    
    
