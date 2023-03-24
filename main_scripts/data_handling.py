import spikeinterface.full as si
from datetime import datetime
from params_NP import base_input_folder, base_sorting_cache_folder

def get_spikeglx_folder(implant_name, rec_name):
    spikeglx_folder = base_input_folder / implant_name / "Recordings" / rec_name

    return spikeglx_folder

def get_working_folder(
    implant_name,
    rec_name,
    time_range,
    depth_range,
    time_stamp="default",
):
    """Create working directory

    Parameters
    ----------
    spikeglx_folder: Path
        path to spikeglx folder

    time_range: None or list
        time range to slice recording

    depth_range: None or list
        depth range to slice recording

    time_stamp: str
        time stamp on folder. default = current month

    Returns
    -------
    rec: spikeinterface object
        recording

    working_folder: Path
        working folder
    """

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{rec_name}-full"
        )

    else:
        time_range = tuple(float(e) for e in time_range)

        working_folder = (
            base_sorting_cache_folder
            / implant_name
            / "sorting_cache"
            / f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
        )


    if depth_range is not None:
        print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
        working_folder = working_folder / f"depth_{depth_range[0]}_to_{depth_range[1]}"
    else:
        print(f"Using all channels")
    
    return working_folder


def get_sorting_folder(implant_name, rec_name, time_range, depth_range, time_stamp, sorter_name):

    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        name = f"{time_stamp}-{rec_name}-full"
        
        sorting_clean_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / sorter_name
        )

    else:
        time_range = tuple(float(e) for e in time_range)
        name = f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
        sorting_clean_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / sorter_name
        )

    if depth_range is not None:
        print(f"Depth slicing between {depth_range[0]} and {depth_range[1]}")
        sorting_clean_folder = sorting_clean_folder / f"depth_{depth_range[0]}_to_{depth_range[1]}"
    else:
        print(f"Using all channels")


    # name = working_folder.parts[6]
    # if len(working_folder.parts)>7:
    #     name = working_folder.parts[6]+'/'+ working_folder.parts[7]
    #     print(name)


    # sorting_clean_folder = (
    #         base_input_folder / implant_name / "Sortings_clean" / name / sorter_name
        # )
    return sorting_clean_folder

def get_synchro_file(implant_name, rec_name, time_range, time_stamp):
    if time_stamp == "default":
        time_stamp = datetime.now().strftime("%Y-%m")

    # Time slicing
    if time_range is None:
        name = f"{time_stamp}-{rec_name}-full"
        
        synchro_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / 'synchro')

    else:
        time_range = tuple(float(e) for e in time_range)
        name = f"{time_stamp}-{rec_name}-{int(time_range[0])}to{int(time_range[1])}"
        synchro_folder = (
            base_input_folder / implant_name / "Sortings_clean" / name / 'synchro'
        )
        
    synchro_file = synchro_folder / 'synchro_imecap_corr_on_nidq.json'
        
    return synchro_file



#### Tests

def test_get_sorting_folder():
    sorter_name = 'kilosort2_5'
    implant_name, rec_name, time_range, depth_range, drift, time_stamp = ('Anesth_21_01_2023', 'Rec_21_01_2023_1_g0', None, None, False, "2023-02")
    sorting_clean_folder = get_sorting_folder(implant_name, rec_name, time_range, depth_range, time_stamp, sorter_name)

    print(sorting_clean_folder)


if __name__ == '__main__':
    # select_folder_and_open()
    test_get_sorting_folder()
    