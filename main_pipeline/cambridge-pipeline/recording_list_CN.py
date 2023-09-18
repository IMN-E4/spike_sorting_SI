# bird_name, session_name, time_range, depth_range

# recording_list = [
#     # ('b13g13', '08122022', [0,10*60], None)
#     ('Test_Data_troubleshoot', '2023-08-23_15-56-05', None, None, False),
# ]



_recording_list = [
    {'bird_name': 'Test_Data_troubleshoot',
     'session_name': '2023-08-23_15-56-05',
     'time_slice': None,
     'channel_slice': None,
     'do_drift_correction': False,
     'node': None
     },
     
    {'bird_name': 'Test_Data_troubleshoot',
     'session_name': '2023-08-24_12-06-07',
     },
    {'bird_name': 'Test_Data_troubleshoot2',
     'session_name': '2023-08-24_12-06-07',
     },
]

import pandas as pd

# main_index = pd.DataFrame(recording_list)
# print(main_index)


def get_main_index():
    main_index = pd.DataFrame(_recording_list)

    return main_index


if __name__ == '__main__':
    print(get_main_index())
