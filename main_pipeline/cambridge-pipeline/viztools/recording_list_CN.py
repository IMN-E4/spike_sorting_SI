"""
Here we store all recordings per system
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/09/1"
__status__ = "Production"


# bird_name, session_name, time_range, depth_range

# recording_list = [
#     # ('b13g13', '08122022', [0,10*60], None)
#     ('Test_Data_troubleshoot', '2023-08-23_15-56-05', None, None, False),
# ]



_recording_list = [
    {'bird_name': 'Test_Data_troubleshoot',
     'session_name': '2023-08-23_15-56-05',
     'brain_area': 'AreaX_LMAN',
     'node': 'Record Node 101',
     'experiment_number': 'experiment1'
     },
     {'bird_name': 'Test_Data_troubleshoot',
     'session_name': '2023-08-23_17-22-38',
     'brain_area': 'AreaX_LMAN',
     'node': 'Record Node 101',
     'experiment_number': 'experiment1'
     },
     {'bird_name': 'Test_Data_troubleshoot',
     'session_name': '2023-08-24_12-06-07',
     'brain_area': 'AreaX_LMAN',
     'node': 'Record Node 101',
     'experiment_number': 'experiment1'
     },
     {'bird_name': 'Test_Data_troubleshoot',
     'session_name': '2023-08-24_12-06-07',
     'brain_area': 'AreaX_LMAN',
     'node': 'Record Node 101',
     'experiment_number': 'experiment2'
     }
]

import pandas as pd

def get_main_index():
    main_index = pd.DataFrame(_recording_list)

    return main_index


if __name__ == '__main__':
    print(get_main_index())
