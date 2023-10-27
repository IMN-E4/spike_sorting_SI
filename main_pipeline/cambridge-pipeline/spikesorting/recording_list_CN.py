"""
Here we store all recordings per system
"""

__author__ = "Eduarda Centeno & Samuel Garcia"
__contact__ = "teame4.leblois@gmail.com"
__date__ = "2023/09/1"
__status__ = "Production"


# bird_name, session_name, time_range, depth_range

recording_list = [
    # ('b13g13', '08122022', [0,10*60], None)
    ('Test_Data_troubleshoot', # implant name
     '2023_08_24_12_06_07', # rec name
     101, # node
     1, # experiment number
     [0,300], # time_range
     None, # depth_range
     False #drift correction
     ),
     ('Test_Data_troubleshoot', # implant name
     '2023_08_24_12_06_07', # rec name
     101, # node
     1, # experiment number
     None, # time_range
     [0,200], # depth_range
     False #drift correction
     ),
       ('Test_Data_troubleshoot', # implant name
     '2023_08_24_12_06_07', # rec name
     101, # node
     1, # experiment number
     [0,300], # time_range
     [0,200], # depth_range
     False #drift correction
     ),
     
]