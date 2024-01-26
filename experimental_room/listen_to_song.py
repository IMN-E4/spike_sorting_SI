from pathlib import Path
import spikeinterface.full as si
import numpy as np
import ephyviewer
from ephyviewer.myqt import QT
import soundfile as sf

# def open_data_from_rec_system(recording_folder, rec_system):




def open_and_play_audio(file_name, output_path='/home/eduarda/Desktop'):

    file_name = Path(file_name)
    output_path = Path(output_path)

    if 'nidq' in file_name.stem:
        recording_nidq = si.SpikeGLXRecordingExtractor(file_name.parent, stream_id='nidq') # microphone
        trace = recording_nidq.get_traces(channel_ids=['nidq#XA0'])

    print(trace)
    sf.write(output_path/f'{file_name.stem}.wav', trace, int(recording_nidq.get_sampling_frequency()))

    print('Done') 
    # sd.play(data=trace, samplerate=recording_nidq.get_sampling_frequency())

       


def select_folder_and_open_file():
    app = ephyviewer.mkQApp()
    dia = QT.QFileDialog(fileMode=QT.QFileDialog.AnyFile, acceptMode=QT.QFileDialog.AcceptOpen)
    dia.setViewMode(QT.QFileDialog.Detail)
    if dia.exec_():
        file_name = dia.selectedFiles()[0]
    else:
        return   
    
    print(file_name)
    open_and_play_audio(file_name)


if __name__ == '__main__':
    select_folder_and_open_file()
    
    
