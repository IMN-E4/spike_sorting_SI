import cv2
import numpy
import glob
from pathlib import Path
import re

path_to_photos = Path('/nas/Neuropixel_Recordings/AreaX-LMAN/Imp_29_11_2022/Recordings/Rec_2_30_11_2022_morning_nap_g0/camera/')
file_extension = '.jpg'
fps = 0.1953 # 1 every 5 s (Approx!!!)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = path_to_photos / 'video.mp4'

files = glob.glob((path_to_photos/f'**{file_extension}').as_posix())

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


img_array = []
for filename in sorted(files, key=numericalSort):
    print(Path(filename).stem)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

