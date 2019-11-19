import os
from allfiles import all_files_in
import random

# Use full absolute path here:
#path = "/media/vitek/6C969016968FDEC8/Users/vitek-ntb-win/Desktop"
path = "/home/vitek/Datasets/_ART DATASET/images"
#path = "/home/vitek/Generated/progressive_growing_of_gans_2nd"

max_num_files = None
max_num_files = 1000
#max_num_files = 100

duration = 0.2
output_video_name = 'output_timeOrder.avi'


files = all_files_in(path)
print("Found", len(files), "images!")

# Sort by name or by date:
#files.sort()
files.sort(key=lambda x: os.path.getmtime(x))

if max_num_files is not None:
    if max_num_files < len(files):
        files = [files[i] for i in sorted(random.sample(xrange(len(files)), max_num_files))]


# make list formating this:

with open('list_files.txt', 'w') as the_file:
    for file in files:
        str_line = "file '"+file+"'\nduration "+str(duration)+"\n"
        the_file.write(str_line)


# file '/path/to/file1'
# file '/path/to/file2'
# file '/path/to/file3'

# `ffmpeg -f concat -i list_files.txt ... <output>`
import os
import subprocess
#os.chdir('?')
#subprocess.call(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list_files.txt', 'output.avi'])
# -vf scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:1024:1024:color=black,setsar=1 $file; done
subprocess.call(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list_files.txt', '-vf', 'scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:1024:1024:color=black,setsar=1', output_video_name])
