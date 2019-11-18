from allfiles import all_files_in

# Use full absolute path here:
path = "/home/vitek/Vitek/Projects_local_for_ubuntu/video_grabber_python/"
path = "/media/vitek/6C969016968FDEC8/Users/vitek-ntb-win/Desktop/NEW_GENS/000-pgan-cctv_prague_Ruzyne_512_30k-preset-v2-1gpu-fp32"
#path = "/media/vitek/6C969016968FDEC8/Users/vitek-ntb-win/Desktop"

duration = 0.1

files = all_files_in(path)
print("Found", len(files), "images!")
files.sort()
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
subprocess.call(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list_files.txt', 'output.avi'])
