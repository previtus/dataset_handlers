import os
from allfiles import all_files_in
import random

# Use full absolute path here:
#images_path = "/media/vitek/6C969016968FDEC8/Users/vitek-ntb-win/Desktop"
images_path = "/home/vitek/Projects/dataset_handlers/DATASETS/inputs_final_session1"

heicfiles = all_files_in(images_path, file_types=[".heic",".HEIC"])
print("Found", len(heicfiles), "heic files!")

for f in heicfiles:
    print("heif-convert '"+f+"' '"+f+".jpg'")