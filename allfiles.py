import os

def all_files_in(path = '.', file_types = [".jpg", ".png"]):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            for check_type in file_types:
              if check_type in file:
                  #if True:
                  files.append(os.path.join(r, file))
                  break

    for f in files:
        print(f)

    return files