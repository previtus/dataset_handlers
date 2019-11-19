import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt

import cPickle as pickle
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm
from IPython.display import clear_output


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



name_result_dumpfile = './features/features_art1k.p'
tsne_plot_name = "example-tSNE-1k.png"
grid_plot_name = "example-tSNE-grid.jpg"
num_images_to_plot = 1000
#num_images_to_plot = 50
duration = 0.2
output_video_name = 'output_tsneOrder_.avi'
output_video_name = 'output_tsneOrder_TESTDIST.avi'



images, pca_features = pickle.load(open(name_result_dumpfile, 'r'))

for i, f in zip(images, pca_features)[0:5]:
    print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... "%(i, f[0], f[1], f[2], f[3]))


if len(images) > num_images_to_plot:
    sort_order = sorted(random.sample(xrange(len(images)), num_images_to_plot))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]

X = np.array(pca_features)
tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

width = 4000
height = 3000
max_dim = 100

full_image = Image.new('RGBA', (width, height))
print("placing into the tsne image")
for img, x, y in tqdm(zip(images, tx, ty)):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

plt.figure(figsize = (16,12))
#imshow(full_image)

full_image.save(tsne_plot_name)

import rasterfairy

# nx * ny = 1000, the number of images
nx = 40
ny = 25

# assign to grid
grid_assignment = rasterfairy.transformPointCloud2D(tsne, target=(nx, ny))
print(grid_assignment[0].shape)

tile_width = 72
tile_height = 56

full_width = tile_width * nx
full_height = tile_height * ny

print(full_width)
print(full_height)

aspect_ratio = float(tile_width) / tile_height

grid_image = Image.new('RGB', (full_width, full_height))

print("placing into the grid image")
for img, grid_pos in tqdm(zip(images, grid_assignment[0])):
    #print(grid_pos)
    idx_x, idx_y = grid_pos
    x, y = tile_width * idx_x, tile_height * idx_y
    tile = Image.open(img)
    tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
    if (tile_ar > aspect_ratio):
        margin = 0.5 * (tile.width - aspect_ratio * tile.height)
        tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
    else:
        margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
        tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
    tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
    grid_image.paste(tile, (int(x), int(y)))

plt.figure(figsize = (16,12))
#imshow(grid_image)

grid_image.save(grid_plot_name)

# ==========================================================
# ALSO TO VIDEO (from the same TSNE)
# ... using line by line right now ...

import math

images_by_order = {}
distances = {}
for img, grid_pos in tqdm(zip(images, grid_assignment[0])):
    idx_x, idx_y = grid_pos
    idx_i = int(ny*idx_y + idx_x)

    distance_from_lefttop = math.sqrt(idx_y*idx_y + idx_x*idx_x)

    print(grid_pos, "=", idx_i)

    images_by_order[idx_i] = img
    distances[idx_i] = distance_from_lefttop

files = []
keys_sorted = sorted(images_by_order.keys())
# by dist >>
#keys_sorted = sorted(distances.values()) ### not werks?
print("keys_sorted", keys_sorted)

for key in keys_sorted:
    files.append(images_by_order[key])


with open('list_files_tsned.txt', 'w') as the_file:
    for file in files:
        str_line = "file '"+file+"'\nduration "+str(duration)+"\n"
        the_file.write(str_line)

import os
import subprocess
#os.chdir('?')
#subprocess.call(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list_files.txt', 'output.avi'])
# -vf scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:1024:1024:color=black,setsar=1 $file; done
subprocess.call(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', 'list_files_tsned.txt', '-vf', 'scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1024:1024:1024:color=black,setsar=1', output_video_name])


# ===================================================================
# Another animation - tsne to grid
# (optional) and slow
"""
import pytweening
video_path = 'tsne_anim_video.mp4'


width = full_width  # 2880 it could be any number
height = full_height  # 1400 it could be any number
max_dim = 100

print(full_width)
print(full_height)

steps=50

for i in range(steps):
    print(i)
    pctg = pytweening.easeInOutSine(float(i) / (steps - 1))

    print(pctg)

    full_image = Image.new('RGBA', (width, height))

    for img, x, y, grid_pos in tqdm(zip(images, tx, ty, grid_assignment[0])):
        # print(grid_pos)
        idx_x, idx_y = grid_pos
        new_xpos = (width - max_dim) * x * (1 - pctg) + pctg * (tile_width * idx_x + width / 2 - full_width / 2)
        new_ypos = (height - max_dim) * y * (1 - pctg) + pctg * (tile_height * idx_y + height / 2 - full_height / 2)
        tile = Image.open(img)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))
        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        full_image.paste(tile, (int(new_xpos), int(new_ypos)))

    fig = plt.figure(figsize=(16.0, 16.0))
    fig.tight_layout()
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(0, 0, 1, 1)

    plt.axis("off")
    plt.margins(0.0)

    fig.patch.set_facecolor('black')

    # plt.rcParams['axes.facecolor']='black'
    plt.rcParams['savefig.facecolor'] = 'black'

    # ax.set_facecolor('black')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # ax.axes.set_xlim([0.0,dim])
    # ax.axes.set_ylim([0.0,dim])
    # ax.invert_yaxis()
    #imshow(full_image)
    clear_output(wait=True)
    plt.savefig('tmpdir/tsne_anim' + "%03d" % (i,) + '.png', bbox_inches="tight")
    plt.show()

import os
frames_dir = './tmpdir'

cmd = 'ffmpeg -i %s/tsne_anim%%03d.png -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" %s' % (frames_dir, video_path)
print(cmd)
os.system(cmd)
"""