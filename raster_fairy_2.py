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

images, pca_features = pickle.load(open('./results/features_caltech101.p', 'r'))

for i, f in zip(images, pca_features)[0:5]:
    print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... "%(i, f[0], f[1], f[2], f[3]))

num_images_to_plot = 1000

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
for img, x, y in tqdm(zip(images, tx, ty)):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

plt.figure(figsize = (16,12))
imshow(full_image)

full_image.save("example-tSNE-animals1k.png")

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
imshow(grid_image)

grid_image.save("example-tSNE-grid-animals.jpg")

"""
totalDataPoints=1000
radius,adjustmentFactor,count = rasterfairy.getBestCircularMatch(totalDataPoints)
print "The smallest circle that can hold",totalDataPoints,"has a radius of",radius,"and will fit",count,"points"

#the adjustmentFactor is a value between 0.0 and 1.0 that controls
#if a pixel that is on the edge of the circle will be included or not
arrangement = rasterfairy.getCircularArrangement(radius,adjustmentFactor)
rasterMask = rasterfairy.arrangementToRasterMask(arrangement)
grid_xy,(emb_width,emb_height) = rasterfairy.transformPointCloud2D(tsne,target=rasterMask,autoAdjustCount = True)
width = emb_width*tile_width
print(width)
height = emb_height*tile_height
print(height)
print(tile_width,tile_height)
max_dim = 100

print(full_width)
print(full_height)

steps =50

for i in range(steps):
    print(i)
    pctg = pytweening.easeInOutSine(float(i)/(steps-1))
    
    print(pctg)
    
    full_image = Image.new('RGBA', (width, height))

    for img, x,y, grid_pos in tqdm(zip(images, tx, ty, grid_xy)):
        #print(grid_pos)
        idx_x, idx_y = grid_pos
        new_xpos = (width-max_dim)*x*(1-pctg)+pctg*(tile_width * idx_x)
        new_ypos = (height-max_dim)*y*(1-pctg)+pctg*(tile_height * idx_y)
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

    fig = plt.figure(figsize=(16.0,16.0))
    fig.tight_layout()
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(0,0,1,1)

    
    plt.axis("off")
    plt.margins(0.0)
    

    fig.patch.set_facecolor('black')
    
    #plt.rcParams['axes.facecolor']='black'
    plt.rcParams['savefig.facecolor']='black'

    #ax.set_facecolor('black')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)
    #ax.axes.set_xlim([0.0,dim])
    #ax.axes.set_ylim([0.0,dim])
    #ax.invert_yaxis()
    imshow(full_image)
    clear_output(wait=True)
    plt.savefig('tmpdir/tsne_circle_anim'+"%03d" % (i,)+'.png',bbox_inches="tight")
    plt.show()
"""