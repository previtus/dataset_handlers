# Dataset tools
Small scripts+tricks to assemble+visualize your datasets fast.
We have two versions right now - 1.) making [a video](https://github.com/previtus/dataset_handlers/blob/master/output_timeOrder.avi) and 2.) making a [t-SNE grid](https://github.com/previtus/dataset_handlers/blob/master/example-tSNE-grid.jpg).


# Video preview of the dataset

## Installation

This code uses and requires ffmpeg to be installed.

On Mac use: `brew install ffmpeg`

On Linux use: 
```
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```

On Windows: follow a guide (maybe [this one](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/))

Test your installation by running this in the terminal:
`ffmpeg -version`

## Using the code

Open the file `ffmpeg_video.py` and change the paths so they point where you have the images:

It is this line:
```
images_path = "/home/vitek/Projects/dataset_handlers/DATASETS/inputs_v4" << REPLACE THIS WHOLE PATH
```

PS: The code will look for images (jpg or png) in this directory and all folders under it.

After editing the path run:

```
python ffmpeg_video.py
```

If everything went well, you'll have your own version of output_timeOrder.avi as the output

### What could go wrong:

If the folder doesn't have any images or the images are not jpg or png. If the ffmpeg is not setup correctly.

