# thanks to repo https://github.com/bapoczos/ArtML/blob/master/Embeddings/ ...

import os
import random
import cPickle as pickle
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

images_path = "/home/vitek/Projects/dataset_handlers/DATASETS/inputs_final_session1"
max_num_images = 10000
name_result_dumpfile = './features/features_session1_final.p'



model = keras.applications.VGG16(weights='imagenet', include_top=True)
model.summary()

# get_image will return a handle to the image itself, and a numpy array of its pixels to input the network
def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]

images = [f for f in images if os.path.getsize(f) != 0]


print("keeping %d images to analyze" % len(images))

features = []
for image_path in tqdm(images):
    img, x = get_image(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

features = np.array(features)
n_comp = min(300, len(features))
pca = PCA(n_components=n_comp)
pca.fit(features)
pca_features = pca.transform(features)


pickle.dump([images, pca_features], open(name_result_dumpfile, 'wb'))
