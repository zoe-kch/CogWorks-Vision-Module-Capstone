"""

*Here is where all the functions will come together, with the interface
*Face detection model is defined here too.


"""

from facenet_models import FacenetModel
import skimage.io as io
import numpy as np


# this will download the pretrained weights for MTCNN and resnet
# (if they haven't already been fetched)
# which should take just a few seconds
model = FacenetModel()

def image_to_rgb(image_path):
    # shape-(Height, Width, Color)
    image = io.imread(str(image_path))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
        

def cos_dist(a,b):
    return 1 - np.dot(a,b) / (np.abs(a) @ np.abs(b))


def match(descriptors,threshold):
    all_dists = {}
    
    for name,d in database.items():
        all_dists[name] = cos_dist(descriptors, np.mean(d))
        
    min = np.min(all_dists)
    min_name = min(all_dists, key=all_dists.get)
    
    if min < threshold:
        return str(min_name) # assuming min_name is a profile stored in database
    else:
        print('unknown')


def connected_components():
    pass


def propogate_label():
    pass


pic = "imgs/gauss_train.jpg"
rgb_pic = image_to_rgb(pic)


# detect all faces in an image
# returns a tuple of (boxes, probabilities, landmarks)
# assumes ``pic`` is a numpy array of shape (R, C, 3) (RGB is the last dimension)
#
# If N faces are detected then arrays of N boxes, N probabilities, and N landmark-sets
# are returned.
boxes, probabilities, landmarks = model.detect(rgb_pic)


# Crops the image once for each of the N bounding boxes
# and produces a shape-(512,) descriptor for that face.
#
# If N bounding boxes were supplied, then a shape-(N, 512)
# array is returned, corresponding to N descriptor vectors
descriptors = model.compute_descriptors(rgb_pic, boxes)
