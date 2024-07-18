"""

*Here is where all the functions will come together, with the interface
*Face detection model is defined here too.


"""

from facenet_models import FacenetModel
import skimage.io as io # reading an image file in as a numpy array
import numpy as np
import pickle # for database
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
from database import database
from user_profile import Profile

#for vscode, remove later
'''
from . import database
from . import Profile
'''

# this will download the pretrained weights for MTCNN and resnet
# (if they haven't already been fetched)
# which should take just a few seconds
model = FacenetModel()

def image_to_rgb(image_path):
    # shape-(Height, Width, Color)
    image = io.imread(str(image_path)).astype(np.float32) # had to change it to np.float32 bc I was getting problems without it
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
    return image.astype(np.float32)

def cos_dist(a,b):
    return 1 - np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)) # changed from @ to * because a and b have different float types 

# check if cos_dist == cosine_sim
from sklearn.metrics.pairwise import cosine_similarity
def cosine_sim(descriptors1, descriptors2):
    return cosine_similarity(descriptors1, descriptors2)


def connected_components():
    pass


def propogate_label():
    pass


def match(descriptors, threshold):
    all_dists = {}
    
    for name,d in database.items():
        # check is np.mean(axis=0) returns average vector im so confused :3
        all_dists[name] = cos_dist(descriptors, np.mean(d,axis=0))
        
    # there might be a more efficient way to do this
    min = np.min(all_dists) # min dist
    min_name = min(all_dists, key=all_dists.get) # name pertaining to min dist
    
    if min < threshold:
        Profile.add(min_name,descriptors) # adds pic to profile
        return str(min_name) # assuming min_name is a profile stored in database
    else:
        print('unknown')
        Profile.add("Unknown",descriptors)


def detect_faces(image, threshold=.9):
    """
    It takes in an image and return a list of
    boxes that are already filtered respect to the threshold
    """
    boxes, probabilities, landmarks = model.detect(image)
    # just assuming that this is the threshold
    
    # these are a list of boxes filtered after the threshold
    filtered = [box for box, prob in zip(boxes, probabilities) if prob > threshold]
    
    return filtered

def get_descriptors(image, boxes):
    return model.compute_descriptors(image, boxes)

def draw_boxes(image, boxes, name):
    """
    It just draw boxes
    """
    for box, name in zip(boxes, name):
        start_x, start_y = (int(box[0])), (int(box[1]))
        end_x, end_y = (int(box[2])), (int(box[3]))
        color = (0, 255, 25) if name != "Unknown" else (255, 25, 0)
        image = cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
        image = cv2.putText(image, name, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

def main():
    """
    This will be the main function for the program, where it will call all the necessary functions
    and also displays interface of some sort
    """
    

    pic = "imgs/newton.png"
    rgb_pic = image_to_rgb(pic)
    
    boxes = detect_faces(rgb_pic)
    descriptors = get_descriptors(rgb_pic, boxes)
    names = [match(descriptor, 0.6) for descriptor in descriptors]
    
    boxes_drawn = draw_boxes(rgb_pic, boxes, names)
    cv2.imshow('Some really cool and intersting name for this project', boxes_drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main()
