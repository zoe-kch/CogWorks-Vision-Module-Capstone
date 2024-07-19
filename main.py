"""
*Here is where all the functions will come together, with the interface
*Face detection model is defined here too.
"""
from facenet_models import FacenetModel
from user_profile import *

import skimage.io as io # reading an image file in as a numpy array
import numpy as np
import pickle # for database
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
import sklearn
from camera import take_picture

from pathlib import Path
import os
from typing import Union, List

# this will download the pretrained weights for MTCNN and resnet
# (if they haven't already been fetched)
# which should take just a few seconds
model = FacenetModel()
threshold = None

# Initialize database
image_path = Path("imgs")
database = {}

def find_threshold(name: str):

    # might have to modify indexing based on how db/profile are setup
    
    # basically threshold is that for this current label/profile we are checking
    # it is the dist of the furthest description vector to the mean of this profile 
    # plus the mean of half the dist of that furthest vector to any other given label's mean vector
    # idk this might be a stupid way to do it and im happy to change it

    """dists_to_mean1 = {}

    dists_to_all_means = []    

    for vector in database[profile1]:
        dist = cos_dist(vector, profile1.mean_descriptor)
        dists_to_mean1[dist] = vector
    
    furthest = max(dists_to_mean1)
    furthest_vector = dists_to_mean1[furthest]
    """

    global database

    dists_to_mean = {} # dis: vector
    profile = database[name]
    mean_vector = profile.mean_descriptor

    for vector in profile.descriptors: # shape (N, 512)
        dists_to_mean[cos_dist(mean_vector, vector)] = vector

    # Identify the furthest distance to the mean
    furthest = max(dists_to_mean)
    furthest_vector = dists_to_mean[furthest]

    # Calculate errors as distance between furthest descriptor and other profiles
    errors = []

    for profile in database.values():
        dist = cos_dist(furthest_vector, profile.mean_descriptor)
        errors.append(dist)

    threshold = np.mean(np.array(errors))+ furthest

    return threshold # returns a float


def initialize_database():
    global database
    global image_path
    global threshold
    ret_descriptors, ret_rgb_img, imgs = [], [], []

    for image_dir in os.listdir(image_path):
        profile = None
        for i, image in enumerate(os.listdir(f"{image_path}//{image_dir}")):
            full_img_path = Path(image_path / image_dir / image)
            
            print(full_img_path)
            img = image_to_rgb(full_img_path)
            img_rgb = image_to_rgb(full_img_path)

            boxes = detect_faces(img_rgb)
            print(f"The boxes of the image: {boxes}")

            if boxes: #* check if boxes are empty or not
                descriptor_vector = get_descriptors(img_rgb, boxes)

                if profile == None: # check if profile is None, then create profile
                    profile = Profile(str(image_dir), descriptors=descriptor_vector)
                else: # add profile when profile exist
                    profile.add_descriptors(descriptor_vector)

                database[profile.name] = profile
                ret_descriptors.append(descriptor_vector)
                ret_rgb_img.append(img_rgb)
                imgs.append(img)


        if threshold is None and boxes:
            threshold = find_threshold(str(image_dir))

    return ret_descriptors, ret_rgb_img, imgs


def add_descriptor_vectors_to_database(descriptor_vectors: np.ndarray, names: List[str]):
    """descriptors_vectors is a shape (N, 512) array and names is a length N list."""

    assert len(descriptor_vectors) == len(names), f"length of descriptor_vectors and names do not match | length of vectors: {len(descriptor_vectors)}, length of names: {len(names)}"

    for name, descriptor_vector in zip(names, descriptor_vectors):
        if name in database:
            profile = database[name]
            profile.add_descriptors(descriptor_vector[np.newaxis, :])
        else:
            profile = Profile(name, descriptors=descriptor_vector[np.newaxis, :])
            database[profile.name] = profile


def image_to_rgb(image_to_rgb_path, max_width=500):
    """# shape-(Height, Width, Color)
    print(f"{image_to_rgb_path}: image path when passed to image_to_rgb")
    image = io.imread(str(image_to_rgb_path)).astype(np.uint8) # had to change it to np.float32 bc I was getting problems without it
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
        
    return image.astype(np.uint8)"""
    from PIL import Image, ImageOps
    image = Image.open(image_to_rgb_path)
    image = ImageOps.exif_transpose(image)  # Correct the image orientation based on EXIF data
    width, height = image.size
    if width > max_width:
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        image = image.resize((new_width, new_height))
    return np.array(image).astype(np.uint8)

def cos_dist(a,b):
    out = 1 - np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)) # changed from @ to * because a and b have different float types 
    # assert out == cosine_similarity(a, b) # checking to see if these functions are equivalent
    return out

# check if cos_dist == cosine_sim
from sklearn.metrics.pairwise import cosine_similarity
def cosine_sim(descriptors1: np.ndarray, descriptors2: np.ndarray):
    # assert cos_dist(descriptors1, descriptors2) == cosine_similarity(descriptors1, descriptors2) # checking to see if these functions are equivalent
    return cosine_similarity(descriptors1, descriptors2)


def match(descriptor_vector: np.ndarray, threshold: float):
    """all_dists = {}
    
    for name,d in database.items():
        # check is np.mean(axis=0) returns average vector im so confused :3
        all_dists[name] = cos_dist(descriptor_vector, np.mean(d,axis=0))
        
    # there might be a more efficient way to do this
    min = np.min(all_dists) # min dist
    min_name = min(all_dists, key=all_dists.get) # name pertaining to min dist
    
    if min < threshold:
        Profile.add(min_name,descriptor_vector) # adds pic to profile
        return str(min_name) # assuming min_name is a profile stored in database
    else:
        print('unknown')
        Profile.add("Unknown",descriptor_vector)"""
    
    ### descriptor_vectors.shape should be (1, 512)

    global database

    lowest_dist_and_profile = [3, None] # set to 3 because the bound is 2
    for profile in database.values():
        mean_discriptor_vector = profile.mean_descriptor # a 1-D array
        if cos_dist(descriptor_vector[0], mean_discriptor_vector) < lowest_dist_and_profile[0]:
            lowest_dist_and_profile = [cos_dist(descriptor_vector, mean_discriptor_vector), profile]
            print(f"lowest_dist_and_profile for {profile.name}: {lowest_dist_and_profile}")

    if lowest_dist_and_profile[0] < threshold:
        if descriptor_vector not in lowest_dist_and_profile[1].descriptors:
            lowest_dist_and_profile[1].add_descriptors(descriptor_vector)
        return lowest_dist_and_profile[1].name
    else:
        return "Unknown"

def detect_faces(image, threshold=.6):
    """
    It takes in an image and return a list of
    boxes that are already filtered respect to the threshold
    """
    global model

    boxes, probabilities, landmarks = model.detect(image)
    # just assuming that this is the threshold
    
    # these are a list of boxes filtered after the threshold
    filtered = [box for box, prob in zip(boxes, probabilities) if prob > threshold] #####
    
    return filtered

def get_descriptors(image, boxes):
    return model.compute_descriptors(image, boxes)


def draw_boxes(image, boxes, names, camera=False):
    """
    It just draw boxes
    """
    if not camera:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        pass
    print(f"Image type: {type(image)}, Image shape: {image.shape}")
    for box, name in zip(boxes, names):
        start_x, start_y = int(box[0]), int(box[1])
        end_x, end_y = int(box[2]), int(box[3])
        color = (0, 255, 25) if name != "Unknown" else (255, 25, 0)
        image = cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
        image = cv2.putText(image, name, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image
        

def main(descriptors, threshold, rgb_pic, img, camera=True):
    """
    This will be the main function for the program, where it will call all the necessary functions
    and also displays interface of some sort.

    descriptors: np.ndarray(), shape (N, 512)
    threshold: float, global variable for matching
    rgb_pic: np.ndarray: shape

    """
    if camera:
        video_capture = cv2.VideoCapture(0)
        #raw_capture = cv2.VideoCapture(0)
        WIDTH, HEIGHT = 640, 360
        video_capture.set(3, WIDTH)
        video_capture.set(4, HEIGHT)
        
        while True:
            # capture the video
            ret, frame = video_capture.read()

            # mirror the frame for perfect coordination
            if ret:
                frame = cv2.flip(frame, 1)
                
                frame = np.array(frame).astype(np.uint8)
                boxes = detect_faces(frame)
                names = [match(descriptor[np.newaxis, :], threshold) for descriptor in descriptors]
                
                boxes_drawn = draw_boxes(frame, boxes, names, camera=True)
                cv2.imshow('Some really cool and interesting name for this project', boxes_drawn)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        
    else:

        boxes = detect_faces(rgb_pic)
        names = [match(descriptor[np.newaxis, :], threshold) for descriptor in descriptors]
        
        boxes_drawn = draw_boxes(img, boxes, names)

        cv2.imshow('Some really cool and interesting name for this project', boxes_drawn)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    




if __name__ == '__main__':
    vars = initialize_database()
    descriptors, rgb_pics, imgs = vars
    print(f"Returns of initialize_database exist? : {descriptors and rgb_pics}")
    print(f"Length of descriptors passed to main function: {len(descriptors)} | length of images: {len(rgb_pics)}")
    for des, rgbpic, img in zip(descriptors, rgb_pics, imgs):
        main(des, threshold, rgbpic, img)
