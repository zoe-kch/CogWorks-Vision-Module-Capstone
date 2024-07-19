from facenet_models import FacenetModel
from database import database
import numpy as np
import matplotlib.pyplot as plt
from camera import take_picture

def cos_dist(a,b):
    return 1 - np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_threshold(profile1):

    # might have to modify indexing based on how db/profile are setup
    
    # basically threshold is that for this current label/profile we are checking
    # it is the dist of the furthest description vector to the mean of this profile 
    # plus the mean of half the dist of that furthest vector to any other given label's mean vector
    # idk this might be a stupid way to do it and im happy to change it

    dists_to_mean1 = {}

    dists_to_all_means = []    

    for vector in database[profile1]:
        dist = cos_dist(vector, profile1.mean_descriptor)
        dists_to_mean1[dist] = vector
    
    furthest = max(dists_to_mean1)
    furthest_vector = dists_to_mean1[furthest]
    
    errors = []

    for profile in database:
        dist = cos_dist(furthest_vector, profile.mean_descriptor)
        errors += [dist / 2]

    threshold = np.mean(np.array(errors))+ furthest

    return threshold






model = FacenetModel()

pic1 = take_picture()
pic2 = take_picture()

boxes, probabilities, landmarks = model.detect(pic1)
boxes2, probabilities2, landmarks2 = model.detect(pic2)


descriptor1 = model.compute_descriptors(pic1, boxes)
descriptor2 = model.compute_descriptors(pic2, boxes2)


find_threshold(descriptor1, descriptor2)











