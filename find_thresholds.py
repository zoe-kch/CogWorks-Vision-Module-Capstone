### This program is meant to find and save the optimal threshold value for this application. Only run it to
### update this value and `database.pkl` exists. Stored in `threshold.txt.` ###

from main import cos_dist
import numpy as np
import pickle
from user_profile import *

database = {}
with open('database.pkl', 'rb') as f:
    database = pickle.load(f)

def find_threshold(name: str):

    # might have to modify indexing based on how db/profile are setup
    
    # basically threshold is that for this current label/profile we are checking
    # it is the dist of the furthest description vector to the mean of this profile 
    # plus the mean of half the dist of that furthest vector to any other given label's mean vector
    # idk this might be a stupid way to do it and im happy to change it

    """`name` is the name of the key in the database dictionary you want to base the threshold off
        off. Saves calculated threshold value in `threshold.txt`."""

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
        errors.append(dist / 2) ### to update threshold value, change this line to something else !!!!!!

    threshold = np.mean(np.array(errors))+ furthest # type(threshold) = float

    # saves threshold in `threshold.txt` file
    with open('threshold.txt', 'w') as f:
        f.write(str(threshold))


find_threshold("Edwardia")
