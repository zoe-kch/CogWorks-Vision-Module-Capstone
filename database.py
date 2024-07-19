### This program will create and upkeep the database for this project. ###
from pathlib import Path
import os
import cv2
import numpy as np
from user_profile import Profile

database = {}


def initialize_database():
    image_path = Path("imgs/")

    for image_path in os.listdir(image_path):
        ### something something something

        img = cv2.imread("imgs/" + image_path)
        img = np.asarray(img)
        descriptor_vector = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(type(descriptor_vector))

        profile = Profile(image_path, descriptor_vector)

        database[profile.__repr__()] = [descriptor_vector]

print(database)

#def add_to_database():
