### This program will create the database for this project. ###
### It's only purpose is to create the pickle file `database.pkl`; if that already exists, don't run this program-- just update in main.py as needed ###

from pathlib import Path
import os
import numpy as np
import pickle
from user_profile import *
from main import *

database = {}
image_path = Path("imgs/")

def initialize_database():
    global database
    global image_path

    for image_dir in os.listdir(image_path):
        profile = None
        for image in os.listdir(f"{image_path}//{image_dir}"):
            full_img_path = Path(image_path / image_dir / image)
            
            print(full_img_path)
            img_rgb = image_to_rgb(full_img_path)

            boxes = np.array(detect_faces(img_rgb))
            print(f"The boxes of the image: {boxes}")

            if boxes.any(): #* check if boxes are empty or not
                print(f"Image shape: {img_rgb.shape}")
                descriptor_vector = get_descriptors(img_rgb, np.array(boxes)[np.newaxis, :][0]) #boxes[0] because there should only be one box for one face, 
                # descriptor_vector.shape = (N, 512)

                if profile == None: # check if profile is None, then create profile
                    profile = Profile(str(image_dir), descriptors=descriptor_vector)
                else: # add descriptor_vector when profile exist
                    profile.add_descriptors(descriptor_vector)

        database[profile.name] = profile

    with open('database.pkl', 'wb') as f:
        pickle.dump(database, f)


initialize_database()
