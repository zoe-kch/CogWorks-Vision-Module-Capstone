### This program will create and upkeep the database for this project. ###
from pathlib import Path
import os

image_path = Path("imgs/")

for image in os.listdir(image_path):
    ### something something something