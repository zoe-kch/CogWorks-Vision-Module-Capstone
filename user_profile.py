from . import database # remove later
from typing import Union, Sequence
import numpy as np

class Profile:
    def __init__(self, name, descriptors: np.ndarray):
        
        self.name = name
        self.descriptors = descriptors # an array, shape (N, 512)
        # self.mean_descriptor = get_mean_descriptor() # placeholder
        database[name] = self # adds to database

    def __repr__(self):
        return self.name
        
    def __str__(self):
        return self.name, self.descriptors

    def remove(self):
        del database[self.name]

    def add(self, name, new_descriptors: np.ndarray):
        self.descriptors += new_descriptors
