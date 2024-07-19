from . import database # remove later
from typing import Union, Sequence
import numpy as np

class Profile:
    def __init__(self, name, descriptors: np.ndarray):
        
        self.name = name
        self.descriptors = descriptors # an array, shape (N, 512)
        self.mean_descriptor = np.mean(self.descriptors, axis=0) # might not work we'll see
        database[name] = self # add to database, need to change this

    def __repr__(self):
        return self.name
        
    def __str__(self):
        return self.name, self.descriptors

    def remove(self):
        del database[self.name] # remove from database, need to change this

    def add_descriptors(self, new_descriptors: np.ndarray):
        """Adds a shape (N, 512) array of descriptors to this array (M, 512) of descriptor vectors store a (M+N, 512) array of descriptor vectors."""
        self.descriptors = np.concatenate(self.descriptors, new_descriptors)

