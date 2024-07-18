from . import database # remove later

class Profile:
    def __init__(self, name, descriptors):
        database[name] = database.get(name,[]) + descriptors # adds to database, whether or not name is stored
        
    def __repr__(self):
        return self.name
        
    def __str__(self):
        return self.name, self.descriptors

    def remove(self, name):
        del database[name]
