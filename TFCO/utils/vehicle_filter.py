import pandas as pd
import numpy as np 
import math
import random 

def get_nearest_vehicles(vehicle_information, mode, k):

    if mode == "exact" and len(vehicle_information) < k:
        return dict()

    elif mode == "below" and len(vehicle_information) < k:
        return vehicle_information
    
    datapoint = dict()
    distances = list()
    origin = np.array([0, 0])
    
    for vehicle_id, (_, x, y) in vehicle_information.items():
        position = np.array([x, y])
        distance = np.linalg.norm(position - origin)
        distances.append((vehicle_id, distance))

    distances.sort(key=lambda x: x[1])
    distances = distances[:k]

    for vehicle in distances:
        key = vehicle[0]
        value = vehicle_information.get(key)
        datapoint[key] = value

    return datapoint



def get_furthest_vehicles(vehicle_information, mode, k):
  
    if mode == "exact" and len(vehicle_information) < k:
        return dict()

    elif mode == "below" and len(vehicle_information) < k:
        return vehicle_information

    datapoint = dict()
    distances = list()
    origin = np.array([0, 0])
    
    for vehicle_id, (_, x, y) in vehicle_information.items():
        position = np.array([x, y])
        distance = np.linalg.norm(position - origin)
        distances.append((vehicle_id, distance))

    distances.sort(key=lambda x: x[1])
    distances = distances[-k:]

    for vehicle in distances:
        key = vehicle[0]
        value = vehicle_information.get(key)
        datapoint[key] = value

    return datapoint



def get_random_vehicles(vehicle_information, mode, k):
 
    if mode == "exact" and len(vehicle_information) < k:
        return dict()

    elif mode == "below" and len(vehicle_information) < k:
        return vehicle_information
        
    datapoint = dict()
    vehicles = list(vehicle_information.keys())
    random_vehicles_id = random.sample(vehicles, k)

    for key in random_vehicles_id:
        value = vehicle_information.get(key)
        datapoint[key] = value
        
    return datapoint
    

if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be executed directly")