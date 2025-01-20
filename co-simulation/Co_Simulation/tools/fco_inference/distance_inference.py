import traci
import numpy as np
import os
import sys

def run_distance_inference(fco:str, distance=50): 
    distance_inference_dict = {}
    try:
        ego_pos_x, ego_pos_y = traci.vehicle.getPosition(fco)
    except:
        distance_inference_dict = {'detected': [], 'undetected_vehicles': []}
        return distance_inference_dict
    all_vehicles = traci.vehicle.getIDList()
    detected_vehicles = []
    undetected_vehicles = []
    for vehicle in all_vehicles:
        if vehicle != fco:
            pos_x, pos_y = traci.vehicle.getPosition(vehicle)
            calculated_distance = np.sqrt((ego_pos_x - pos_x)**2 + (ego_pos_y - pos_y)**2)
            if calculated_distance < distance:
                detected_vehicles.append(vehicle)
            else:
                undetected_vehicles.append(vehicle)

    distance_inference_dict['detected'] = detected_vehicles
    distance_inference_dict['undetected_vehicles'] = undetected_vehicles
    return distance_inference_dict