import pandas as pd
import libsumo as traci
from detector import Detector
from typing import List, Tuple
import numpy as np
import os
import itertools

class TfcoDatasetGenerator():
    def __init__(self, filename: str, center_of_interest: Tuple[float, float], radius: int, detector: Detector):
        self.filename = os.path.join('tfco_datasets', filename, 'dataset.pkl')
        self.dataset = pd.DataFrame(columns=['id', 'loop', 'timestep', 'vehicle_information'])

        self.detector = detector

        self.center_of_interest = center_of_interest
        self.radius = radius
    
    def get_data(self, loop: int, time: int, fco_vehicles: List[str]):
        # we add a buffer to the radius to since we also want to utelize the detected vehilces that are detected by fcos that are not in the radius
        relevant_vehicles = [v for v in traci.vehicle.getIDList() if np.linalg.norm(np.array(traci.vehicle.getPosition(v)) - np.array(self.center_of_interest)) < self.radius+100]

        # Get the detected vehicles by the detector from the fco_vehicles that are currently in the relevant vehiles
        detected_relevant_vehicles = []
        fco_relevant_vehicles = []
        for fco_vehicle in fco_vehicles:
            if fco_vehicle in relevant_vehicles:
                fco_relevant_vehicles.append(fco_vehicle)
        detected_vehicles = self.detector.detect(fco_relevant_vehicles)
        detected_vehicles_list = list(set(itertools.chain(*detected_vehicles.values())))
        detected_relevant_vehicles = [v for v in detected_vehicles_list if v in relevant_vehicles]

        # get the vehicle information for all the relevant vehiccles
        vehicle_information = {v: {
            'position': traci.vehicle.getPosition(v),
            'angle': traci.vehicle.getAngle(v),
            'type': traci.vehicle.getTypeID(v),
            'width': traci.vehicle.getWidth(v),
            'length': traci.vehicle.getLength(v),
            'detected_label': 1 if v in detected_relevant_vehicles else 0,
            'fco_label': 1 if v in fco_relevant_vehicles else 0
        } for v in relevant_vehicles}

        # add datapoint to the dataset
        datapoint = pd.DataFrame({
            'id': f'{loop}__{str(time).replace(".", "_")}',
            'loop': loop,
            'timestep': time,
            'vehicle_information': [vehicle_information] # wrap the dict to a singel cell in the new df
        })

        self.dataset = pd.concat([self.dataset, datapoint])
    
    def save(self):
        self.dataset.to_pickle(self.filename)


        
    