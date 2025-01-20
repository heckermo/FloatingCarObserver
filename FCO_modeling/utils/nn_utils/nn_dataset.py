from utils.visualize import visualize_frame
from utils.raytracing_utils.raytracing import parse_polygons_from_xml
from utils.polygon_to_tensor import create_bev_tensor
from utils.emulation_visualization import visualize_frame

import PIL
import libsumo as traci
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from typing import List
import threading


import math
import os
import time
from typing import List, Tuple

class EmulationDatasetGenerator:
    def __init__(self, filename: str, buildings_path: str, intersection_name: str, visualize_datapoint: bool=False):
        self.filename = filename
        self.intersection_name = intersection_name
        self.building_polygons = parse_polygons_from_xml(buildings_path)
        self.dataset = pd.DataFrame(columns=['id', 'vector', 'bev_pointer', 'detected_label', 'type']) # image_pointer will point tot the id in the bev_data
        self.bev_data = pd.DataFrame(columns=['id', 'building_polygons', 'vehicle_infos', 'ego_angle']) # ego angle is needed to rotate the poygons to the ego vehicle's orientation
        self.visualize_datapoint = visualize_datapoint

    def _pre_filter_buildings(self, centerpoint: Tuple[float, float], radius: float):
        self.building_polygons = [polygon for polygon in self.building_polygons if np.linalg.norm(np.array(polygon.exterior.coords[0]) - np.array(centerpoint)) < radius]

    def get_data(self, simulation_time: float, ego_id: str, detected_vehicles: List[str], max_distance: int=50):
        # Get the position of the ego vehicle
        x, y = traci.vehicle.getPosition(ego_id)
        alpha = traci.vehicle.getAngle(ego_id)

        # generate the ego info
        ego_info = {ego_id: {
            'position': [x, y],
            'angle': alpha,
            'type': traci.vehicle.getTypeID(ego_id),
            'width': traci.vehicle.getWidth(ego_id),
            'length': traci.vehicle.getLength(ego_id),
        }}

        # Filter the relevant vehicles based on the distance
        relevant_vehicles = [v for v in traci.vehicle.getIDList() if v != ego_id and np.linalg.norm(np.array(traci.vehicle.getPosition(v)) - np.array([x, y])) < max_distance]
        relevant_vehicles = {v: {
            'position': traci.vehicle.getPosition(v),
            'angle': traci.vehicle.getAngle(v),
            'type': traci.vehicle.getTypeID(v),
            'width': traci.vehicle.getWidth(v),
            'length': traci.vehicle.getLength(v),
            'vector': np.array(traci.vehicle.getPosition(v)) - np.array([x, y]),
            'detected_label': 1 if v in detected_vehicles else 0,
            'id': f'{ego_id}__{str(simulation_time).replace(".", "_")}__{v}'
        } for v in relevant_vehicles}

        # Filter the relevant buildings based on the distance
        building_polygon_dict = {f'building_{i}': polygon for i, polygon in enumerate(self.building_polygons) if np.linalg.norm(np.array(polygon.exterior.coords[0]) - np.array([x, y])) < max_distance}

        # Merge all polygons and save datapoint to the bev_data
        unique_identifier = np.random.randint(1000, 9999) # generate a random number to make the id unique also across different dataset generation runs
        bev_id = f'{ego_id}__{str(simulation_time).replace(".", "_")}__{self.intersection_name}__{unique_identifier}'
        # Create a copy of relevant_vehicles without 'vector', 'detected_label', 'id' keys
        vehicle_bev_data = {k: {key: value for key, value in v.items() if key not in ['vector', 'detected_label', 'id']} for k, v in relevant_vehicles.items()}
        vehicle_bev_data = {**vehicle_bev_data, **ego_info}

        self.bev_data = pd.concat([self.bev_data, pd.DataFrame([{
            'id': bev_id,
            'building_polygons': building_polygon_dict,
            'vehicle_infos': vehicle_bev_data,
            'ego_x': x,
            'ego_y': y,
            'ego_angle': alpha, 
        }])], ignore_index=True)

        # Save the datapoints to the dataset
        for v in relevant_vehicles:
            new_data = pd.DataFrame([{
                'id': relevant_vehicles[v]['id'],
                'vector': relevant_vehicles[v]['vector'],
                'bev_pointer': bev_id,
                'detected_label': relevant_vehicles[v]['detected_label'],
                'type': relevant_vehicles[v]['type']
            }])
            self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)
        
                # Visualize the data
        if self.visualize_datapoint:
            bev = create_bev_tensor(building_polygons=building_polygon_dict, vehicle_infos=vehicle_bev_data, 
                              vehicle_representation='box', image_size=224, x_offset=x, y_offset=y, rotation_angle=0, radius_covered=50)
            for v in relevant_vehicles.values():
                vector = torch.from_numpy(v['vector'].copy()).float()
                vector_distance = torch.norm(vector).item() 
                print(f'vector has length {vector_distance}')
                print(f'label is {v["detected_label"]}')
                visualize_frame(bev, vector, 50, 'test_bev.png')
                print('Visualizing the BEV data')

    def store_data(self):
        # Ensure the save directory exists
        os.makedirs(os.path.join('emulation_datasets', self.filename), exist_ok=True)

        # Save dataset and bev_data as Parquet files
        self.dataset.to_pickle(os.path.join('emulation_datasets', self.filename, f'{self.intersection_name}_dataset.pkl'))
        self.bev_data.to_pickle(os.path.join('emulation_datasets', self.filename, f'{self.intersection_name}_bev_data.pkl'))

        #print(f'Dataset and BEV data saved to emulation_datasets/{self.filename}')

    def load_data(self):
        if os.path.exists(os.path.join('emulation_datasets', self.filename, f'{self.intersection_name}_dataset.pkl')):
            self.dataset = pd.read_pickle(os.path.join('emulation_datasets', self.filename, f'{self.intersection_name}_dataset.pkl'))
        if os.path.exists(os.path.join('emulation_datasets', self.filename, f'{self.intersection_name}_bev_data.pkl')):
            self.bev_data = pd.read_pickle(os.path.join('emulation_datasets', self.filename, f'{self.intersection_name}_bev_data.pkl'))
