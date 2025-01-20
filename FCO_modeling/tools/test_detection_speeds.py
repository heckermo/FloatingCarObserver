import time
import os
import shutil
import logging
from typing import List
import tqdm
import libsumo as traci
import pickle
import sys
import itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.raytracing_utils.raytracing import parse_polygons_from_xml

import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg

from detector import detector_factory  # Import the detector factory function

from utils.cv_utils.detect_cv import *
from utils.file_utils import delete_all_items_in_dir
from utils.sumo_utils import configure_sumo, update_sumocfg, update_modal_split, variate_traffic
from utils.tfco_utils.tfco_dataset import TfcoDatasetGenerator
from utils.fco_utils import FcoMonitor
import json

from configs.config_FCO import MULTI_FCO

"""
This script utilizes different detector classes to detect traffic participants based on a specified penetration rate of FCOs at an area of interest and thereby
generates a dataset for the tfco model. The script is designed to be run in a loop to generate multiple datasets with different traffic conditions. 
"""

def main():
    detection_methods = ['cv', 'nn', 'raytracing']
    start_time = 36000
    num_steps = 250
    fco_penetration_rate = 0.10
    model_path = 'ViTEncoderDecoder_intersection_testemulation'
    center_point = (6728.8, 5315.08)
    radius = 250
    rou_file = 'motorized_routes_2020-09-16_24h_dist.rou.xml'
    sumo_file = '24h_sim.sumocfg'   

    # Load and parse the building polygons needed for the detectors
    POLY_PATH='/home/jeremias/sumo_detector/SUMO_detector_plus/sumo_sim/ingolstadt.poly.xml'
    building_polygons = parse_polygons_from_xml(POLY_PATH)
    building_polygons = [polygon for polygon in building_polygons if np.linalg.norm(np.array(polygon.exterior.coords[0]) - np.array(center_point)) < radius]

    kwargs = {
        'cv': {},
        'nn': {'model_path': model_path, 'building_polygons': building_polygons},
        'raytracing': {'building_polygons': building_polygons}
    }

    # Use kwargs.values() to pass the corresponding dictionary to each method
    detectors = [detector_factory(method, **kwarg) for method, kwarg in zip(detection_methods, kwargs.values())]

    # Delete all items in the tmp_3d folder
    delete_all_items_in_dir('tmp_3d')

    results = {detector: {'timestep': [], 'detected_vehicles': [], 'undetected_vehicles': [], 'fcos': [], 'duration': []} for detector in detection_methods}

    inflow_time = 600
    # Initialize the FCO moitor class
    fcos = FcoMonitor(fco_penetration_rate)

    # Create variation in the traffic if desired and start sumo
    variate_traffic('multifco_vehicle_routes.rou.xml')
    update_sumocfg(os.path.join('sumo_sim', sumo_file), None, ['multifco_vehicle_routes.rou.xml'], start_time)
    sumo_cmd = configure_sumo(sumo_file)
    traci.start(sumo_cmd)

    # allow the traffic to flow in the simulation
    for _ in range(inflow_time):
        traci.simulationStep()
        
    for step in tqdm.tqdm(range(num_steps)):
        # Update the current FCOs
        fcos.update()

        for detector, detector_type in zip(detectors, detection_methods):
            t = time.time()
            detected_vehicles, relevant_vehicles = run_detection(fcos.fco_vehicles, detector, center_point, radius)
            duration = time.time() - t
            results[detector_type]['timestep'].append(step)
            results[detector_type]['detected_vehicles'].append(detected_vehicles)
            results[detector_type]['undetected_vehicles'].append([v for v in relevant_vehicles if v not in detected_vehicles])
            results[detector_type]['fcos'].append([v for v in fcos.fco_vehicles if v in relevant_vehicles]) 
            results[detector_type]['duration'].append(duration)

        traci.simulationStep()
        with open('detection_comparison_results.pkl', 'wb') as f:
            pickle.dump(results, f)
traci.close()

def run_detection(fco_vehicles: List[str], detector, center_of_interest, radius: int):
    relevant_vehicles = [v for v in traci.vehicle.getIDList() if np.linalg.norm(np.array(traci.vehicle.getPosition(v)) - np.array(center_of_interest)) < radius+100]

    # Get the detected vehicles by the detector from the fco_vehicles that are currently in the relevant vehiles
    detected_relevant_vehicles = []
    fco_relevant_vehicles = []
    for fco_vehicle in fco_vehicles:
        if fco_vehicle in relevant_vehicles:
            fco_relevant_vehicles.append(fco_vehicle)
    detected_vehicles = detector.detect(fco_relevant_vehicles)
    detected_vehicles_list = list(set(itertools.chain(*detected_vehicles.values())))
    detected_relevant_vehicles = [v for v in detected_vehicles_list if v in relevant_vehicles]
    return detected_relevant_vehicles, relevant_vehicles



if __name__ == '__main__':
    main()