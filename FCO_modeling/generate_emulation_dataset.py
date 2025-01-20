import libsumo as traci
import time
import sys
import logging
import shutil
import subprocess
import numpy as np
import tqdm
import random
import sys
import os
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cv_utils.detect_cv import *
from configs.config_simulation import TRAFFIC
from utils.sumo_utils import configure_sumo, update_modal_split, update_sumocfg
from utils.assign_fco import filter_intersection_fcos
from utils.nn_utils.nn_dataset import EmulationDatasetGenerator
from detector import detector_factory
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def main(config_path: str, intersection_mode: bool=True, num_processes: int=0):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if not intersection_mode:
        raise ValueError("Currently only intersection_mode=True is supported since parsing the complete polygons in the network is very slow.")

    fco_penetration = 1.0
    simulation_steps_between_recordings = config['dataset_general']['simulation_jumps']
    filename = config['dataset_general']['dataset_filename']
    config_file = config['dataset_general']['config_file']
    intersections = config['dataset_intersection']  # list of intersections to generate the dataset for
    polygon_path = config['dataset_general']['poly_path']

    dataset_path = os.path.join('emulation_datasets', filename)

    # Check if directory exists
    if os.path.isdir(dataset_path):
        raise Exception(f'Directory "{dataset_path}" already exists. Please choose a different name.')
    os.makedirs(dataset_path)

    # Set up logging
    log_file = os.path.join(dataset_path, 'log_dataset.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f'Starting dataset generation at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    # Update simulation configuration
    update_sumocfg(os.path.join('sumo_sim', config_file), config['dataset_general']['net_file'],
                   [config['dataset_general']['vehicle_route_file'],
                    config['dataset_general']['bike_route_file'],
                    config['dataset_general']['pedestrian_route_file']],
                   config['dataset_general']['start_time'],
                   config['dataset_general']['end_time'])

    update_modal_split(os.path.join('sumo_sim', config['dataset_general']['vehicle_route_file']), TRAFFIC['MODAL_SPLIT'])

    # Start the simulation
    sumo_cmd = configure_sumo(config_file)
    traci.start(sumo_cmd)
    for _ in range(360):
        traci.simulationStep()  # Allow the vehicles to flow into the system

    intersection_names = [intersection['intersection_name'] for intersection in intersections]
    intersection_centers = [intersection['intersection_center'] for intersection in intersections]
    intersection_radiuss = [intersection['intersection_radius'] for intersection in intersections]
    intersection_dataset_generators = [EmulationDatasetGenerator(filename, buildings_path=polygon_path, intersection_name=intersection_name) for intersection_name in intersection_names]

    logging.info('Pre-filtering buildings...')
    [d._pre_filter_buildings(intersection, intersection_radius) for d, intersection, intersection_radius in zip(intersection_dataset_generators, intersection_centers, intersection_radiuss)]

    logging.info('Starting dataset generation...')
    last_pbupdate = time.time()
    current_total = 0
    with tqdm.tqdm(total=config['dataset_general']['dataset_size']) as pbar:
        while current_total < config['dataset_general']['dataset_size']:

            if num_processes <= 0:
                for dataset_generator, intersection_center, intersection_radius in zip(intersection_dataset_generators, intersection_centers, intersection_radiuss):
                    run_intersection_detection_step(dataset_generator, intersection_center, intersection_radius, fco_penetration)
            else:
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = [
                        executor.submit(
                            run_intersection_detection_step,
                            dataset_generator,
                            intersection_center,
                            intersection_radius,
                            fco_penetration
                        ) for dataset_generator, intersection_center, intersection_radius in zip(
                            intersection_dataset_generators, intersection_centers, intersection_radiuss)
                    ]
                    for future in futures:
                        future.result()

            for _ in range(simulation_steps_between_recordings):
                traci.simulationStep()

            if time.time() - last_pbupdate > 10:
                current_total = get_total_size(os.path.join('emulation_datasets', filename))
                pbar.update(current_total - pbar.n)
                last_pbupdate = time.time()

    traci.close()

    # Write information about the dataset to the logging
    logging.info(f'Finished dataset generation at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    for file in [f for f in os.listdir(dataset_path) if f.endswith('dataset.pkl')]:
        logging.info(f'{file}: {len(pd.read_pickle(os.path.join(dataset_path, file)))} entries')

def run_intersection_detection_step(dataset_generator, intersection_center, intersection_radius, fco_penetration):
    detector = detector_factory(mode='raytracing', building_polygons=dataset_generator.building_polygons)

    dataset_generator.load_data()

    all_vehicles = list(traci.vehicle.getIDList())

    vehicles_to_process = [v for v in all_vehicles if np.linalg.norm(np.array(traci.vehicle.getPosition(v)) - np.array(intersection_center)) < intersection_radius]

    # Further reduce the number of vehicles to process by the fco penetration
    vehicles_to_process = random.sample(vehicles_to_process, int(fco_penetration * len(vehicles_to_process)))

    for v in vehicles_to_process:
        if traci.vehicle.getVehicleClass(v) != 'passenger':
            return

        detected_vehicles = detector.detect([v])[v]
        dataset_generator.get_data(traci.simulation.getTime(), v, detected_vehicles)

    dataset_generator.store_data()

def get_total_size(filename):
    # Get all the ..._dataset.pkl files in the directory
    files = [f for f in os.listdir(filename) if f.endswith('_dataset.pkl')]
    total_length = 0
    for f in files:
        df = pd.read_pickle(os.path.join(filename, f))
        total_length += len(df)
    return total_length


if __name__ == '__main__':
    main(config_path="configs/config_dataset.yaml", intersection_mode=True, num_processes=25)
