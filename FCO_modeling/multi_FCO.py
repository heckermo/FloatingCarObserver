import time
import os
import shutil
import logging
from typing import List
from pathlib import Path
import tqdm
import libsumo as traci
import pickle
import sys
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
import yaml

"""
This script utilizes different detector classes to detect traffic participants based on a specified penetration rate of FCOs at an area of interest and thereby
generates a dataset for the tfco model. The script is designed to be run in a loop to generate multiple datasets with different traffic conditions. 
"""

def main(config_path: str = 'configs/config_FCO.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    detection_method = config['detection_method']
    start_time_range = config['start_times']
    explicit_start_times = config['explicit_start_times']
    num_steps = config['num_steps']
    num_loops = config['num_loops']
    fco_penetration_rate = config['fco_penetration_rate']
    filename = config['filename']
    dataset_name = config['dataset_name']
    add_info = config['additional_data_information']
    model_path = config['model_path']
    center_point = config['center_point']
    radius = config['radius']
    rou_file = config['rou_file']
    sumo_file = config['sumo_file']
    polygons = config['polygons']
    set_seed = config['set_seed']
    inflow_time = config['inflow_time']

    base_root = Path(__file__).resolve().parents[2]

    if set_seed is not None:#
        np.random.seed(set_seed)

    data_fco_path = os.path.join(base_root, 'data', 'tfco_datasets', str(add_info+'_'+filename))#
    if os.path.isdir(data_fco_path):
        shutil.rmtree(data_fco_path)
    os.makedirs(data_fco_path)

    with open(os.path.join(data_fco_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    logger_filename = os.path.join(data_fco_path, 'log_dataset.log')
    logging.basicConfig(filename=logger_filename, level=logging.INFO)
    logging.info('Starting simulation at ' + time.strftime("%H:%M:%S", time.localtime()))

    building_polygons = parse_polygons_from_xml(os.path.join(base_root, polygons))
    building_polygons = [polygon for polygon in building_polygons if np.linalg.norm(np.array(polygon.exterior.coords[0]) - np.array(center_point)) < radius]

    detector = detector_factory(detection_method, model_path=model_path, building_polygons=building_polygons)

    dataset = TfcoDatasetGenerator(base_root, filename, center_point, radius, add_info, dataset_name, detector=detector)#
    fcos = FcoMonitor(fco_penetration_rate)

    delete_all_items_in_dir('tmp_3d')

    start_times = list(explicit_start_times) if explicit_start_times else [np.random.randint(start_time_range[0], start_time_range[1]) for _ in range(num_loops)]
    
    for loop, start_time in enumerate(start_times):
        logging.info(f'Starting simulation loop {loop} at time {start_time}')

        variate_traffic(os.path.join(base_root, rou_file))
        update_sumocfg(os.path.join(base_root, sumo_file), None, rou_file, start_time)

        sumo_cmd = configure_sumo(os.path.join(base_root, sumo_file))
        traci.start(sumo_cmd)

        for _ in range(inflow_time):
            traci.simulationStep()
            
        for step in tqdm.tqdm(range(num_steps)):
            fcos.update()
            dataset.get_data(loop=loop, time=traci.simulation.getTime(), fco_vehicles=fcos.fco_vehicles)
            traci.simulationStep()
        
        dataset.save()
    traci.close()


if __name__ == '__main__':
    main()
