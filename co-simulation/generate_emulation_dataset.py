"""
This script is intended to generate a dataset for the emulation of different 3D detection algorithms. It generates a dataset in the style such that the train_emulation.py 
script in SUMO detector plus can be used. For this, we co-simulate the SUMO and Carla simulation; equip the vehicles with a specific sensor setup to record individual KITTI datapoints.
Using the trained 3D detection algorithms (trained with synthetic KITTI datasets), we can then determine which vehicles are detectable in the SUMO simulation, i.e., give us the binary prediction
label for training the emulation network.
"""
import logging
import os
import shutil
import sys
import time
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Co_Simulation.co_simulation_synchonization import SimulationRunner
from Co_Simulation.tools.callback import CoSimulationCallback
from Co_Simulation.tools.fco_inference.detector_inference import CoSimulationDetector
from Co_Simulation.tools.utils import config_to_dict, get_config
from SUMO_detector_plus.utils.nn_utils.nn_dataset import EmulationDatasetGenerator
from generate_kitti_dataset import DatapointGeneratorCallback
from transforms.default_transforms import Normalize, Pad, ToTensor

map_to_polygons = {
    "KIVI_Twin": 'polygons_KIVI.add.xml',
    "Town01": 'polygons_Town01.add.xml',
    "Town02": 'polygons_Town02.add.xml',
    "Town03": 'polygons_Town03.add.xml',
}

class EmulationDatasetGeneratorCallback(CoSimulationCallback):
    def __init__(self, dataset_generator: EmulationDatasetGenerator, detector: CoSimulationDetector):
        self.dataset_generator = dataset_generator
        self.detector = detector

    def on_simulation_step(self, simulation, step, sensor_data, sensor_manager) -> None:
            for fco, data in sensor_data.items():
                detector_results = self.detector.detect(
                    current_rgb_results=data['current_rgb_results'],
                    sensor_data=data['fco_sensor_data'][fco],
                    sensor_config = simulation.sensor_config,
                    fco = fco,
                    simulation = simulation,
                    sumo_carla_mapping = simulation.synchronization.synchronization.sumo_carla_idmapping,
                    sensor_manager = sensor_manager
                )

                self.dataset_generator.get_data(
                    simulation_time=simulation.simulation_time,
                    ego_id=fco,
                    detected_vehicles=detector_results['detected'],
                    max_distance=50
                ) 


def main():
    config = os.path.join('configs', 'co-simulation.yaml')
    c = config_to_dict(config)
    town = c['map_name']
    simulation_steps_between_recordings = c['recording_frequency']
    detector_type = 'monocon'
    filename = f'{detector_type}_{town}'
    weights_path = '/home/jeremias/OpenPCDet/output/home/jeremias/OpenPCDet/tools/cfgs/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_28.pth' if detector_type == 'openpcdet' else '/home/jeremias/Carla_Sumo_detector/checkpoints/epoch_190.pth'


    dataset_path = os.path.join('emulation_datasets', filename)

    # Check if directory exists
    if os.path.isdir(dataset_path):
        shutil.rmtree(dataset_path) # TODO: Remove this line
        # raise Exception(f'Directory "{dataset_path}" already exists. Please choose a different name.')
    os.makedirs(dataset_path)

    # Set up logging
    log_file = os.path.join(dataset_path, 'log_dataset.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f'Starting dataset generation at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    # Initialize the emulation dataset generator
    dataset_generator = EmulationDatasetGenerator(filename, sumo_connector="traci", intersection_name=town, buildings_path=os.path.join('Co_Simulation', 'sumo_files', map_to_polygons[town]))

    # Initialize the detector with the trained 3d object detector model
    detector = CoSimulationDetector(
        detector_type = detector_type,
        weights_path = weights_path,
        cfg_file = "/home/jeremias/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml",
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        visualize=False, 
        iou_threshold=0.7
    )

    # Initialize the callbacks
    emulation_dataset_generator = EmulationDatasetGeneratorCallback(
        dataset_generator=dataset_generator,
        detector=detector,
    )
    callbacks = [emulation_dataset_generator]

    # Initialize the Co-Simulation runner
    simulation_runner = SimulationRunner(
        config_file=os.path.join('configs', 'co-simulation.yaml'),
        sensor_config_file=os.path.join('configs', 'sensor_setups.yaml'),
        callbacks=callbacks
    )

    # Start the simulation
    simulation_runner.run()

if __name__ == "__main__":
    main()