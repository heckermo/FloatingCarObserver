"""
This script is intended to compare the different detection modeling techniques
"""
import os
import logging
import time
import torch
import sys
import traci
sys.path.append('/home/jeremias/Carla_Sumo_detector/D3-Detection/algorithms/monocon-pytorch/utils')
from SUMO_detector_plus.utils.nn_utils.nn_dataset import EmulationDatasetGenerator
from Co_Simulation.tools.callback import CoSimulationCallback
from Co_Simulation.co_simulation_synchonization import SimulationRunner
from Co_Simulation.tools.fco_inference.detector_inference import CoSimulationDetector
from transforms.default_transforms import Normalize, Pad, ToTensor
import shutil
from Co_Simulation.tools.utils import get_config, config_to_dict
from generate_kitti_dataset import DatapointGeneratorCallback
from typing import Callable
from SUMO_detector_plus.detector import detector_factory
from SUMO_detector_plus.utils.raytracing_utils.raytracing import parse_polygons_from_xml

map_to_polygons = {
    "KIVI_Twin": 'polygons_KIVI.add.xml',
    "Town01": 'polygons_Town01.add.xml',
    "Town02": 'polygons_Town02.add.xml',
    "Town03": 'polygons_Town03.add.xml',
}

class DetectionModelingComparisonCallback(CoSimulationCallback):
    def __init__(self, sumo_carla_detector: CoSimulationDetector, raytracing_detector: Callable, emulation_detector: Callable):
        self.sumo_carla_detector = sumo_carla_detector
        self.raytracing_detector = raytracing_detector
        self.emulation_detector = emulation_detector
        self.detection_results = {}

    def on_simulation_step(self, simulation, step, sensor_data, sensor_manager) -> None:
        timestep = traci.simulation.getTime()
        self.detection_results[timestep] = {'sumo_carla_detections': {}}
        for fco, data in sensor_data.items():
            detector_results = self.sumo_carla_detector.detect(
                current_rgb_results=data['current_rgb_results'],
                sensor_data=data['fco_sensor_data'][fco],
                sensor_config = simulation.sensor_config,
                fco = fco,
                simulation = simulation,
                sumo_carla_mapping = simulation.synchronization.synchronization.sumo_carla_idmapping,
                sensor_manager = sensor_manager
            )
            self.detection_results[timestep]['sumo_carla_detections'][fco] = detector_results['detected']

        # Run the 2D raytracing detector
        raytracing_detections = self.raytracing_detector.detect(sensor_data.keys())
        self.detection_results[timestep]['raytracing_detections'] = raytracing_detections

        # Run the emulation detector
        emulation_detections = self.emulation_detector.detect(sensor_data.keys())
        self.detection_results[timestep]['emulation_detections'] = emulation_detections

        self.get_current_accuracy()
    
    def get_current_accuracy(self):
        """
        Get the accuracy of the current detections based on the sumo_carla detections as ground truth
        """
        intersection_storage = {'raytracing': [], 'emulation': []}
        len_sumo_carla_detections = []
        for timestep in self.detection_results.keys():
            len_detections = 0
            carla_raytracing_intersection_count = 0
            carla_emulation_intersection_count = 0
            for fco in self.detection_results[timestep]['sumo_carla_detections'].keys():
                if fco not in self.detection_results[timestep]['raytracing_detections'] or fco not in self.detection_results[timestep]['emulation_detections']:
                    continue
                sumo_carla_detections = self.detection_results[timestep]['sumo_carla_detections'][fco]
                raytracing_detections = self.detection_results[timestep]['raytracing_detections'][fco]
                emulation_detections = self.detection_results[timestep]['emulation_detections'][fco]

                len_detections += len(sumo_carla_detections)

                carla_raytracing_intersection = set(sumo_carla_detections).intersection(set(raytracing_detections))
                carla_emulation_intersection = set(sumo_carla_detections).intersection(set(emulation_detections))
            
                carla_raytracing_intersection_count += len(carla_raytracing_intersection)
                carla_emulation_intersection_count += len(carla_emulation_intersection)
            
            intersection_storage['raytracing'].append(carla_raytracing_intersection_count)
            intersection_storage['emulation'].append(carla_emulation_intersection_count)
            len_sumo_carla_detections.append(len_detections)
        
        # Calculate the accuracy i.e. total intersection / total sumo_carla detections
        accuracy_raytracing = sum(intersection_storage['raytracing']) / sum(len_sumo_carla_detections)
        accuracy_emulation = sum(intersection_storage['emulation']) / sum(len_sumo_carla_detections)

        print(f'Accuracy for raytracing: {accuracy_raytracing}')
        print(f'Accuracy for emulation: {accuracy_emulation}')



def main():
    config = os.path.join('configs', 'co-simulation.yaml')
    c = config_to_dict(config)
    detector_type = 'monocon'
    weights_path = '/home/jeremias/OpenPCDet/output/home/jeremias/OpenPCDet/tools/cfgs/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_28.pth' if detector_type == 'openpcdet' else '/home/jeremias/Carla_Sumo_detector/checkpoints/epoch_190.pth'
    sys.path.append('/home/jeremias/sumo_detector/SUMO_detector_plus') # Add the path to the SUMO detector dir where the 
    emulation_model_path = 'ViTEncoderDecoder_openpcdet' if detector_type == 'openpcdet' else 'ViTEncoderDecoder_monocon'
    # Set up logging
    log_file = os.path.join('log_comparison.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f'Starting comparison at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    # Initialize the detector with the trained 3d object detector model
    sumo_carla_detector = CoSimulationDetector(
        detector_type = detector_type,
        weights_path = weights_path,
        cfg_file = "/home/jeremias/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml",
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        visualize=False
    )

    poly_path = os.path.join('Co_Simulation', 'sumo_files', map_to_polygons[c['map_name']])
    building_polygons = parse_polygons_from_xml(poly_path)

    raytracing_detector = detector_factory('raytracing', building_polygons=building_polygons, use_pure_traci = True)

    emulation_detector = detector_factory('nn', building_polygons=building_polygons, model_path=emulation_model_path, use_pure_traci = True)

    # Initialize teh comparison callback
    comparison_callback = DetectionModelingComparisonCallback(
        sumo_carla_detector=sumo_carla_detector,
        raytracing_detector=raytracing_detector,
        emulation_detector=emulation_detector
    )
    callbacks = [comparison_callback]

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