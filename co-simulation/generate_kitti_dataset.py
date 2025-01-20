import os
import shutil
import sys
import time
import yaml
from typing import Any, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Co_Simulation.co_simulation_synchonization import SimulationRunner
from Co_Simulation.tools.callback import CoSimulationCallback
from Co_Simulation.tools.kitti_style import save_kitti_data
from Co_Simulation.tools.opv2v_style import save_opv2v_data
from Co_Simulation.tools.utils import create_logger, config_to_dict


class DatapointGeneratorCallback(CoSimulationCallback):
    """
    Callback for generating KITTI and OPV2V datapoints for the current vehicles that send sensor data."""
    def __init__(self, dataset_config: str,  kitti_dataset_path: str = None, opv2v_dataset_path: str = None, num_datapoints: int = 1000):
        self.dataset_config = config_to_dict(dataset_config)
        self.num_datapoints = 1000
        self.kitti_dataset_path = kitti_dataset_path
        if self.kitti_dataset_path is not None:
            os.makedirs(self.kitti_dataset_path, exist_ok=True)

        self.opv2v_dataset_path = opv2v_dataset_path
        if self.opv2v_dataset_path is not None:
            os.makedirs(self.opv2v_dataset_path, exist_ok=True)

        self.recording_steps = 0

    def on_simulation_step(self, simulation, step, sensor_data: Dict[str, Any], **kwargs) -> None:
        """
        Handle the processing and saving of sensor data.

        Args:
            fco (str): Identifier for the vehicle or object of interest.
            simulation_time (float): Current simulation time.
            fco_sensor_data (Dict[str, Any]): Sensor data collected from the vehicle.
            current_rgb_results (Any): Results from RGB sensor processing.
        """
        simulation_time = simulation.simulation_time
        for fco, data in sensor_data.items():
            self.recording_steps += 1   
            fco_sensor_data = data['fco_sensor_data']
            current_rgb_results = data['current_rgb_results']
            if self.kitti_dataset_path is not None:
                save_kitti_data(
                    self.kitti_dataset_path,
                    simulation_time,
                    simulation.fco_cameramanager_mapping[fco],
                    fco,
                    fco_sensor_data[fco],
                    current_rgb_results
                )

                          

            if self.opv2v_dataset_path is not None:
                save_opv2v_data(
                    self.opv2v_dataset_path,
                    simulation_time,
                    simulation.synchronization.carla_simulation.world,
                    simulation.synchronization.synchronization.sumo_carla_idmapping,
                    simulation.fco_cameramanager_mapping,
                    fco,
                    fco_sensor_data[fco],
                    current_rgb_results
                )
                
        dirs = os.listdir(self.kitti_dataset_path)
        total_items = 0
        for item in dirs:
            if os.path.isdir(os.path.join(self.kitti_dataset_path, item)):
                total_items += len(os.listdir(os.path.join(self.kitti_dataset_path, item))) 
        logger.info(f'Processed sensor data for fcos at time {simulation_time} dataset now has {total_items} items')

        if total_items >= self.num_datapoints:
            logger.info('Finished generating dataset')
            sys.exit(0)
    
    def on_simulation_end(self, simulation) -> None:
        """
        Save metadata and mappings after simulation steps.

        Args:
            simulation_time (float): Current simulation time.
            recording_steps (int): Number of recording steps completed.
        """

        # Save SUMO-CARLA ID mapping
        with open(os.path.join(self.kitti_dataset_path, 'sumo_carla_id_mapping.yaml'), 'w') as f:
            yaml.dump(
                self.synchronization.synchronization.sumo_carla_idmapping,
                f,
                default_flow_style=False
            )

        # Update and save metadata
        simulation.meta_data['end_time'] = time.time()
        simulation.meta_data['recording_steps'] = self.recording_steps
        with open(os.path.join(self.kitti_dataset_path, 'meta_data.yaml'), 'w') as f:
            yaml.dump(self.meta_data, f, default_flow_style=False)



if __name__ == "__main__":
    config = os.path.join('configs', 'co-simulation.yaml')
    c = config_to_dict(config)
    dataset_name = f'{c["map_name"]}__{c["weather"]}'
    sensor_config = os.path.join('configs', 'sensor_setups.yaml')
    dataset_config = os.path.join('configs', 'kitti_generation.yaml')
    # create the dir for logging the data
    dataset_path = os.path.join('kitti_datasets', dataset_name)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path) # TODO value error one ready
    os.makedirs(dataset_path)

    # copy the config and sensor config to the dataset folder
    shutil.copy(config, os.path.join(dataset_path, 'co-simulation.yaml'))
    shutil.copy(sensor_config, os.path.join(dataset_path, 'sensor_setup.yaml'))
    shutil.copy(dataset_config, os.path.join(dataset_path, 'kitti_generation.yaml'))

    # create the logger
    logger = create_logger(os.path.join(dataset_path, 'logs.log'))

    # Initialize the callback for the dataset generation
    datapoint_generator_callback = DatapointGeneratorCallback(
        dataset_config = dataset_config,
        kitti_dataset_path = dataset_path
    )

    # Initialize the simulation runner
    simulation_runner = SimulationRunner(
        config_file = config,
        sensor_config_file = sensor_config,
        logger = logger,
        callbacks = [datapoint_generator_callback]
    )

    simulation_runner.run()

    
    
