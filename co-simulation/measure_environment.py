from typing import Any, Dict
from Co_Simulation.co_simulation_synchonization import SimulationRunner
from Co_Simulation.tools.callback import CoSimulationCallback
from Co_Simulation.tools.kitti_style import save_kitti_data
from Co_Simulation.tools.opv2v_style import save_opv2v_data
from Co_Simulation.tools.utils import create_logger
from Co_Simulation.tools.utils import get_config, config_to_dict
import time
import yaml
import os
import shutil
import sys
import pickle
import traci

class MapCreationCallback(CoSimulationCallback):
    """
    Callback to generate a dataset to measure the environment.
    """
    def __init__(self, dataset_config: str,  measure_path: str = None):
        self.dataset_config = config_to_dict(dataset_config)
        self.num_datapoints = 1000
        self.measure_path = measure_path
        if self.measure_path is not None:
            os.makedirs(self.measure_path, exist_ok=True)
        
        self.recording_infos = {'timestep': [], 'carla_position': [], 'carla_fco': None, 'carla_rotation': [], 'sumo_position': [], 'sumo_rotation': [], 'fco': None, 'sumo_carla_offset': None}
        self.counter = 0

    def on_simulation_step(self, simulation, step, sensor_data: Dict[str, Any], sensor_manager) -> None:
        """
        Handle the processing and saving of sensor data.

        Args:
            fco (str): Identifier for the vehicle or object of interest.
            simulation_time (float): Current simulation time.
            fco_sensor_data (Dict[str, Any]): Sensor data collected from the vehicle.
            current_rgb_results (Any): Results from RGB sensor processing.
        """
        simulation_time = simulation.simulation_time
        self.counter += 1
        for fco, data in sensor_data.items(): 
            fco_sensor_data = data['fco_sensor_data']
            current_rgb_results = data['current_rgb_results']
            save_kitti_data(
                    self.measure_path,
                    simulation_time,
                    simulation.fco_cameramanager_mapping[fco],
                    fco,
                    fco_sensor_data[fco],
                    current_rgb_results
                )

            # get the position of lidar attached to the vehicle
            carlaid = simulation.synchronization.synchronization.sumo2carla_ids[fco]
            carla_world = simulation.synchronization.carla_simulation.world
            location = carla_world.get_actor(carlaid).get_transform().location
            rotation = carla_world.get_actor(carlaid).get_transform().rotation
            yaw, pitch, roll = rotation.yaw, rotation.pitch, rotation.roll
            x,y,z = location.x, location.y, location.z
            self.recording_infos['timestep'].append(simulation_time)
            self.recording_infos['carla_position'].append((x,y,z))
            self.recording_infos['carla_rotation'].append((yaw, pitch, roll))
            self.recording_infos['carla_fco'] = fco
            location = traci.vehicle.getPosition(fco)
            x,y,z = location[0], location[1], 0
            rotation = traci.vehicle.getAngle(fco)
            self.recording_infos['sumo_position'].append((x,y,z))
            self.recording_infos['sumo_rotation'].append(rotation)
            self.recording_infos['fco'] = fco
            self.recording_infos['sumo_carla_offset'] = simulation.synchronization.sumo_simulation.get_net_offset()

            # save the recording infos to pickle
            with open(os.path.join(self.measure_path, 'recording_infos.pkl'), 'wb') as f:
                pickle.dump(self.recording_infos, f)
            
            if self.counter == 50:
                pass
                #sys.exit()

    
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
    dataset_name = f'{c["map_name"]}'
    sensor_config = os.path.join('configs', 'sensor_setups.yaml')
    dataset_config = os.path.join('configs', 'kitti_generation.yaml')
    # create the dir for logging the data
    dataset_path = os.path.join('measurements', dataset_name)
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
    datapoint_generator_callback = MapCreationCallback(
        dataset_config = dataset_config,
        measure_path = dataset_path
    )

    # Initialize the simulation runner
    simulation_runner = SimulationRunner(
        config_file = config,
        sensor_config_file = sensor_config,
        logger = logger,
        callbacks = [datapoint_generator_callback]
    )

    simulation_runner.run()

    
    
