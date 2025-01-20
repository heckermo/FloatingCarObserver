import os
import sys
import time
import logging
import yaml
import pickle
import tqdm
import traci
from typing import Any, Dict, List, Optional

# Add necessary paths to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.expanduser('~'), 'CARLA_Shipping_0.9.14_KIVI', 'LinuxNoEditor', 'PythonAPI'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3D-Detection", "algorithms", "monocon-pytorch"))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

# Import custom modules
from tools.utils import get_config, config_to_dict
from tools.synchronization import SumoCarlaSynchronization
from tools.carla_processes import adapt_carla_map
from co_simulation_visualization import CosimulationVisualization
from tools.sensor_queues import SensorManager, SensorDataProcessor
from tools.sumo_integration.sumo_simulation import get_fco_ids
from sumo_files.mapping import map_to_sumocfg
from tools.callback import CoSimulationCallback


class SimulationRunner:
    def __init__(
        self,
        config_file: str,
        sensor_config_file: str,
        logger: Optional[logging.Logger] = None,
        callbacks: Optional[List[CoSimulationCallback]] = None
    ) -> None:
        """
        Initialize the SimulationRunner by loading configurations and initializing components.

        Args:
            config_file (str): Path to the main configuration YAML file.
            sensor_config_file (str): Path to the sensor setups YAML file.
            kitti_dataset_path (st
        """
        # Load the configuration
        self.config = get_config(config_file)
        self.sensor_config = config_to_dict(sensor_config_file)
        self.meta_data: Dict[str, Any] = {'start_time': time.time()}

        # Initialize internal components and configurations
        self.logger = logger

        # Initialize components
        sumo_cfg_path = os.path.join(os.path.dirname(__file__), 'sumo_files', map_to_sumocfg[self.config.map_name])
        self.fco_ids = get_fco_ids(self.config, sumo_cfg_path)
        self.synchronization = SumoCarlaSynchronization(sumo_cfg_file=sumo_cfg_path, args=self.config)
        # self.visualizer = CosimulationVisualization(self.synchronization)
        self.sensor_manager = SensorManager(
            synchronization=self.synchronization,
            fco_ids=self.fco_ids,
            sensor_config=self.sensor_config,
            max_sensors=self.config.max_sensors if hasattr(self.config, 'max_sensors') else 30
        )
        self.data_processor = SensorDataProcessor(self.synchronization, self.config)

        # Other initializations
        self.fco_sensor_queues = self.sensor_manager.fco_sensor_queues
        self.fco_cameramanager_mapping = self.sensor_manager.fco_cameramanager_mapping

        # Set the CARLA map based on the configuration
        adapt_carla_map(self.config.map_name, self.config.weather)

        # Initialize the callback list
        self.callbacks = callbacks if callbacks is not None else []

    def run(self) -> None:
        """
        Run the simulation, including warm-up, sensor attachment, and data processing loops.
        """
        steps_since_last_sensor_update = 0

        # Warm-up phase
        for _ in tqdm.tqdm(range(self.config.warmup_time), desc='Warm-up phase'):
            carla_frame = self.synchronization.synchronization_step()

        # Attach sensors to vehicles that spawned in warm-up phase
        self.sensor_manager.attach_sensors_to_vehicles()

        for callback in self.callbacks:
            callback.on_simulation_start(self)

        # Simulation loop
        for step in tqdm.tqdm(range(self.config.recording_length*self.config.recording_frequency), desc='Running simulation'):
            self.simulation_time: float = traci.simulation.getTime()  # Get the current simulation time
            for _ in range(5):
                try:
                    carla_frame = self.synchronization.synchronization_step()  # Step the simulation and synchronize CARLA and SUMO
                    break
                except Exception as e:
                    print('error in synchronization, retrying due to:', e)
                    time.sleep(1)   

            # Update visualizer
            # self.visualizer.update()

            steps_since_last_sensor_update += 1
            if steps_since_last_sensor_update >= self.config.recording_frequency:
                t = time.time()

                # Remove and attach sensors to vehicles
                self.sensor_manager.remove_sensors_from_vehicles()
                if self.config.sensor_attachments == 'dynamic':
                    self.sensor_manager.attach_sensors_to_vehicles()

                # Process sensor data
                sensor_data = {}
                for fco in tqdm.tqdm(self.fco_sensor_queues, desc='Processing sensor data'):
                    if all(len(self.fco_sensor_queues[fco][sensor_type].queue) > 0 for sensor_type in self.fco_sensor_queues[fco]):
                        time.sleep(0.1)
                        if all(any(q[1] == carla_frame for q in self.fco_sensor_queues[fco][sensor_type].queue) for sensor_type in self.fco_sensor_queues[fco]):
                            fco_sensor_data, current_rgb_results = self.data_processor.process_sensor_data(
                                fco,
                                self.fco_sensor_queues,
                                self.fco_cameramanager_mapping
                            )

                            sensor_data[fco] = {'fco_sensor_data': fco_sensor_data, 'current_rgb_results': current_rgb_results}
                        else:
                            print(f'Not all sensor data for {fco} is available for frame {carla_frame}')
                
                for callback in self.callbacks:
                    callback.on_simulation_step(self, step, sensor_data, sensor_manager=self.sensor_manager)

                self.do_step_log(t)
                steps_since_last_sensor_update = 0
        
        for callback in self.callbacks:
            callback.on_simulation_end(self)

        # Clean up
        self.sensor_manager.remove_all_sensors()
        self.synchronization.synchronization.close()

    def do_step_log(self, t: float) -> None:
        """
        Log the simulation step details.

        Args:
            t (float): Start time of the current simulation step.
        """
        if self.logger is None:
            return
        traci_vehicles: List[str] = traci.vehicle.getIDList()
        s_t_storage: Dict[str, int] = {}

        for s_t in set(self.fco_ids.values()):
            fco_ids_in_simulation = [k for k, v in self.fco_ids.items() if v == s_t]
            fco_ids_in_simulation = [fco for fco in fco_ids_in_simulation if fco in traci_vehicles]
            s_t_storage[s_t] = len(fco_ids_in_simulation)

        s_t_storage['no_sensor'] = len(traci_vehicles) - sum(s_t_storage.values())

        log_str = ''
        for s_t, count in s_t_storage.items():
            log_str += f'Number of vehicles with sensor setup {s_t}: {count}\n'
        log_str += f'Recording took {time.time() - t} seconds for this step with {len(self.fco_ids)} observers'

        self.logger.info(log_str)


if __name__ == '__main__':
    simulation_runner = SimulationRunner(config_file='config.yaml', sensor_config_file='sensor_config.yaml')
    simulation_runner.run()
