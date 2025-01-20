from sumo_integration.bridge_helper import BridgeHelper
from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.constants import INVALID_ACTOR_ID
from sumo_integration.sumo_simulation import SumoSimulation
import traci
import logging
from typing import Any

class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    Class from the SUMO run_synchronozation.py file.
    """
    def __init__(self,
                 sumo_simulation,
                 carla_simulation,
                 tls_manager='none',
                 sync_vehicle_color=False,
                 sync_vehicle_lights=False):

        self.sumo = sumo_simulation
        self.carla = carla_simulation

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights

        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        self.sumo_carla_idmapping = {}

    def tick(self, fco_sensor_queues=None, fco_cameramanager_mapping=None):
        """
        Tick to simulation synchronization
        """
        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)

                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
                
                    # add the sumo-carla id mapping
                    self.sumo_carla_idmapping[sumo_actor_id] = carla_actor_id
                
                else: 
                    # remove the sumo vehicle from the simulation
                    traci.vehicle.remove(sumo_actor_id)
                    logger.warning(f"Actor {sumo_actor_id} could not be spawned in Carla, removing from SUMO")
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                carla_id = self.sumo2carla_ids.pop(sumo_actor_id)    

                # remove the sensors from the parent actor in carla
                all_actors = self.carla.world.get_actors()
                for sensor in all_actors.filter('sensor.*'):
                    if sensor.parent and sensor.parent.id == carla_id:
                        sensor.destroy()
                        logger.info("Sensor {sensor.id} destroyed")
                
                self.carla.destroy_actor(carla_id)
            
                # remove also the sensors associated with the vehicle
                if fco_sensor_queues is not None:
                    if sumo_actor_id in fco_sensor_queues:
                        fco_sensor_queues.pop(sumo_actor_id)
                
                # remove also the camera manager associated with the vehicle
                if fco_cameramanager_mapping is not None:
                    if sumo_actor_id in fco_cameramanager_mapping:
                        fco_cameramanager_mapping.pop(sumo_actor_id)
            
                

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # tick the carla simulation
        carla_frame = self.carla.tick()

        return carla_frame

    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        # settings = self.carla.world.get_settings()
        # settings.synchronous_mode = False
        # settings.fixed_delta_seconds = None
        # self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()


class SumoCarlaSynchronization():
    def __init__(self, sumo_cfg_file: str, args: Any):
        
        self.sumo_simulation = SumoSimulation(sumo_cfg_file, args.step_length, args.sumo.sumo_host,
                                        args.sumo.sumo_port, args.sumo.sumo_gui)
        self.carla_simulation = CarlaSimulation(args.carla.host, args.carla.port, args.step_length)

        self.synchronization = SimulationSynchronization(self.sumo_simulation, self.carla_simulation,
                                                    args.sync.sync_vehicle_color, args.sync.sync_vehicle_lights)
    
    def synchronization_step(self, fco_sensor_queues=None, fco_cameramanager_mapping=None):
        try:
            carla_frame = self.synchronization.tick(fco_sensor_queues, fco_cameramanager_mapping)
        except KeyboardInterrupt:
            logger.info('Cancelled by user.')
        
        return carla_frame

    def synchronization_loop(self):
        while True:
            self.synchronization_step()

if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly.")

else: 
    logger = logging.getLogger(__name__)