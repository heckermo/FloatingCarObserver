import numpy as np
from queue import Queue, LifoQueue
from carla import ColorConverter as cc
import traci
import time
import math
import carla
from Co_Simulation.tools.kitti_style import generate_kitti_datapoints
from Co_Simulation.tools.visualization import draw_3d_bounding_boxes_on_image
import logging
import open3d as o3d

class SensorManager:
    def __init__(self, synchronization, fco_ids: dict, sensor_config: dict, max_sensors: int = 1):
        self.synchronization = synchronization
        self.fco_ids = fco_ids  # Mapping from vehicle IDs to sensor setups
        self.sensor_config = sensor_config
        self.max_sensors = max_sensors

        self.fco_cameramanager_mapping = {}  # Mapping from FCO IDs to CameraManager instances
        self.fco_sensor_queues = {}  # Mapping from FCO IDs to sensor queues

    def attach_sensors_to_vehicles(self):
        for fco in self.fco_ids:
            if len(self.fco_cameramanager_mapping) >= self.max_sensors:
                logger.warning('Max number of sensors reached')
                return
            elif fco in traci.vehicle.getIDList():
                if fco not in self.fco_cameramanager_mapping:
                    if len(self.fco_cameramanager_mapping) >= self.max_sensors:
                        print('Max number of sensors reached')
                        return
                    t = time.time()
                    carla_id = self.synchronization.synchronization.sumo_carla_idmapping.get(fco)
                    # Double-check if the vehicle is in the CARLA world
                    carla_actors = self.synchronization.carla_simulation.world.get_actors()
                    carla_actors_ids = [actor.id for actor in carla_actors]
                    assert carla_id in carla_actors_ids, f'Carla id {carla_id} not in CARLA actors {carla_actors_ids}'
                    carla_vehicle = self.synchronization.carla_simulation.world.get_actor(carla_id)
                    sensor_setup = self.fco_ids[fco]
                    camera_manager = CameraManager(carla_vehicle, self.sensor_config, sensor_setup)
                    self.fco_cameramanager_mapping[fco] = camera_manager
                    self.fco_sensor_queues[fco] = {}

                    for sensor in camera_manager.sensors:
                        sensor_info = camera_manager.sensors[sensor]
                        if sensor_info['type'] == 'sensor.camera.rgb':
                            rgb_queue = LifoQueue()
                            rgb_callback = make_process_rgb_image(rgb_queue)
                            sensor_info['sensor'].listen(rgb_callback)
                            self.fco_sensor_queues[fco][sensor] = rgb_queue
                        elif sensor_info['type'] == 'sensor.camera.depth':
                            depth_queue = LifoQueue()
                            depth_callback = make_process_depth_image(depth_queue)
                            sensor_info['sensor'].listen(depth_callback)
                            self.fco_sensor_queues[fco][sensor] = depth_queue
                        elif sensor_info['type'] == 'sensor.lidar.ray_cast':
                            lidar_queue = LifoQueue()
                            lidar_callback = make_process_lidar_data(lidar_queue)
                            sensor_info['sensor'].listen(lidar_callback)
                            self.fco_sensor_queues[fco][sensor] = lidar_queue
                        elif sensor_info['type'] == 'sensor.lidar.ray_cast_semantic':
                            lidar_queue = LifoQueue()
                            lidar_callback = make_process_semantic_lidar_data(lidar_queue)
                            sensor_info['sensor'].listen(lidar_callback)
                            self.fco_sensor_queues[fco][sensor] = lidar_queue
                        else:
                            raise ValueError(f'Unknown sensor type: {sensor_info["type"]}')
                    logger.info(f'Attached sensors to vehicle {fco} in {time.time() - t} seconds')
                    print(f'Attached sensors to vehicle {fco} in {time.time() - t} seconds')

    def remove_sensors_from_vehicles(self):
        removed_vehicles = []
        for fco, camera_manager in self.fco_cameramanager_mapping.items():
            if fco not in traci.vehicle.getIDList():
                removed_vehicles.append(fco)
                for sensor in camera_manager.sensors.values():
                    carla_sensor = sensor.get('sensor')
                    if carla_sensor is not None:
                        carla_sensor.destroy()
        for fco in removed_vehicles:
            if fco in self.fco_cameramanager_mapping:
                del self.fco_cameramanager_mapping[fco]
                del self.fco_sensor_queues[fco]
        if len(removed_vehicles) > 0:
            logger.info(f'Removed sensors from vehicles: {removed_vehicles}')

    def remove_all_sensors(self):
        for fco, camera_manager in self.fco_cameramanager_mapping.items():
            for sensor in camera_manager.sensors.values():
                carla_sensor = sensor.get('sensor')
                if carla_sensor is not None:
                    carla_sensor.destroy()


class SensorDataProcessor:
    def __init__(self, synchronization, config):
        self.synchronization = synchronization
        self.config = config

    def process_sensor_data(self, fco, fco_sensor_queues, fco_cameramanager_mapping):
        fco_sensor_data = {}
        current_rgb_results = {}
        if all(len(queue.queue) > 0 for queue in fco_sensor_queues[fco].values()):
            fco_sensor_data[fco] = {}
            for sensor, queue in fco_sensor_queues[fco].items():
                data = queue.get()
                sensor_info = fco_cameramanager_mapping[fco].sensors.get(sensor)
                data = data[0]
                # Clear the queue
                with queue.mutex:
                    queue.queue.clear()

                # Process sensor data based on type
                if 'sensor.lidar' in sensor:
                    results = process_lidar_data(data, sensor_info)
                    fco_sensor_data[fco][sensor] = dict(zip(['image', 'points', 'lidar_height', 'lidar_cam_mat'], results))
                if 'sensor.semantic_lidar' in sensor:
                    results = process_lidar_data(data, sensor_info)
                    fco_sensor_data[fco][sensor] = dict(zip(['image', 'points', 'lidar_height', 'lidar_cam_mat'], results))
                elif 'sensor.camera.depth' in sensor:
                    results = process_depth_data(data, sensor_info)
                    fco_sensor_data[fco][sensor] = dict(zip(['image', 'depth'], results))
                elif 'sensor.camera.rgb' in sensor:
                    results = process_rgb_data(data, sensor_info)
                    sensor_info['calibration'] = results[1]
                    fco_sensor_data[fco][sensor] = dict(zip(['image', 'calibration'], results))

            # Process RGB sensor data for KITTI datapoints
            for sensor, data in fco_sensor_data[fco].items():
                if 'sensor.camera.rgb' in sensor:
                    related_depth_sensor = next(
                        (d_sensor for d_sensor in fco_sensor_data[fco]
                         if 'sensor.camera.depth' in d_sensor and
                         d_sensor.split('.')[-1] == sensor.split('.')[-1]), None)
                    if related_depth_sensor:
                        fco_carla_vehicle = self.synchronization.carla_simulation.world.get_actor(
                            self.synchronization.synchronization.sumo_carla_idmapping.get(fco))
                        results = generate_kitti_datapoints(
                            self.synchronization.carla_simulation.world,
                            fco_carla_vehicle,
                            fco_cameramanager_mapping[fco].sensors.get(sensor),
                            data['image'],
                            fco_sensor_data[fco][related_depth_sensor]['depth'],
                            self.config
                        )
                        image, kitti_datapoint, bounding_box, boxes_2d, ids = results
                        bb_img = draw_3d_bounding_boxes_on_image(image.copy(), bounding_box, (255, 0, 0))
                        sumo_detected_ids = [
                            next((k for k, v in self.synchronization.synchronization.sumo_carla_idmapping.items() if v == x), None)
                            for x in ids
                        ] if ids else []
                        current_rgb_results[sensor] = {
                            'image': image,
                            'kitti_datapoint': kitti_datapoint,
                            'bounding_box': bounding_box,
                            'boxes_2d': boxes_2d,
                            'ids': ids,
                            'sumo_detected_ids': sumo_detected_ids,
                            "carla_detected_ids": ids,
                            'depth_image': fco_sensor_data[fco][related_depth_sensor]['image'],
                            'bb_img': bb_img,
                            'P2_calib': fco_sensor_data[fco][sensor]['calibration']
                        }
                    else:
                        # Add only the image data to the results
                        current_rgb_results[sensor] = {'image': data['image']}
        return fco_sensor_data, current_rgb_results


class CameraManager(object):
    """
    A simplified camera manager class that spawns and keeps all required sensors for KITTI dataset
    """

    def __init__(self, parent_actor, sensor_config, sensor_setup):
        """Constructor method"""
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        self.index = 0
        self.sensor_config = sensor_config[sensor_setup]
        self.default_sensor_attachment_type = carla.AttachmentType.Rigid

        # Note that camera.rgb and camera.depth should have the same config for correct detection of occluded agents
        self.sensors = {}
        for sensor_type in self.sensor_config.keys():
            for orientation in self.sensor_config[sensor_type].keys():
                assert 'position' in self.sensor_config[sensor_type][orientation], f'Position not defined for {sensor_type}'
                assert 'rotation' in self.sensor_config[sensor_type][orientation], f'Rotation not defined for {sensor_type}'
                sensor_transform = carla.Transform(carla.Location(x=self.sensor_config[sensor_type][orientation]['position']['x'],
                                                                    y=self.sensor_config[sensor_type][orientation]['position']['y'],
                                                                    z=self.sensor_config[sensor_type][orientation]['position']['z']),
                                                    carla.Rotation(pitch=self.sensor_config[sensor_type][orientation]['rotation']['pitch'],
                                                                     yaw=self.sensor_config[sensor_type][orientation]['rotation']['yaw'],
                                                                     roll=self.sensor_config[sensor_type][orientation]['rotation']['roll']))
                if sensor_type == 'rgb':
                    camera_rgb_attributes = {'image_size_x': str(self.sensor_config[sensor_type][orientation]['width']),  # [px]
                                            'image_size_y': str(self.sensor_config[sensor_type][orientation]['height']),  # [px]
                                            'fov': str(self.sensor_config[sensor_type][orientation]['fov'])}  # [°]
                    self.sensors.update({f'sensor.camera.rgb.{orientation}': {'name': f'Camera RGB {orientation}',
                                                                         'type': 'sensor.camera.rgb',
                                                                         'position': orientation,
                                                                         'attributes': camera_rgb_attributes,
                                                                         'transform': sensor_transform}})
                    # position depth sensor at the same location as the rgb camera
                    if self.sensor_config[sensor_type][orientation]['depth'] is True:
                        self.sensors.update({f'sensor.camera.depth.{orientation}': {'name': f'Camera Depth {orientation}',
                                                                                    'type': 'sensor.camera.depth',
                                                                                    'position': orientation,
                                                                                    'attributes': camera_rgb_attributes,
                                                                                    'transform': sensor_transform}})
                
                elif sensor_type == 'lidar':
                    lidar_attributes = {'channels': '64',
                                        'range': str(self.sensor_config[sensor_type][orientation]['range']),  # [m]
                                        'points_per_second': '1300000',
                                        'rotation_frequency': '100',
                                        'upper_fov': '7.0',
                                        'lower_fov': '-16.0',
                                        'atmosphere_attenuation_rate': '0.004',
                                        'noise_stddev': '0.0',
                                        'dropoff_general_rate': '0.10',
                                        'dropoff_zero_intensity': '0.4',
                                        'dropoff_intensity_limit': '0.8'}
                    self.sensors.update({f'sensor.lidar.{orientation}': {'name': f'Lidar {orientation}',
                                                                    'type': 'sensor.lidar.ray_cast', 
                                                                   'position': orientation,
                                                                   'attributes': lidar_attributes,
                                                                   'transform': sensor_transform}})
                
                elif sensor_type == 'semantic-lidar':
                    lidar_attributes = {'channels': '64',
                                        'range': str(self.sensor_config[sensor_type][orientation]['range']),  # [m]
                                        'points_per_second': '1300000',
                                        'rotation_frequency': '100',
                                        'upper_fov': '7.0',
                                        'lower_fov': '-16.0'}
                    self.sensors.update({f'sensor.semantic_lidar.{orientation}': {'name': f'Semantic Lidar {orientation}',
                                                                    'type': 'sensor.lidar.ray_cast_semantic', 
                                                                   'position': orientation,
                                                                   'attributes': lidar_attributes,
                                                                   'transform': sensor_transform}})
                
                else:
                    raise ValueError(f'Unknown sensor type: {sensor_type}')

        self.setup_sensors()

    def get_intrinsic_matrix(self, camera):

        width = int(camera.attributes['image_size_x'])
        height = int(camera.attributes['image_size_y'])
        fov = float(camera.attributes['fov'])

        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))

        return k

    def toggle_camera(self):
        """Activate a camera"""
        raise NotImplementedError

    def setup_sensors(self, lidar_same_orientation_only=True):
        # Spawns all sensors
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for sensor_key in self.sensors.keys():
            blp = bp_library.find(self.sensors[sensor_key]['type'])
            for key, val in self.sensors[sensor_key]['attributes'].items():
                blp.set_attribute(key, val)
            transform = self.sensors[sensor_key]['transform']
            attach_type = self.sensors[sensor_key].get('attach_type', self.default_sensor_attachment_type)
            sensor = self._parent.get_world().spawn_actor(blp,
                                                          transform,
                                                          attach_to=self._parent,
                                                          attachment_type=attach_type)
            self.sensors[sensor_key].update({'sensor': sensor})

            # Setup intrinsic matrix for camera rgb sensor
            if 'sensor.camera.rgb' in sensor_key:
                camera_rgb_intrinsic = self.get_intrinsic_matrix(sensor)
                self.sensors[sensor_key].update({'calibration': camera_rgb_intrinsic})

            # Setup relative to camera matrix for lidar sensors
            elif 'lidar' in sensor_key:
                orientation = sensor_key.split('.')[-1]
                self.sensors[sensor_key]['lidar_cam_mat'] = {}
                for s in self.sensors.keys():
                    if 'sensor.camera.rgb' in s:
                        if lidar_same_orientation_only and orientation != s.split('.')[-1]:
                            continue
                        rgb_direction = self.sensors[s]['position']
                        veh_cam_mat = self.sensors[s]['transform'].get_inverse_matrix()
                        lidar_veh_mat = self.sensors[sensor_key]['transform'].get_matrix()
                        lidar_cam_mat = np.dot(veh_cam_mat, lidar_veh_mat)
                        self.sensors[sensor_key]['lidar_cam_mat'][s] = lidar_cam_mat

    def set_sensor(self, index, notify=True):
        """Set a sensor"""
        index = index % len(self.sensors.keys())
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        raise NotImplementedError

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

def make_process_rgb_image(queue):
    def process_rgb_image(image):
        color_converter = cc.Raw
        image.convert(color_converter)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # RGB, discard alpha
        queue.put((array, image.frame))
    return process_rgb_image

def make_process_depth_image(queue):
    def process_depth_image(image):
        color_converter = cc.Depth
        image.convert(color_converter)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        queue.put((array, image.frame))
    return process_depth_image

def make_process_lidar_data(queue):
    def process_lidar_data(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        lidar_points = np.reshape(points, (int(points.shape[0] / 4), 4))
        queue.put((lidar_points.copy(), data.frame))
    return process_lidar_data

def make_process_semantic_lidar_data(queue):
    def process_semantic_lidar_data(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        # Each point has 6 floats: x, y, z, cos_inc_angle, object_idx, object_tag
        lidar_points = np.reshape(points, (int(points.shape[0] / 6), 6))

        # Extract x, y, z
        coordinates = lidar_points[:, 0:3]  # shape (N, 3)
        object_idx = lidar_points[:, 4]   # if you want the actual object ID
        # The semantic label is typically the 6th float (index 5)
        labels = lidar_points[:, 5].astype(np.int32)

        # Combine coordinates + label in a single array if you wish
        lidar_points = np.hstack((coordinates, object_idx[:, np.newaxis], labels[:, np.newaxis]))

        queue.put((lidar_points.copy(), data.frame))
    return process_semantic_lidar_data


def process_lidar_data(data, lidar_sensor):
    lidar_points = data.copy()
    lidar_data = np.array(lidar_points[:, :2])
    lidar_data *= min([1248, 384]) / (2.0 * 100)
    lidar_data += (0.5 * 1248, 0.5 * 384)
    lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (1248, 384, 3)
    lidar_img = np.zeros(lidar_img_size)
    try: 
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
    except:
        lidar_img = np.zeros(lidar_img_size)
    lidar_height = lidar_sensor.get('transform').location.z
    lidar_cam_mat = lidar_sensor.get('lidar_cam_mat')
    return lidar_img, lidar_points, lidar_height, lidar_cam_mat

def process_depth_data(data, depth_sensor):
    depth_data = data.copy()
    # Decoding depth: https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-depth-map
    depth_array = depth_data.astype(np.float32)
    normalized_depth = np.dot(depth_array, [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth = normalized_depth * 1000
    return depth_data, depth

def process_rgb_data(data, sensor):
    rgb_image = data.copy()
    calibration = sensor.get('calibration')
    return rgb_image, calibration


if __name__ == "__main__":
    raise NotImplementedError # This script is not meant to be run directly

else:
    logger = logging.getLogger(__name__)

