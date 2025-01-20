import numpy as np
import os
import math
import traci
import yaml
import carla
import time
import tqdm
import sys
import shutil

SENSOR_OPV2V_MAPPING = {'front': 'camera0', 'back': 'camera3', 'left': 'camera2', 'right': 'camera1'}
OPV2V_BASE_PATH = '/data/opv2v_dataset'

def save_opv2v_data(dirname, timestep, world, sumo_carla_idmapping, fco_cameramanager_mapping, fco_id, fco_sensor_data, current_rgb_results, recording_range = 1000):
    # convert the timestep to string and replace the dot with an underscore
    timestep_str = str(timestep).replace('.', '_')
    # create dir under with the current timestep and the fco id
    dirname = os.path.join(dirname, timestep_str, fco_id)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    # get the detected ids from the sensor data
    detected_carla_ids = []
    for result in current_rgb_results:
        detected_carla_ids.extend(current_rgb_results[result]['carla_detected_ids'])
    
    # get all vehicles that are in range of the fco (in sumo and then the carla ids)
    fco_carla_id = sumo_carla_idmapping.get(fco_id)
    carla_sumo_idmapping = {v: k for k, v in sumo_carla_idmapping.items()}
    fco_carla_vehicle = world.get_actor(fco_carla_id)
    all_vehicles = traci.vehicle.getIDList()
    # get the transformation of the fco_vehicle
    fco_transform = fco_carla_vehicle.get_transform()
    range_vehicles = list()
    for vehicle in all_vehicles:
        distance = math.sqrt((traci.vehicle.getPosition(vehicle)[0] - traci.vehicle.getPosition(fco_id)[0])**2 + (traci.vehicle.getPosition(vehicle)[1] - traci.vehicle.getPosition(fco_id)[1])**2)
        if distance < recording_range and vehicle != fco_id:
            range_vehicles.append(vehicle)
    # get the carla ids of the vehicles in range
    carla_range_vehicles = [sumo_carla_idmapping.get(vehicle) for vehicle in range_vehicles]
    # get transformation for all carla vehicles in range and save in dict
    carla_range_transforms = {carla_vehicle: world.get_actor(carla_vehicle).get_transform() for carla_vehicle in carla_range_vehicles}
    # get the transforms for the cameras of the fco
    carla_sensor_transforms = {}
    carla_sensor_intrinsics = {}
    carla_sensor_extrinsics = {}

    data_to_save = {}
    
    for sensor in fco_cameramanager_mapping[fco_id].sensors:
        if 'sensor.camera.rgb' in sensor:
            camera_transform = fco_cameramanager_mapping[fco_id].sensors.get(sensor).get('sensor').get_transform()
            carla_sensor_transforms[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]] = [camera_transform.location.x, camera_transform.location.y, camera_transform.location.z, camera_transform.rotation.roll, camera_transform.rotation.yaw, camera_transform.rotation.pitch]
            carla_sensor_intrinsics[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]] = fco_cameramanager_mapping[fco_id].sensors.get(sensor).get('calibration').tolist()
            carla_sensor_extrinsics[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]] = calculate_extrinsics(fco_transform, camera_transform).tolist()

            data_to_save[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]] = {}
            data_to_save[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]]['cords'] = carla_sensor_transforms[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]]
            data_to_save[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]]['intrinsic'] = carla_sensor_intrinsics[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]]
            data_to_save[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]]['extrinsic'] = carla_sensor_extrinsics[SENSOR_OPV2V_MAPPING[sensor.split('.')[-1]]]
        
        if 'sensor.lidar' in sensor:
            lidar_transform = fco_cameramanager_mapping[fco_id].sensors.get(sensor).get('sensor').get_transform()
            data_to_save['lidar_pose'] = [lidar_transform.location.x, lidar_transform.location.y, lidar_transform.location.z, lidar_transform.rotation.roll, lidar_transform.rotation.yaw, lidar_transform.rotation.pitch]
    
    data_to_save['ego_speed'] = get_sumo_speed(fco_id)
    data_to_save['true_ego_pos'] = [fco_transform.location.x, fco_transform.location.y, fco_transform.location.z, fco_transform.rotation.roll, fco_transform.rotation.yaw, fco_transform.rotation.pitch]

    data_to_save['vehicles'] = {}
    for carla_vehicle in carla_range_transforms:
        data_to_save['vehicles'][carla_vehicle] = {}
        data_to_save['vehicles'][carla_vehicle]['angle'] = [carla_range_transforms[carla_vehicle].rotation.roll, carla_range_transforms[carla_vehicle].rotation.yaw, carla_range_transforms[carla_vehicle].rotation.pitch]
        data_to_save['vehicles'][carla_vehicle]['center'] = [0,0, world.get_actor(carla_vehicle).bounding_box.extent.z]
        data_to_save['vehicles'][carla_vehicle]['extent'] = [world.get_actor(carla_vehicle).bounding_box.extent.x, world.get_actor(carla_vehicle).bounding_box.extent.y, world.get_actor(carla_vehicle).bounding_box.extent.z]
        data_to_save['vehicles'][carla_vehicle]['location'] = [carla_range_transforms[carla_vehicle].location.x, carla_range_transforms[carla_vehicle].location.y, carla_range_transforms[carla_vehicle].location.z]
        data_to_save['vehicles'][carla_vehicle]['speed'] = get_sumo_speed(carla_sumo_idmapping[carla_vehicle])
        if carla_vehicle in detected_carla_ids:
            data_to_save['vehicles'][carla_vehicle]['detected'] = True
        else:
            data_to_save['vehicles'][carla_vehicle]['detected'] = False
    
    # save the results dict to a yaml file
    with open(os.path.join(dirname, 'data.yaml'), 'w') as file:
        yaml.dump(data_to_save, file, default_flow_style=False)

def get_matrix_from_transform(transform):
    """
    Converts a CARLA Transform to a 4x4 transformation matrix.
    """
    rotation = transform.rotation
    location = transform.location
    
    c_yaw = np.cos(np.radians(rotation.yaw))
    s_yaw = np.sin(np.radians(rotation.yaw))
    c_pitch = np.cos(np.radians(rotation.pitch))
    s_pitch = np.sin(np.radians(rotation.pitch))
    c_roll = np.cos(np.radians(rotation.roll))
    s_roll = np.sin(np.radians(rotation.roll))
    
    # Construct the rotation matrix
    rotation_matrix = np.matrix([
        [c_pitch*c_yaw, c_yaw*s_pitch*s_roll - s_yaw*c_roll, c_yaw*s_pitch*c_roll + s_yaw*s_roll],
        [s_yaw*c_pitch, s_yaw*s_pitch*s_roll + c_yaw*c_roll, s_yaw*s_pitch*c_roll - c_yaw*s_roll],
        [-s_pitch, c_pitch*s_roll, c_pitch*c_roll]
    ])
    
    # Construct the full 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = (location.x, location.y, location.z)
    
    return transform_matrix

def calculate_extrinsics(fco_transform, camera_transform):
    """
    Calculates the extrinsics matrix from the vehicle to the camera.
    """
    # Convert CARLA transforms to 4x4 matrices
    M_fco = get_matrix_from_transform(fco_transform)
    M_camera = get_matrix_from_transform(camera_transform)
    
    # Calculate the inverse of the vehicle's matrix
    M_fco_inv = np.linalg.inv(M_fco)
    
    # Calculate extrinsics
    M_extrinsics = M_fco_inv @ M_camera
    
    return M_extrinsics

def get_camera_intrinsics(image_width, image_height, fov):
    """
    Calculate intrinsic parameters of a CARLA RGB camera sensor.
    
    :param image_width: Width of the camera image in pixels.
    :param image_height: Height of the camera image in pixels.
    :param fov: Field of view of the camera in degrees.
    :return: A dictionary containing fx, fy, cx, cy, and the intrinsic matrix K.
    """
    # Assuming the sensor width is equal to the sensor height
    # and the camera has a square pixel aspect ratio
    f = image_width / (2.0 * math.tan(math.radians(fov) / 2.0))
    cx = image_width / 2.0
    cy = image_height / 2.0
    s = 0  # Assuming no skew
    
    K = [[f, s, cx],
         [0, f, cy],
         [0, 0, 1]]
    
    return K

def get_carla_speed(carla_actor):
    """
    Get the velocity of a CARLA actor in the world frame.
    """
    velocity = carla_actor.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
    return float(speed)

def get_sumo_speed(sumo_vehicle):
    """
    Get the speed of a sumo vehicle.
    """
    return float(traci.vehicle.getSpeed(sumo_vehicle))

def create_opv2v_dataset(b_path):
    """
    This function is used to create a dataset form the tmp_recording forlder and convert it to the OPV2V style.
    """
    for base_path in os.listdir(b_path):
        config = get_config(os.path.join(b_path, base_path, 'co-simulation.yaml'))
        sumo_carla_idmapping = config_to_dict(os.path.join(b_path, base_path, 'sumo_carla_id_mapping.yaml'))
        meta_data = get_config(os.path.join(b_path, base_path, 'meta_data.yaml'))
        town = config.map_name
        dataset_path = os.path.join(OPV2V_BASE_PATH, base_path.split('/')[-1])
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        timestamp_folders = os.listdir(os.path.join(b_path, base_path))
        timestamp_folders = [folder for folder in timestamp_folders if not folder.endswith('.yaml')]
        seen_fco_ids = []
        # iterate through the timestamp folders
        for timestamp in tqdm.tqdm(timestamp_folders):
            opv2v_timestamp = f"{int(float(timestamp.replace('_', '.'))*10):06d}"
            print(f"Processing {opv2v_timestamp}")
            # get all the fco folders
            fco_folders = os.listdir(os.path.join(b_path, base_path, timestamp))
            for fco in fco_folders:
                carla_fco = str(sumo_carla_idmapping.get(fco))
                if fco not in seen_fco_ids:
                    seen_fco_ids.append(fco)
                    os.makedirs(os.path.join(dataset_path, carla_fco))
                # copy .yaml file to the fco folder
                shutil.copy(os.path.join(b_path,base_path, timestamp, fco, 'data.yaml'), os.path.join(dataset_path, carla_fco, opv2v_timestamp + '.yaml'))
                #shutil.copy(os.path.join(b_path,base_path, timestamp, fco, 'lidar_points.pcd'), os.path.join(dataset_path, carla_fco, opv2v_timestamp + '.pcd'))
                shutil.copy(os.path.join(b_path,base_path, timestamp, fco, 'front', 'rgb_image.png'), os.path.join(dataset_path, carla_fco, opv2v_timestamp + '_camera0.png'))
                shutil.copy(os.path.join(b_path,base_path, timestamp, fco, 'back', 'rgb_image.png'), os.path.join(dataset_path, carla_fco, opv2v_timestamp + '_camera3.png'))
                shutil.copy(os.path.join(b_path,base_path, timestamp, fco, 'left', 'rgb_image.png'), os.path.join(dataset_path, carla_fco, opv2v_timestamp + '_camera2.png'))
                shutil.copy(os.path.join(b_path,base_path, timestamp, fco, 'right', 'rgb_image.png'), os.path.join(dataset_path, carla_fco, opv2v_timestamp + '_camera1.png'))
                #if os.path.exists(os.path.join(b_path,base_path, timestamp, fco, 'sumo_bev.png')):
                #    shutil.copy(os.path.join(b_path,base_path, timestamp, fco, 'sumo_bev.png'), os.path.join(dataset_path, carla_fco, opv2v_timestamp + '_bev.png'))
                    

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from Carla.utils.utils import get_config, config_to_dict
    create_opv2v_dataset('/data/recordings/')