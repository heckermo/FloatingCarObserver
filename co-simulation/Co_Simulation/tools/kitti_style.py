import sys
import os
import numpy as np
import traci
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from Co_Simulation.tools.CARLA_KITTI.bounding_box import create_kitti_datapoint
from Co_Simulation.tools.CARLA_KITTI.dataexport import save_calibration_matrices
import cv2
import yaml

def generate_kitti_datapoints(world, ego, rgb_camera, image, depth_map, args):
        """
        Returns a list of datapoints (labels and such) that are generated this frame
        together with the main image image
        params:
            world: the carla world object
            ego: the ego carla vehicle object
            image: the image from the rgb camera
            depth_map: the depth map from the depth camera (may needs to be adapted image and depth map)
            args: the arguments from the argument parser (lidar range needed)    
        """

        datapoints = []
        bounding_boxes = []
        boxes_2d = []
        image = image.copy()
        agents_list = []
        ids = []

        pedestrians_list = world.get_actors().filter('walker.pedestrian.*')
        agents_list.extend(pedestrians_list)
        vehicles_list = world.get_actors().filter('vehicle.*')
        agents_list.extend(vehicles_list)
        agents_list = [agent for agent in agents_list if agent.id != ego.id]

        # Stores all datapoints for the current frames
        for agent in agents_list:
            assert str(ego.get_transform()) != str(agent.get_transform())
            image, kitti_datapoint, bounding_box = create_kitti_datapoint(agent=agent,
                                                                          camera=rgb_camera['sensor'],
                                                                          position=rgb_camera['position'],
                                                                          cam_calibration=rgb_camera['calibration'],
                                                                          image=image,
                                                                          depth_map=depth_map,
                                                                          player_transform=ego.get_transform(),
                                                                          max_render_depth=100,
                                                                          kitti_criteria='hard', 
                                                                          window_width = int(rgb_camera['sensor'].attributes['image_size_x']),
                                                                          window_height = int(rgb_camera['sensor'].attributes['image_size_y']))
            if kitti_datapoint is not None:
                datapoints.append(kitti_datapoint)
                #if not kitti_datapoint.type == 'DontCare':
                bounding_boxes.append(bounding_box)
                boxes_2d.append(kitti_datapoint.bbox)
                ids.append(agent.id)
        return image, datapoints, bounding_boxes, boxes_2d, ids


def save_pointcloud(points: np.array, path, name): 
    """
    Saves the point cloud to a pcd file
    """
    points[:, 1] = -points[:, 1]
    header = f"""# .PCD v.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z intensity
        SIZE 4 4 4 4
        TYPE F F F F
        COUNT 1 1 1 1
        WIDTH {len(points)}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {len(points)}
        DATA ascii
        """ 
    with open(os.path.join(path, name), 'w') as pcd_file:
        pcd_file.write(header)
        for point in points:
            pcd_file.write(' '.join(map(str, point)) + '\n')


def run_kitti_inference(fco, rgb_results:dict, sumo_carla_idmapping, distance = 50):
    detected_vehicles = []
    for sensor in rgb_results:
        # check that the rgb results have the ids of the detected vehicles
        assert 'ids' in rgb_results[sensor] 
        for id in rgb_results[sensor]['ids']:
            for sumo_id, carla_id in sumo_carla_idmapping.items():
                if carla_id == id:
                    detected_vehicles.append(sumo_id)
    all_vehicles = traci.vehicle.getIDList()
    ego_pos_x, ego_pos_y = traci.vehicle.getPosition(fco)
    distance_vehicles = []
    for vehicle in all_vehicles:
        pos_x, pos_y = traci.vehicle.getPosition(vehicle)
        distance_to_ego = np.sqrt((pos_x - ego_pos_x)**2 + (pos_y - ego_pos_y)**2)
        if distance_to_ego < distance:
            distance_vehicles.append(vehicle)
    undetected_vehicles = [v for v in distance_vehicles if v not in detected_vehicles]
    detected_vehicles = [v for v in detected_vehicles if v in distance_vehicles]

    kitti_results_dict = {'detected': detected_vehicles, 'undetected_vehicles': undetected_vehicles}
    return kitti_results_dict

def save_kitti_data(dirname: str, timestep: float, camera_manager, fco_id: str, sensor_data, current_rgb_results: dict):
    results_dict = {}
    # convert the timestep to string and replace the dot with an underscore
    timestep_str = str(timestep).replace('.', '_')
    # create dir under with the current timestep and the fco id
    dirname = os.path.join(dirname, timestep_str, fco_id)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    # save kitti datapoint for each rgb sensor
    detected_vehicles = []
    for rgb_result in current_rgb_results:
        oreintation = rgb_result.split('.')[-1]
        orientation = rgb_result.split('.')[-1]
        kitti_datapoint = current_rgb_results[rgb_result]['kitti_datapoint']
        # if orientation == 'left' or orientation == 'right':
        #     # rotation the bounding box by 90 degrees
        #     for point in kitti_datapoint:
        #         angle = point.alpha + np.pi/2
        #         adjusted_rotation_y = (angle + np.pi) % (2 * np.pi) - np.pi # make sure the angle is between -pi and pi
        #         point.alpha = adjusted_rotation_y
        bounding_box = current_rgb_results[rgb_result]['bounding_box']
        ids = current_rgb_results[rgb_result]['ids']
        bb_img = current_rgb_results[rgb_result]['bb_img']
        sensor_path = os.path.join(dirname, rgb_result.split('.')[-1])

        if not os.path.exists(sensor_path):
            os.makedirs(sensor_path)

        # save the kitti datapoint
        filename = os.path.join(sensor_path, 'kitti_datapoint.txt')
        with open(filename, 'w') as f:
            out_str = "\n".join([str(point) for point in kitti_datapoint if point])
            f.write(out_str)
        
        datapoint_dict = {i: kitti_datapoint[i].__dict__ for i in range(len(kitti_datapoint))}
        with open(os.path.join(sensor_path, 'kitti_datapoint.yaml'), 'w') as f:
            yaml.dump(datapoint_dict, f, default_flow_style=False)
        
        # TODO allow for settings without lidar and where one lidar is used for multiple cameras
        lidar_name = [sensor for sensor in camera_manager.sensors.keys() if 'sensor.lidar' in sensor and orientation in sensor][0]
        save_calibration_matrices(os.path.join(sensor_path, 'calib.txt'), camera_manager.sensors.get(rgb_result).get('calibration'), 
                              camera_manager.sensors.get(lidar_name).get('lidar_cam_mat').get(rgb_result))

        # save the bounding box image
        cv2.imwrite(os.path.join(sensor_path, 'bounding_box_image.png'), bb_img)

        # save the rgb image data
        cv2.imwrite(os.path.join(sensor_path, 'rgb_image.png'), current_rgb_results[rgb_result]['image'])

        # save the depth image data
        cv2.imwrite(os.path.join(sensor_path, 'depth_image.png'), current_rgb_results[rgb_result]['depth_image'])

        # save raw kitti datapoint
        if len(kitti_datapoint) > 0:
            datapoint_dict = {i: kitti_datapoint[i].__dict__ for i in range(len(kitti_datapoint))}
            with open(os.path.join(sensor_path, 'raw_kitti_datapoint.yaml'), 'w') as f:
                yaml.dump(datapoint_dict, f, default_flow_style=False)

        # save the lidar data
        cv2.imwrite(os.path.join(sensor_path, 'lidar_image.png'), sensor_data[lidar_name]['image'])
        save_pointcloud(sensor_data[lidar_name]['points'], sensor_path, 'lidar_points.pcd')
    return dirname
            
    
