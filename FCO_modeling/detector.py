# Standard library imports
import os
import time
import math
import shutil
import importlib
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Union, Optional, Dict

# Third-party library imports
import numpy as np
import pandas as pd
import torch
import libsumo as traci
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from shapely.prepared import prep

# Project-specific imports
from configs.config_cv import DEFAULT, CAMERAS

# SUMO Detector Plus specific imports
from utils.polygon_to_tensor import create_bev_tensor
from utils.cv_utils.detect_cv import run_computer_vision_approach, split_detected_from_undetected_objects
from utils.cv_utils.utils_cv import create_camera_matrices_from_config
from utils.cv_utils.create_3d import (
    create_3d,
    create_all_raw_participants,
    create_all_raw_buildings,
)
from utils.vit_pytorch import ViT
from utils.file_utils import delete_all_items_in_dir
from utils.nn_utils.create_box import create_nn_input, get_tmp_image_tensor
from utils.nn_utils.create_cnn import CustomResNet
from utils.raytracing_utils.raytracing import (
    detect_intersections,
    generate_rays,
    create_vehicle_polygon,
    parse_polygons_from_xml,
    save_polygons_and_rays_to_file,
)



class Detector(ABC):
    @abstractmethod
    def detect(self, vehicles: List[str]) -> Dict[str, List[str]]:
        pass

class D3RaytracingDetector(Detector):
    def __init__(self, camera_settings=None, distance=50, **kwargs):
        """
        Initializes the CV Detector with optional camera settings and detection distance.
        
        Parameters:
            camera_settings: Settings for camera configuration.
            distance (int): Default detection distance.
        """
        self.distance = distance
        self.camera_settings = camera_settings
        self._initialize_cv_mode()
    
    def _initialize_cv_mode(self):
        """Initializes the resources needed for computer vision (CV) mode."""
        self._3d_distance = 55
        # Initialize raw participants and buildings
        self.raw_small_car, self.raw_large_car, self.raw_delivery_car, \
        self.raw_bus, self.raw_truck, self.raw_bike, self.raw_person = create_all_raw_participants()
        self.raw_3d_buildings, self.raw_buildings = create_all_raw_buildings()
        shutil.rmtree('tmp_3d', ignore_errors=True)  # Remove temporary 3D directory
    
    def detect(self, vehicle_ids):
        """Executes detection using computer vision (CV) mode."""
        results = {}
        for vehicle_id in vehicle_ids:
            points = create_3d(vehicle_id, self.raw_small_car, self.raw_large_car, self.raw_delivery_car, self.raw_bus,
                            self.raw_truck, self.raw_bike, self.raw_person, self.raw_3d_buildings, self.raw_buildings,
                            self._3d_distance, save=False)
            results[vehicle_id] = self._execute_cv(points)
        return results
    
    def _execute_cv(self, points):
        """Executes computer vision processing on the given points."""
        ply_dir = 'tmp_3d'
        default_settings = DEFAULT  # Assuming DEFAULT is a configuration dictionary
        radius = default_settings['detection_range']
        objects_to_detect = default_settings['objects_to_detect']

        detected_objects = self._run_cv_detection(ply_dir, radius, objects_to_detect)
        delete_all_items_in_dir(ply_dir)  # Clean up temporary files

        return detected_objects
    
    def _run_cv_detection(self, ply_dir, radius, objects_to_detect):
        """Runs the computer vision approach to detect objects."""
        cameras = create_camera_matrices_from_config(CAMERAS)  # Assuming CAMERAS is a configuration
        # Iterates all frames from ply_dir
        for par_folder, camera_detections, camera_detections_occupancy, seq_pt_clouds, objects_pt_cloud_mask, objects_pt_cloud_disparity in run_computer_vision_approach(
                ply_dir=ply_dir,
                cameras=cameras,
                scan_radius=radius,
                objects_to_detect=objects_to_detect,
                save_depth_image_dir=None
            ):
            detected_objects = []
            for cam_pos in camera_detections:
                for obj in camera_detections[cam_pos]['vehicle']:
                    if camera_detections[cam_pos]['vehicle'][obj] == True:
                        detected_objects.append(obj)
        return detected_objects
    
    def _extract_detected_objects(self, detected_objects, other_objects):
        """Extracts and categorizes detected objects."""
        all_vehicles = self._get_object_ids(detected_objects, other_objects, 'vehicle')
        all_cyclists = self._get_object_ids(detected_objects, other_objects, 'cyclist')
        all_pedestrians = self._get_object_ids(detected_objects, other_objects, 'pedestrian')
        return all_vehicles, all_cyclists, all_pedestrians
    
    def _get_object_ids(self, detected_objects, other_objects, obj_type):
        """Helper to extract object IDs based on type."""
        objects = detected_objects.get(obj_type, []) + other_objects.get(obj_type, [])
        return [obj[obj.find('_') + 1: obj.rfind('.')] for obj in objects]

class EmulationDetector(Detector):
    def __init__(self, model_path: str, building_polygons: list, distance=50, batch_size=256, **kwargs):
        """
        Initializes the NN Detector with the specified model path.
        
        Parameters:
            model_path (str): Path to the trained model for 'nn' mode.
            distance (int): Default detection distance.
        """
        if model_path is None:
            raise ValueError("Path to the trained model is required for 'nn' mode.")
        self.model_path = model_path
        self.distance = distance
        self.batch_size = batch_size
        self.building_polygons = building_polygons
        self.building_polygon_centroids = np.array([polygon.centroid.coords[0] for polygon in building_polygons])
        self._initialize_nn_mode()
    
    def _initialize_nn_mode(self):
        """Initializes the neural network (NN) mode with the specified model path."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Import network configurations
        network_module = f"trained_emulation_models.{self.model_path}.config_networks"
        NETWORK_CONFIG = importlib.import_module(network_module).NETWORK_CONFIG
    
        self.model = torch.load(os.path.join('trained_emulation_models', self.model_path, 'best_val_model.pt'), map_location=self.device)
        self.image_size = NETWORK_CONFIG['IMAGE_SIZE']
        self.model.eval().to(self.device)
    
    def detect(self, fco_vehicles: List[str]) -> Dict[str, List[str]]:  
        """Executes detection using neural network (NN) mode."""
        t = time.time()
        self.dataset = []  # List to hold detection data
        self.bev_data = {}  # Dict to hold BEV data with fco_id as key

        for fco in fco_vehicles:
            self._prepare_nn_data(fco)
        print('----------------')
        print(f'Prepared data in {time.time() - t} seconds.')
        t = time.time()
        
        # Create the BEV images
        self._generate_bev_images()

        print(f'Generated BEV images in {time.time() - t} seconds.')

        # Iterate through the dataset and get the detectability label
        detection_results = {}
        total_samples = len(self.dataset)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size  # Ceiling division

        total_decisions = 0
        for i in range(num_batches):
            current_data = self.dataset[i * self.batch_size: (i + 1) * self.batch_size]
            bev_data = []
            vector_data = []
            fco_ids = []
            vehicle_ids = []

            for data in current_data:
                bev_data.append(self.bev_tensors[data['bev_pointer']])
                vector_data.append(data['vector'])
                fco_ids.append(data['fco_id'])
                vehicle_ids.append(data['vehicle_id'])
            
            # Convert to torch tensors and move to device
            bev_data_tensor = torch.stack(bev_data).to(self.device)
            vector_data_tensor = torch.tensor(vector_data, dtype=torch.float32).to(self.device)

            # Pass through the neural network
            outputs = self.model(bev_data_tensor, vector_data_tensor/50)
            
            # Run the results through a torch sigmoid function
            outputs = torch.sigmoid(outputs).cpu().detach().numpy()

            for idx, result in enumerate(outputs):
                fco_id = fco_ids[idx]
                if fco_id not in detection_results:
                    detection_results[fco_id] = []
                if result >= 0.5:
                    detection_results[fco_id].append(vehicle_ids[idx])
            
            total_decisions += len(outputs)

        total_detections = sum([len(detections) for detections in detection_results.values()])
        print(f'Executed NN in {time.time() - t} seconds for {len(self.bev_data)} fcos and {total_samples} total detection decisions and {total_detections} detections from {total_decisions} decisions.')
        print('----------------')
        return detection_results

    def _generate_bev_images(self):
        self.bev_tensors = {}

        for fco_id, data in self.bev_data.items():
            bev_tensor = create_bev_tensor(
                building_polygons=data['building_polygons'],
                vehicle_infos=data['vehicle_infos'],
                rotation_angle=0,
                x_offset=data['ego_x'],
                y_offset=data['ego_y'],
                image_size=self.image_size,
                radius_covered=self.distance,
                vehicle_representation='box'
            )
            if isinstance(bev_tensor, np.ndarray):
                bev_tensor = torch.from_numpy(bev_tensor)
            self.bev_tensors[fco_id] = bev_tensor

    def _prepare_nn_data(self, fco_id):
        x, y = traci.vehicle.getPosition(fco_id)
        alpha = traci.vehicle.getAngle(fco_id)

        ego_info = {
            'position': [x, y],
            'angle': alpha,
            'width': traci.vehicle.getWidth(fco_id),
            'length': traci.vehicle.getLength(fco_id),
        }

        # Get positions and IDs of all vehicles except fco_id
        all_vehicle_ids = list(traci.vehicle.getIDList())
        all_vehicle_ids.remove(fco_id)

        # Collect positions
        positions = np.array([traci.vehicle.getPosition(v) for v in all_vehicle_ids])
        position_diffs = positions - np.array([x, y])
        distances = np.linalg.norm(position_diffs, axis=1)

        # Get indices of vehicles within distance
        within_distance_indices = np.where(distances < self.distance)[0]
        relevant_vehicle_ids = [all_vehicle_ids[i] for i in within_distance_indices]
        relevant_positions = positions[within_distance_indices]
        relevant_diffs = position_diffs[within_distance_indices]

        # Collect other data for relevant vehicles
        relevant_vehicle_data = {}
        for idx, v in enumerate(relevant_vehicle_ids):
            vx, vy = relevant_positions[idx]
            v_data = {
                'position': [vx, vy],
                'angle': traci.vehicle.getAngle(v),
                'type': traci.vehicle.getTypeID(v),
                'width': traci.vehicle.getWidth(v),
                'length': traci.vehicle.getLength(v),
                'vector': relevant_diffs[idx],
                'id': v
            }
            relevant_vehicle_data[v] = v_data

        # Get building polygons within distance
        centroid_diffs = self.building_polygon_centroids - np.array([x, y])
        centroid_distances = np.linalg.norm(centroid_diffs, axis=1)
        within_distance_indices = np.where(centroid_distances < self.distance)[0]
        building_polygon_dict = {f'building_{i}': self.building_polygons[i] for i in within_distance_indices}

        # Prepare vehicle BEV data
        vehicle_bev_data = {v: {key: value for key, value in data.items() if key not in ['vector', 'id']} for v, data in relevant_vehicle_data.items()}
        vehicle_bev_data[fco_id] = ego_info  # Add ego vehicle info

        bev_entry = {
            'id': fco_id,
            'building_polygons': building_polygon_dict,
            'vehicle_infos': vehicle_bev_data,
            'ego_x': x,
            'ego_y': y,
            'ego_angle': alpha,
        }

        self.bev_data[fco_id] = bev_entry  # Store in dict with fco_id as key

        # Prepare dataset entries
        for v, data in relevant_vehicle_data.items():
            dataset_entry = {
                'fco_id': fco_id,
                'vehicle_id': v,
                'vector': data['vector'],
                'bev_pointer': fco_id,
                'type': data['type']
            }
            self.dataset.append(dataset_entry)

class DistanceDetector(Detector):
    def __init__(self, max_distance=50, log_duration=False, **kwargs):
        """
        Initializes the DistanceDetector.

        Parameters:
            max_distance (int): The maximum distance to detect other vehicles.
            log_duration (bool): Whether to log the duration of the detection process.
        """
        self.max_distance = max_distance
        self.durations = [] if log_duration else None

    def detect(self, vehicle_id: str) -> List[str]:
        """
        Detects vehicles within a certain distance from the ego vehicle.

        Parameters:
            vehicle_id (str): The ID of the ego vehicle.

        Returns:
            List[str]: IDs of detected vehicles within the max_distance.
        """
        t_start = time.time()

        if vehicle_id not in traci.vehicle.getIDList():
            print(f'Vehicle {vehicle_id} not in simulation.')
            if self.durations is not None:
                self.durations.append(time.time() - t_start)
            return []

        ego_position = np.array(traci.vehicle.getPosition(vehicle_id))
        detected_vehicles = []

        for other_vehicle_id in traci.vehicle.getIDList():
            if other_vehicle_id == vehicle_id:
                continue
            other_position = np.array(traci.vehicle.getPosition(other_vehicle_id))
            distance = np.linalg.norm(other_position - ego_position)
            if distance <= self.max_distance:
                detected_vehicles.append(other_vehicle_id)

        if self.durations is not None:
            self.durations.append(time.time() - t_start)

        return detected_vehicles

class D2RaytracingDetector(Detector):
    def __init__(self, building_polygons: Optional[Dict[str, Polygon]]=None, max_distance: int=50, min_hit_threshold: int=1, num_rays: int=360, visualize: bool=False, log_duration: bool=False, **kwargs):
        """
        Initializes the RayTracingDetector.
        
        Parameters:
            building_polygons: Polygons of buildings for ray tracing.
            max_distance (int): The length of the rays i.e also the maximum distance to detect objects.
            min_hit_threshold (int): Minimum number of ray hits to consider an object detected.
            num_rays (int): Number of rays to cast wich will be split evenly in 360 degrees.
            visualize (bool): Whether to visualize the detection.
            log_duration (bool): Whether to log the duration of the detection.
        """
        self.building_polygons = {} if building_polygons is None else building_polygons
        self.max_distance = max_distance
        self.min_hit_threshold = min_hit_threshold
        self.num_rays = num_rays
        self.visualize = visualize
        self.durations = [] if log_duration else None
    
    def detect(self, fco_vehicles: List[str]) -> Dict[str, List[str]]:
        """
        Detects vehicles using ray tracing.
        Args: 
            vehicle_id (str): The ID of the ego vehicle that currently performs the detection and acts as FCO
        
        Returns:
            Tuple[List[str], List[str]]: Detected and undetected vehicle IDs.
        """
        t_start = time.time()
        fco_detections = {}
        for fco in fco_vehicles:
            polygon_vehicles = {}
            ego_position = traci.vehicle.getPosition(fco)
            x,y = ego_position
            filtered_vehicles =  [v for v in traci.vehicle.getIDList() if v != fco and np.linalg.norm(np.array(traci.vehicle.getPosition(v)) - np.array([x, y])) < self.max_distance]
            polygon_buildings = {}
            for counter, building in enumerate(self.building_polygons):
                for point in building.exterior.coords:
                    if math.hypot(point[0] - ego_position[0], point[1] - ego_position[1]) <= self.max_distance:
                        polygon_buildings[f'building_{counter}'] = building
                        break
            for vehicle in filtered_vehicles:
                polygon_vehicles[vehicle] = create_vehicle_polygon(vehicle)
            rays = generate_rays(ego_position, num_rays=self.num_rays, radius=self.max_distance)
            hit_rays = []
            hit_counter_vehicles = {vehicle: 0 for vehicle in filtered_vehicles}
            all_polygons = {**polygon_vehicles, **polygon_buildings}
            for ray in rays:
                closest_hit_object, closest_hit_coordinate = detect_intersections(ray, all_polygons)
                if closest_hit_object is not None:
                    if closest_hit_object in filtered_vehicles:
                        hit_counter_vehicles[closest_hit_object] += 1
                if closest_hit_coordinate is not None:
                    short_ray = (ray[0], closest_hit_coordinate)
                    hit_rays.append(short_ray)
                else:
                    hit_rays.append(ray)
            detected_vehicles = [vehicle for vehicle, hits in hit_counter_vehicles.items() if hits >= self.min_hit_threshold]

            # Log the duration that is needed for the detection
            if self.durations is not None:
                self.durations.append(time.time() - t_start)

            # Visualize the current detection
            if self.visualize:
                save_polygons_and_rays_to_file([*polygon_vehicles.values(), *polygon_buildings.values()], hit_rays)
            fco_detections[fco] = detected_vehicles
        return fco_detections

def detector_factory(mode: str, **kwargs) -> Detector:
    """
    Factory function to create a detector based on the mode.
    
    Parameters:
        mode (str): The detection mode ('cv', 'nn', 'distance', 'raytracing').
        **kwargs: Additional keyword arguments specific to each detector.
        
    Returns:
        Detector: An instance of a detector subclass.
    """
    if mode == '3d-raytracing':
        return D3RaytracingDetector(**kwargs)
    elif mode == 'emulation':
        return EmulationDetector(**kwargs)
    elif mode == 'distance':
        return DistanceDetector(**kwargs)
    elif mode == '2d-raytracing':
        return D2RaytracingDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detection mode: {mode}")


