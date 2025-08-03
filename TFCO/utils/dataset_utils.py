import os
import sys
import math
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List
from torch.utils.data import Dataset
from einops import rearrange

from utils.vehicle_filter import get_furthest_vehicles, get_nearest_vehicles, get_random_vehicles

try:
    from utils.bev_utils import create_bev_tensor
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.bev_utils import create_bev_tensor

class SequenceTfcoDataset(Dataset):
    def __init__(self, dataset_path: List[str], sequence_len: int = 10, max_vehicles: int = 100, min_timesteps_seen: int = 3, overlap_mode: bool = False, filter: list = None,
                 split: Optional[tuple] = None, radius: Optional[int] = None, centerpoint: Optional[Tuple[int, int]] = None, loop: Optional[int] = None):
        """
        Args:
            dataset_path (str): Path to the dataset folder
            sequence_len (int): Number of timesteps to consider in the sequence (ignored if pretrain is True)
            max_vehicles (int): Maximum number of vehicles to consider (required if using raw input/output)
            min_timesteps_seen (int): Minimum number of timesteps a vehicle must be seen to be considered as a target (i.e. target will be labelled as 1 else 2)
            overlap_mode (bool): Indicates whether overlap mode is activated 
            filter (list): List containing all necessary informations for filtering the vehicles 
        """
        if split is not None:
            assert len(split) == len(dataset_path), "Split must have the same length as the dataset_path"
        else:
            split = [None] * len(dataset_path)
        
        self.dataset = None
        for path, split in tqdm(zip(dataset_path, split), desc='Loading datasets'):
            name = path.split('/')[-1]
            current_dataset = pd.read_pickle(os.path.join(path, 'dataset.pkl'))
            current_dataset = current_dataset.reset_index(drop=True)
            current_dataset['dataset_name'] = name

            # only keep a split of the dataset for training/validation/testing
            current_dataset = self._split_dataset(current_dataset, split)

            # only keep the loop specified
            if loop is not None:
                current_dataset = current_dataset[current_dataset['loop'] == loop]

            if self.dataset is None:
                self.dataset = current_dataset
            else:
                self.dataset = pd.concat([self.dataset, current_dataset], ignore_index=True)

        
        if filter is not None:
            self.filter_vehicles = True
            self.filter_mode = filter[0]
            self.selection_mode = filter[1]
            self.k = filter[2]
        else:
            self.filter_vehicles = False
            self.filter_mode = "-"

        self.sequence_len = sequence_len
        self.max_vehicles = max_vehicles
        self.min_timesteps_seen = min_timesteps_seen

        try:
            with open(os.path.join(dataset_path[0], 'config.yaml'), 'r', encoding="utf8") as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            with open(os.path.join(dataset_path[0], 'config.pkl'), 'rb') as f:
                self.config = pd.read_pickle(f)
            print("Using pickle file for config")

        if radius is not None:
            self.radius = radius
        else:
            self.radius = self.config['radius']
        
        if centerpoint is not None and overlap_mode is False:
            self.center_point = centerpoint
        elif centerpoint is None and overlap_mode is True:
            self.pois = list()
            for poi in self.config["center_point"]:
                self.pois.append(poi)
        else:
            try:
                self.center_point = self.config['center_point']
            except:
                self.center_point = self.config['CENTER_POINT']

        self.max_vehicles_counter = 0

        base_root = Path(__file__).resolve().parents[3]
        with open (os.path.join(base_root, "data", "stats", "mean.npy"), "rb") as m:
            self.mean = float(np.load(m))

        with open (os.path.join(base_root, "data", "stats", "std.npy"), "rb") as s:
            self.std = float(np.load(s))

        self._get_allowed_indexes()  # Will create self.allowed_indexes

        self._create_inputs(overlap_mode)   # Will create self.input_tensors
        self._create_targets(overlap_mode)  # Will create self.target_tensors

        print(f"Max vehicles in the dataset: {self.max_vehicles_counter}")
        print(f"Overlap mode: {overlap_mode}")
        print(f"Filter mode: {self.filter_vehicles}")
        print(f"target_len {len(self.target_information)} and input len {len(self.input_information)}")

    def __len__(self):
        return len(self.allowed_indexes)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            idx (int): Index of the dataset to retrieve
        Returns:
            input_tensor (torch.Tensor): Input tensor of shape (sequence_len, max_vehicles, 3)
            target_tensor (torch.Tensor): Target tensor of shape (max_vehicles, 3)
            index (int): Index of the dataset
        """
        index = self.allowed_indexes[idx]
        datapoint_id = self.dataset.loc[index, 'id']

        # Get the sequence of datapoint IDs
        sequence_indexes = list(range(index - self.sequence_len + 1, index + 1))
        sequence_datapoint_ids = [self.dataset.loc[i, 'id'] for i in sequence_indexes]

        # Retrieve target vehicles --> all vehilces in the system in the current timestep
        target_vehicles = self.target_information.get(datapoint_id, {})
        target_vehicle_ids = list(target_vehicles.keys())
        num_target_vehicles = len(target_vehicle_ids)

        # Ensure the number of vehicles does not exceed max_vehicles
        assert num_target_vehicles <= self.max_vehicles

        # Prepare the target tensor
        # If target_vehicles is empty, create a tensor of zeros
        if num_target_vehicles == 0:
            target_tensor = torch.zeros(self.max_vehicles, 3)
        else:
            # Pad the vehicle IDs list to have length max_vehicles
            padded_vehicle_ids = target_vehicle_ids + [0] * (self.max_vehicles - num_target_vehicles)
            target_tensor = torch.stack(
                [target_vehicles.get(i, torch.zeros(3)) for i in padded_vehicle_ids]
            )  # Shape: (max_vehicles, 3)

        # Prepare the input tensor
        input_tensor_list = []
        for vehicle_id in target_vehicle_ids:
            vehicle_sequence = []
            for sequence_datapoint_id in sequence_datapoint_ids:
                vehicle_info = self.input_information.get(sequence_datapoint_id, {})
                if vehicle_id in vehicle_info:
                    # Vehicle is visible in the current timestep
                    vehicle_sequence.append(vehicle_info[vehicle_id])
                else:
                    # Vehicle is not visible in the current timestep
                    vehicle_sequence.append(torch.tensor([-1.0, 0.0, 0.0]))
            vehicle_sequence = torch.stack(vehicle_sequence)  # Shape: (sequence_len, 3)
            input_tensor_list.append(vehicle_sequence)

        # If input_tensor_list is empty, create a zero tensor --> no target vehicles in the scene
        if len(input_tensor_list) == 0:
            input_tensor = torch.zeros(self.sequence_len, self.max_vehicles, 3)
        else:
            # Stack the input tensors and rearrange
            input_tensor_stacked = torch.stack(input_tensor_list)  # Shape: (num_target_vehicles, sequence_len, 3)
            input_tensor_stacked = rearrange(input_tensor_stacked, 'v s c -> s v c')  # Shape: (sequence_len, num_target_vehicles, 3)
            # Initialize the full input tensor with zeros
            input_tensor = torch.zeros(self.sequence_len, self.max_vehicles, 3)
            # Fill in the available vehicle data
            input_tensor[:, :num_target_vehicles, :] = input_tensor_stacked
        
        # 3: vehicles currently seen in the input
        # 2: vehicles not seen at enough timesteps based on min_timesteps_seen
        # 1: vehicles seen at some point in the past but not in the current timestep
        # 0: zero vehicles i.e. just the padding to fill the max_vehicles
        zero_vehicles = target_tensor[:, 0] == 0
        currently_seen_vehicles = input_tensor[-1, :, 0] == 1
        seen_vehicle_counts = torch.sum(input_tensor[:, :, 0] == 1, dim=0)

        # Vehicles not seen at enough timesteps based on min_timesteps_seen
        unseen_vehicles = seen_vehicle_counts < self.min_timesteps_seen
        target_tensor[unseen_vehicles, 0] = 2

        # Vehicles seen at some point in the past but not in the current timestep for at least min_timesteps_seen
        in_sequence_vehicles = seen_vehicle_counts >= self.min_timesteps_seen
        target_tensor[in_sequence_vehicles, 0] = 1

        # Overwrite the target tensor with vehicles currently seen in the input
        target_tensor[currently_seen_vehicles, 0] = 3

        # Overwrite the target tensor with the zero vehicles
        target_tensor[zero_vehicles, 0] = 0

        #print(f"Size Input: {input_tensor.size()}, Size target: {target_tensor.size()}")

        return input_tensor, target_tensor, index

    def _get_allowed_indexes(self):
        # Initialize allowed indexes
        self.allowed_indexes = []
        
        # Group the dataset by 'dataset_name' and 'loop'
        grouped = self.dataset.groupby(['dataset_name', 'loop'])
        
        for (dataset_name, loop), group in tqdm(grouped, desc='Finding allowed indexes'):
            # Sort the group by 'timestep'
            group = group.sort_values('timestep').reset_index()
            indices = group['index'].tolist()
            
            # Generate allowed indexes within each combination of 'dataset_name' and 'loop'
            if len(indices) >= self.sequence_len:
                # We start from index sequence_len - 1 to ensure we have enough previous timesteps
                for i in range(self.sequence_len - 1, len(indices)):
                    self.allowed_indexes.append(indices[i])

    def _create_inputs(self, overlap_mode):
        # Prepare input tensors for all data points
        if overlap_mode:
            self.input_information = {}

            for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):

                detected_vehicles = {
                    key: vehicle for key, vehicle in data.vehicle_information.items()
                    if vehicle.get('detected_label') == 1 or vehicle.get('fco_label') == 1
                }

                processed_vehicle_information = {}

                for poi in self.pois:
                    for vehicle_id, vehicle_data in detected_vehicles.items():
                        # Normalize vehicle positions relative to the center point and radius
                        normalized_position = [
                            1,  # Assuming this is a fixed label or value
                                                                #poi
                            (vehicle_data['position'][0] - poi[0]) / self.radius,
                            (vehicle_data['position'][1] - poi[1]) / self.radius,
                        ]
                        
                        # Calculate the distance and filter vehicles within a unit radius
                        distance = math.sqrt(normalized_position[1]**2 + normalized_position[2]**2)
                        if distance <= 1:
                            processed_vehicle_information[vehicle_id] = torch.tensor(normalized_position)

                    # Store processed vehicle information for this data point
                    self.input_information[data.id] = processed_vehicle_information
            
        else:
            self.input_information = {}

            for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
                detected_vehicles = {
                    key: vehicle for key, vehicle in data.vehicle_information.items()
                    if vehicle.get('detected_label') == 1 or vehicle.get('fco_label') == 1
                }

                processed_vehicle_information = {}
                for vehicle_id, vehicle_data in detected_vehicles.items():
                    # Normalize vehicle positions relative to the center point and radius
                    #normalized_position = [
                    #    1,  # Assuming this is a fixed label or value
                                                            #poi
                    #    (vehicle_data['position'][0] - self.center_point[0]) / self.radius,
                    #    (vehicle_data['position'][1] - self.center_point[1]) / self.radius,
                    #]
                    
                    # Calculate the distance and filter vehicles within a unit radius
                    #distance = math.sqrt(normalized_position[1]**2 + normalized_position[2]**2)
                    normalized_position = [1, (vehicle_data["position"][0] - self.mean) / self.std,
                            (vehicle_data["position"][1] - self.mean) / self.std,]
                    #if distance <= 1:
                    processed_vehicle_information[vehicle_id] = torch.tensor(normalized_position)

                if self.filter_vehicles:
                    if self.selection_mode == "nearest":
                        processed_vehicle_information = get_nearest_vehicles(processed_vehicle_information, self.filter_mode, self.k)
                    elif self.selection_mode == "random":
                        processed_vehicle_information = get_random_vehicles(processed_vehicle_information, self.filter_mode, self.k)
                    elif self.selection_mode == "furthest":
                        processed_vehicle_information = get_furthest_vehicles(processed_vehicle_information, self.filter_mode, self.k)
                    else:
                        assert "No correct mode was given!"

                    # 0 or k --> exact
                    self.input_information[data.id] = processed_vehicle_information
                else:
                    self.input_information[data.id] = processed_vehicle_information



    def _create_targets(self, overlap_mode):
        # Prepare input tensors for all data points
        if overlap_mode:
            self.target_information = {}

            for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
                processed_vehicle_information = {}
                
                for poi in self.pois:
                    for vehicle_id, vehicle_data in data.vehicle_information.items():
                        # Normalize vehicle positions relative to the center point and radius
                                                                #pos
                        #normalized_position = [1,
                        #    (vehicle_data['position'][0] - poi[0]) / self.radius,
                        #    (vehicle_data['position'][1] - poi[1]) / self.radius,
                        #]
                        
                        # Calculate the distance and filter vehicles within a unit radius
                        #distance = math.sqrt(normalized_position[1]**2 + normalized_position[2]**2)
                        #if distance <= 1:
                        normalized_position = [1, (vehicle_data["position"][0] - 198.52158002732125) / 496.2554675738291,
                                               (vehicle_data["position"][1] - 198.52158002732125) / 496.2554675738291]

                        processed_vehicle_information[vehicle_id] = torch.tensor(normalized_position)

                    # Store processed vehicle information for this data point
                    self.target_information[data.id] = processed_vehicle_information
                    assert len(processed_vehicle_information) <= self.max_vehicles # Ensure the number of vehicles does not exceed max_vehicles
                    if len(processed_vehicle_information) > self.max_vehicles_counter:
                        self.max_vehicles_counter = len(processed_vehicle_information)
        else:
            self.target_information = {}

            for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
                processed_vehicle_information = {}
                for vehicle_id, vehicle_data in data.vehicle_information.items():
                    # Normalize vehicle positions relative to the center point and radius
                                                            #pos
                    #normalized_position = [1,
                    #    (vehicle_data['position'][0] - self.center_point[0]) / self.radius,
                    #    (vehicle_data['position'][1] - self.center_point[1]) / self.radius,
                    #]
                    
                    # Calculate the distance and filter vehicles within a unit radius
                    #distance = math.sqrt(normalized_position[1]**2 + normalized_position[2]**2)
                    #if distance <= 1:
                    normalized_position = [1, (vehicle_data["position"][0] - self.mean) / self.std,
                                               (vehicle_data["position"][1] - self.mean) / self.std,]
                    processed_vehicle_information[vehicle_id] = torch.tensor(normalized_position)

                   #Filter
                if self.filter_vehicles:
                    if self.selection_mode == "nearest":
                        processed_vehicle_information = get_nearest_vehicles(processed_vehicle_information, self.filter_mode, self.k)
                    elif self.selection_mode == "random":
                        processed_vehicle_information = get_random_vehicles(processed_vehicle_information, self.filter_mode, self.k)
                    elif self.selection_mode == "furthest":
                        processed_vehicle_information = get_furthest_vehicles(processed_vehicle_information, self.filter_mode, self.k)
                    else:
                        assert "No correct mode was given!"

                    #if len(processed_vehicle_information) == 0:
                    #--> Exact mode only 0 or k
                    #else:
                    self.target_information[data.id] = processed_vehicle_information
                    assert len(processed_vehicle_information) <= self.max_vehicles 
                    if len(processed_vehicle_information) > self.max_vehicles_counter:
                        self.max_vehicles_counter = len(processed_vehicle_information)
                else:
                    self.target_information[data.id] = processed_vehicle_information
                    assert len(processed_vehicle_information) <= self.max_vehicles 
                    if len(processed_vehicle_information) > self.max_vehicles_counter:
                            self.max_vehicles_counter = len(processed_vehicle_information)
    
    
    def _split_dataset(self, dataset, splits):
        if splits is not None:
            if isinstance(splits, list):
                datasets = []
                for split in splits:
                    start = 0 if split[0] == ':' else (
                        int(split[0] * len(dataset)) if isinstance(split[0], float) else split[0]
                    )
                    end = len(dataset) if split[1] == ':' else (
                        int(split[1] * len(dataset)) if isinstance(split[1], float) else split[1]
                    )
                    datasets.append(dataset.iloc[start:end])
                return pd.concat(datasets, ignore_index=True)   
            else:
                start = 0 if splits[0] == ':' else (
                    int(splits[0] * len(dataset)) if isinstance(splits[0], float) else splits[0]
                )
                end = len(dataset) if splits[1] == ':' else (
                    int(splits[1] * len(dataset)) if isinstance(splits[1], float) else splits[1]
                )
                dataset = dataset.iloc[start:end]
                return dataset
        else:
            return dataset

class TfcoDataset(Dataset):
    def __init__(self, dataset_path: str, sequence_len: int = 5, image_size: Optional[int] = None, max_vehicles: Optional[int] = None,
                 pre_train: bool = False, input_type: str = 'bev', output_type: str = 'bev'):
        """
        Args:
            dataset_path (str): Path to the dataset folder
            sequence_len (int): Number of timesteps to consider in the sequence (ignored if pretrain is True)
            image_size (int): Size of the BEV image (only required if using BEV input/output)
            max_vehicles (int): Maximum number of vehicles to consider (required if using raw input/output)
            pretrain (bool): If True, returns the same data as input and target for pretraining
            input_type (str): 'bev' or 'raw' to specify input data type
            output_type (str): 'bev' or 'raw' to specify output data type
        """
        self.dataset_path = dataset_path
        self.sequence_len = sequence_len
        self.image_size = image_size
        self.max_vehicles = max_vehicles
        self.pretrain = pre_train
        self.input_type = input_type
        self.output_type = output_type

        assert input_type in ['bev', 'raw'], "input_type must be 'bev' or 'raw'"
        assert output_type in ['bev', 'raw'], "output_type must be 'bev' or 'raw'"
        assert max_vehicles is not None if input_type == 'raw' or output_type == 'raw' else True, "max_vehicles must be provided when using raw input or output"
        if input_type == 'bev' or output_type == 'bev':
            assert image_size is not None, "image_size must be provided when using BEV input or output"

        self.dataset = pd.read_pickle(os.path.join(dataset_path, 'dataset.pkl'))
        self.dataset = self.dataset.reset_index(drop=True)
        self.config = pd.read_pickle(os.path.join(dataset_path, 'config.pkl'))

        self.radius = self.config['RADIUS']
        self.center_point = self.config['CENTER_POINT']

        if not self.pretrain:
            self._get_allowed_indexes()  # Will create self.allowed_indexes
        else:
            self.allowed_indexes = list(range(len(self.dataset)))

        self._create_inputs()   # Will create self.input_tensors
        self._create_targets()  # Will create self.target_tensors

    def __len__(self):
        return len(self.allowed_indexes)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            idx (int): Index of the dataset to retrieve
        Returns:
            input_tensor (torch.Tensor): Input tensor of shape (sequence_len, 1, image_size, image_size) if input_type is 'bev',
                                         or (sequence_len, max_vehicles, 3) if input_type is 'raw'
            target_tensor (torch.Tensor): Target tensor of shape (1, image_size, image_size) if output_type is 'bev',
                                          or (max_vehicles, 3) if output_type is '
            
            index (int): Index of the dataset
        """
        index = self.allowed_indexes[idx]
        datapoint_id = self.dataset.loc[index, 'id']

        if not self.pretrain:
            sequence_indexes = list(range(index - self.sequence_len + 1, index + 1))
            sequence_datapoint_ids = [self.dataset.loc[i, 'id'] for i in sequence_indexes]

            # Prepare input
            if self.input_type == 'bev':
                input_tensor = torch.stack([self.input_tensors[_id] for _id in sequence_datapoint_ids])  # sequence_len x 1 x image_size x image_size
            elif self.input_type == 'raw':
                input_tensor = torch.zeros(self.sequence_len, self.max_vehicles, 3)
                for i, _id in enumerate(sequence_datapoint_ids):
                    curr_input = self.input_tensors[_id]
                    num_vehicles = curr_input.shape[0]
                    if num_vehicles > 0:
                        input_tensor[i, :num_vehicles, :] = curr_input
            else:
                raise ValueError("Invalid input_type")

            # Prepare target
            target_tensor = self.target_tensors[datapoint_id]
            if self.output_type == 'bev':
                pass  # target_tensor is already correct
            elif self.output_type == 'raw':
                num_vehicles = target_tensor.shape[0]
                full_target_tensor = torch.zeros(self.max_vehicles, 3)
                if num_vehicles > 0:
                    full_target_tensor[:num_vehicles, :] = target_tensor
                target_tensor = full_target_tensor
            else:
                raise ValueError("Invalid output_type")
        else:
            # Pretraining: input and target are from the same timestep
            if self.input_type == 'bev':
                input_tensor = self.input_tensors[datapoint_id]
            elif self.input_type == 'raw':
                input_tensor = torch.zeros(1, self.max_vehicles, 3)
                curr_input = self.input_tensors[datapoint_id]
                num_vehicles = curr_input.shape[0]
                if num_vehicles > 0:
                    input_tensor[0, :num_vehicles, :] = curr_input
            else:
                raise ValueError("Invalid input_type")

            if self.output_type == 'bev':
                target_tensor = self.input_tensors[datapoint_id]
            elif self.output_type == 'raw':
                target_tensor = input_tensor.squeeze(0) # removing sequence len dimension
                first_column = ((target_tensor[:, 0] != 0) | (target_tensor[:, 1] != 0)).float().unsqueeze(1)
                target_tensor = torch.cat((first_column, target_tensor[:, :2]), dim=1)
            else:
                raise ValueError("Invalid output_type")

        return input_tensor, target_tensor, index

    def _get_allowed_indexes(self):
        # Initialize allowed indexes
        self.allowed_indexes = []
        # Iterate over the dataset for easier access to row elements
        for i, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc='Finding allowed indexes'):
            timestep = int(row['timestep'])
            loop = row['loop']
            # Check if any rows match the previous timestep for the same loop
            if ((self.dataset['timestep'] == timestep - self.sequence_len + 1) & (self.dataset['loop'] == loop)).any():
                self.allowed_indexes.append(i)

    def _create_inputs(self):
        # Create input tensors for all data points
        self.input_tensors = {}
        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc='Preparing input tensors'):
            detected_vehicles = {}
            for key, vehicle in data.vehicle_information.items():
                if vehicle.get('detected_label') == 1 or vehicle.get('fco_label') == 1:
                    detected_vehicles[key] = vehicle
            if self.input_type == 'bev':
                bev_tensor = create_bev_tensor(
                    building_polygons={},
                    vehicle_infos=detected_vehicles,
                    rotation_angle=0,
                    x_offset=self.center_point[0],
                    y_offset=self.center_point[1],
                    image_size=self.image_size,
                    vehicle_representation='box',
                    radius_covered=self.radius
                )
                self.input_tensors[data.id] = bev_tensor
            elif self.input_type == 'raw':
                vehicle_information = []
                for information in detected_vehicles.values():
                    current_information = [
                        (information['position'][0] - self.center_point[0]) / self.radius,
                        (information['position'][1] - self.center_point[1]) / self.radius,
                        information['angle'] / 360
                    ]
                    distance = math.sqrt(current_information[0] ** 2 + current_information[1] ** 2)
                    if distance <= 1:
                        vehicle_information.append(current_information)
                self.input_tensors[data.id] = torch.tensor(vehicle_information)
            else:
                raise ValueError("Invalid input_type")

    def _create_targets(self):
        # Create target tensors for all data points
        self.target_tensors = {}
        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc='Preparing target tensors'):
            if not self.pretrain and data.Index not in self.allowed_indexes:
                continue
            if self.pretrain:
                # Pretraining: target is same as input
                self.target_tensors[data.id] = self.input_tensors[data.id]
            else:
                # Collect vehicles detected over the sequence and present at current timestep
                detected_sequence_vehicles = set()
                timestep = data.timestep
                loop = data.loop
                current_vehicles = set(data.vehicle_information.keys())
                relevant_rows = self.dataset[
                    (self.dataset.timestep >= timestep - self.sequence_len + 1) &
                    (self.dataset.timestep <= timestep) &
                    (self.dataset.loop == loop)
                ]
                # Collect vehicle IDs from relevant rows
                for _, row in relevant_rows.iterrows():
                    detected_sequence_vehicles.update(row.vehicle_information.keys())
                # Retain vehicles present in current timestep
                detected_sequence_vehicles.intersection_update(current_vehicles)
                # Build dictionary of filtered current vehicles
                current_vehicles_dict = {key: data.vehicle_information[key] for key in detected_sequence_vehicles}
                if self.output_type == 'bev':
                    bev_tensor = create_bev_tensor(
                        building_polygons={},
                        vehicle_infos=current_vehicles_dict,
                        rotation_angle=0,
                        x_offset=self.center_point[0],
                        y_offset=self.center_point[1],
                        image_size=self.image_size,
                        vehicle_representation='box',
                        radius_covered=self.radius
                    )
                    self.target_tensors[data.id] = bev_tensor
                elif self.output_type == 'raw':
                    vehicle_information = []
                    for information in current_vehicles_dict.values():
                        current_information = [1,
                            (information['position'][0] - self.center_point[0]) / self.radius,
                            (information['position'][1] - self.center_point[1]) / self.radius,
                        ]
                        distance = math.sqrt(current_information[1] ** 2 + current_information[2] ** 2)
                        if distance <= 1:
                            vehicle_information.append(current_information)
                    self.target_tensors[data.id] = torch.tensor(vehicle_information)
                else:
                    raise ValueError("Invalid output_type")


if __name__ == "__main__":
    dataset_path = '/home/jeremias/tfco/solving_occlusion/datasets/test'
    dataset = TfcoDataset(dataset_path, 5)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for data in tqdm(dataloader):
        input_tensor, target_tensor = data
        print(input_tensor.shape)
        print(target_tensor.shape)

    

    
        


    