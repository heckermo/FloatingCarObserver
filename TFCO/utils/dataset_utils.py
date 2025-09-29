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
from utils.normalization_utils import load_normalization_stats

try:
    from utils.bev_utils import create_bev_tensor
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.bev_utils import create_bev_tensor

class SequenceTfcoDataset(Dataset):
    def __init__(self, dataset_path: List[str], sequence_len: int = 10, max_vehicles: int = 100, min_timesteps_seen: int = 3, filter: list = None,
                 split: Optional[tuple] = None, radius: Optional[int] = None, centerpoint: Optional[Tuple[int, int]] = None, loop: Optional[int] = None, normalization: str = "radius"):
        """
        Args:
            dataset_path (str): Path to the dataset folder
            sequence_len (int): Number of timesteps to consider in the sequence (ignored if pretrain is True)
            max_vehicles (int): Maximum number of vehicles to consider (required if using raw input/output)
            min_timesteps_seen (int): Minimum number of timesteps a vehicle must be seen to be considered as a target (i.e. target will be labelled as 1 else 2)
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



        #Extract all necessary parameters for vehicle filtering         
        if filter is not None:
            
            self.filter_vehicles = True
            self.filter_mode = filter[0]
            self.filter_selection_mode = filter[1]
            self.filter_k = filter[2]
        
        else:
            
            self.filter_vehicles = False


        self.sequence_len = sequence_len
        self.max_vehicles = max_vehicles
        self.min_timesteps_seen = min_timesteps_seen
        self.normalization = normalization


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
        

        if centerpoint is not None:
            self.center_point = centerpoint
        else:
            try:
                self.center_point = self.config['center_point']
            except:
                self.center_point = self.config['CENTER_POINT']

        self.max_vehicles_counter = 0


        if normalization == "zscore":
            # Normalize vehicle positions using z-score

            self.mean, self.std = load_normalization_stats(dataset_path)
        
        elif normalization == "radius":
            # Normalize vehicle positions relative to the center point and radius  
            # For clarity and to reduce code overhead, the mean is set to the center point and the std to the radius

            self.mean = self.center_point
            self.std = (self.radius, self.radius)
        
        else:
            raise ValueError(f"Error: No suitable normalization in config found: {normalization}")



        self._get_allowed_indexes()  # Will create self.allowed_indexes
        self._create_inputs()   # Will create self.input_tensors
        self._create_targets()  # Will create self.target_tensors


        print(f"Max vehicles in the dataset: {self.max_vehicles_counter}")
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

    def _create_inputs(self):
        # Prepare input tensors for all data points
        
        self.input_information = {}

        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
            detected_vehicles = {
                key: vehicle for key, vehicle in data.vehicle_information.items()
                if vehicle.get('detected_label') == 1 or vehicle.get('fco_label') == 1
            }
            processed_vehicle_information = {}
            
            for vehicle_id, vehicle_data in detected_vehicles.items(): 
                
                normalized_position = [1,
                        (vehicle_data["position"][0] - self.mean[0]) / self.std[0],
                        (vehicle_data["position"][1] - self.mean[1]) / self.std[1],
                        ]
            
                processed_vehicle_information[vehicle_id] = torch.tensor(normalized_position)

            
            #If Filter is true, the vehicles are filtered according to the given paramters 
            if self.filter_vehicles:
                self.input_information[data.id] = self._get_filtered_vehicle_information(processed_vehicle_information)
            
            else:
                self.input_information[data.id] = processed_vehicle_information



    def _create_targets(self):
        # Prepare input tensors for all data points
        
        self.target_information = {}

        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
            processed_vehicle_information = {}
            
            for vehicle_id, vehicle_data in data.vehicle_information.items():
                normalized_position = [1,
                            (vehicle_data["position"][0] - self.mean[0]) / self.std[0],
                            (vehicle_data["position"][1] - self.mean[1]) / self.std[1],
                            ]
            
                processed_vehicle_information[vehicle_id] = torch.tensor(normalized_position)

            
            #If Filter is true, the vehicles are filtered according to the given paramters 
            if self.filter_vehicles:

                processed_vehicle_information = self._get_filtered_vehicle_information(processed_vehicle_information)
                self.target_information[data.id] = processed_vehicle_information

                assert len(processed_vehicle_information) <= self.max_vehicles 
                
                if len(processed_vehicle_information) > self.max_vehicles_counter:
                    self.max_vehicles_counter = len(processed_vehicle_information)
            
            else:
                
                self.target_information[data.id] = processed_vehicle_information
                
                assert len(processed_vehicle_information) <= self.max_vehicles, f"current max vehicles: {len(processed_vehicle_information)}" 
                
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
        
    def _get_filtered_vehicle_information(self, processed_vehicle_information):

        """
        Function 
        """

        if self.filter_selection_mode == "nearest":
            filtered_vehicle_information = get_nearest_vehicles(processed_vehicle_information, self.filter_mode, self.filter_k)
        
        elif self.filter_selection_mode == "random":
            filtered_vehicle_information = get_random_vehicles(processed_vehicle_information, self.filter_mode, self.filter_k)
        
        elif self.filter_selection_mode == "furthest":
            filtered_vehicle_information = get_furthest_vehicles(processed_vehicle_information, self.filter_mode, self.filter_k)
        
        else:
            raise ValueError(f"No correct mode was given! Check config {self.filter_selection_mode}")

        return filtered_vehicle_information
    




class SequenceTfcoDatasetOverlap(Dataset):
    def __init__(self, dataset_path: List[str], sequence_len: int = 10, max_vehicles: int = 100, min_timesteps_seen: int = 3, filter: list = None,
                 split: Optional[tuple] = None, radius: Optional[int] = None, centerpoint: Optional[Tuple[int, int]] = None, loop: Optional[int] = None, normalization: str = "radius"):
        """
        Args:
            dataset_path (str): Path to the dataset folder
            sequence_len (int): Number of timesteps to consider in the sequence (ignored if pretrain is True)
            max_vehicles (int): Maximum number of vehicles to consider (required if using raw input/output)
            min_timesteps_seen (int): Minimum number of timesteps a vehicle must be seen to be considered as a target (i.e. target will be labelled as 1 else 2)
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



        #Extract all necessary parameters for vehicle filtering         
        if filter is not None:
            
            self.filter_vehicles = True
            self.filter_mode = filter[0]
            self.filter_selection_mode = filter[1]
            self.filter_k = filter[2]
        
        else:
            
            self.filter_vehicles = False


        self.sequence_len = sequence_len
        self.max_vehicles = max_vehicles
        self.min_timesteps_seen = min_timesteps_seen
        self.normalization = normalization

        self.input_dim = 3
        self.target_dim = 3


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
        

        if centerpoint is not None:
            self.center_point = centerpoint
        else:
            try:
                self.center_point = self.config['center_point']
            except:
                self.center_point = self.config['CENTER_POINT']

        self.max_vehicles_counter = 0


        if normalization == "zscore":
            # Normalize vehicle positions using z-score

            self.mean, self.std = load_normalization_stats(dataset_path)
        
        elif normalization == "radius":
            # Normalize vehicle positions relative to the center point and radius  
            # For clarity and to reduce code overhead, the mean is set to the center point and the std to the radius

            self.mean = self.center_point
            self.std = (self.radius, self.radius)
        
        else:
            raise ValueError(f"Error: No suitable normalization in config found: {normalization}")



        self._get_allowed_indexes()  # Will create self.allowed_indexes
        self._create_inputs()   # Will create self.input_tensors
        self._create_targets()  # Will create self.target_tensors


        print(f"Max vehicles in the dataset: {self.max_vehicles_counter}")
        print(f"Filter mode: {self.filter_vehicles}")
        print(f"Target len {len(self.target_information)} and input len {len(self.input_information)}")

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

        features = list()
        overlap_tag = list()
        poi_id = list()

        for vehicle_id in target_vehicle_ids:
            vehicle_sequence_features = list()
            vehicle_sequence_overlap = list()
            vehicle_sequence_poi = list()

            for sequence_datapoint_id in sequence_datapoint_ids:
                vehicle_info = self.input_information.get(sequence_datapoint_id, {})
                if vehicle_id in vehicle_info:
                    # Vehicle is visible in the current timestep
                    vehicle_sequence_features.append(vehicle_info[vehicle_id]["features"])
                    vehicle_sequence_overlap.append(vehicle_info[vehicle_id]["overlap_tag"])
                    vehicle_sequence_poi.append(vehicle_info[vehicle_id]["poi_id"])
                else:
                    # Vehicle is not visible in the current timestep
                    vehicle_sequence_features.append(torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32))
                    vehicle_sequence_overlap.append(-1)
                    vehicle_sequence_poi.append(-1)

            features.append(torch.stack(vehicle_sequence_features))  # Shape: (sequence_len, 3)
            overlap_tag.append(torch.tensor(vehicle_sequence_overlap, dtype=torch.long))
            poi_id.append(torch.tensor(vehicle_sequence_poi, dtype=torch.long))

        # If input_tensor_list is empty, create a zero tensor --> no target vehicles in the scene
        if len(features) == 0:
            input_tensor = torch.zeros(self.sequence_len, self.max_vehicles, self.input_dim, dtype=torch.float32)
            overlap_tensor = torch.full((self.sequence_len, self.max_vehicles), -1, dtype=torch.long)
            poi_id_tensor = torch.full((self.sequence_len, self.max_vehicles), -1, dtype=torch.long)
        else:
            # Stack the input tensors and rearrange
            input_tensor_stacked = torch.stack(features)  # Shape: (num_target_vehicles, sequence_len, 3)
            overlap_tensor_stacked = torch.stack(overlap_tag, dim=1)
            poi_id_tensor_stacked = torch.stack(poi_id, dim=1)
            
            input_tensor_stacked = rearrange(input_tensor_stacked, 'v s c -> s v c')  # Shape: (sequence_len, num_target_vehicles, 3)
            # Initialize the full input tensor with zeros
            input_tensor = torch.zeros(self.sequence_len, self.max_vehicles, self.input_dim)
            overlap_tensor = torch.full((self.sequence_len, self.max_vehicles), -1, dtype=torch.long)
            poi_id_tensor = torch.full((self.sequence_len, self.max_vehicles), -1, dtype=torch.long)
            # Fill in the available vehicle data
            input_tensor[:, :num_target_vehicles, :] = input_tensor_stacked
            overlap_tensor[:, :num_target_vehicles] = overlap_tensor_stacked
            poi_id_tensor[:, :num_target_vehicles] = poi_id_tensor_stacked
        
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

        return {"features": input_tensor, "overlap_tag": overlap_tensor, "poi_id": poi_id_tensor}, target_tensor, index

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

    def _create_inputs(self):
        # Prepare input tensors for all data points
        
        self.input_information = {}

        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
            detected_vehicles = {
                key: vehicle for key, vehicle in data.vehicle_information.items()
                if vehicle.get('detected_label') == 1 or vehicle.get('fco_label') == 1
            }
            processed_vehicle_information = {}
            
            for vehicle_id, vehicle_data in detected_vehicles.items(): 
                
                normalized_features = [1,
                        (vehicle_data["position"][0] - self.mean[0]) / self.std[0],
                        (vehicle_data["position"][1] - self.mean[1]) / self.std[1],
                        ]
                
                overlap_tag = int(vehicle_data["overlap_tag"])
                poi_id = int(vehicle_data["poi_id"])
            
                processed_vehicle_information[vehicle_id] = {"features": torch.tensor(normalized_features, dtype=torch.float32),
                                                             "overlap_tag": overlap_tag,
                                                             "poi_id": poi_id,
                                                            }

            
            #If Filter is true, the vehicles are filtered according to the given paramters 
            if self.filter_vehicles:
                self.input_information[data.id] = self._get_filtered_vehicle_information(processed_vehicle_information)
            
            else:
                self.input_information[data.id] = processed_vehicle_information



    def _create_targets(self):
        # Prepare input tensors for all data points
        
        self.target_information = {}

        for data in tqdm(self.dataset.itertuples(), total=len(self.dataset), desc="Preparing input tensors"):
            processed_vehicle_information = {}
            
            for vehicle_id, vehicle_data in data.vehicle_information.items():
                normalized_features = [1,
                            (vehicle_data["position"][0] - self.mean[0]) / self.std[0],
                            (vehicle_data["position"][1] - self.mean[1]) / self.std[1],
                            ]
            
                processed_vehicle_information[vehicle_id] = torch.tensor(normalized_features, dtype=torch.float32)

            
            #If Filter is true, the vehicles are filtered according to the given paramters 
            if self.filter_vehicles:

                processed_vehicle_information = self._get_filtered_vehicle_information(processed_vehicle_information)
                self.target_information[data.id] = processed_vehicle_information

                assert len(processed_vehicle_information) <= self.max_vehicles 
                
                if len(processed_vehicle_information) > self.max_vehicles_counter:
                    self.max_vehicles_counter = len(processed_vehicle_information)
            
            else:
                
                self.target_information[data.id] = processed_vehicle_information
                
                assert len(processed_vehicle_information) <= self.max_vehicles, f"current max vehicles: {len(processed_vehicle_information)}" 
                
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
        
    def _get_filtered_vehicle_information(self, processed_vehicle_information):

        """
        Function 
        """

        if self.filter_selection_mode == "nearest":
            filtered_vehicle_information = get_nearest_vehicles(processed_vehicle_information, self.filter_mode, self.filter_k)
        
        elif self.filter_selection_mode == "random":
            filtered_vehicle_information = get_random_vehicles(processed_vehicle_information, self.filter_mode, self.filter_k)
        
        elif self.filter_selection_mode == "furthest":
            filtered_vehicle_information = get_furthest_vehicles(processed_vehicle_information, self.filter_mode, self.filter_k)
        
        else:
            raise ValueError(f"No correct mode was given! Check config {self.filter_selection_mode}")

        return filtered_vehicle_information


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be executed directly")

    

    
        


    