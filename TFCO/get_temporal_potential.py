import logging
import os
import time
import yaml
import sys
from datetime import datetime
from typing import Dict, Tuple, Union, List
sys.path.append('../TFCO')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.profiler as profiler
import time

import matplotlib.pyplot as plt
import numpy as np


from utils.dataset_utils import SequenceTfcoDataset
# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True



def extract_filter_parameters(config):

    """
    Extracts filtering parameters for vehicle selection from the config file.
    """
    filter_parameters = list()
    
    if config["filter"]:
        
        try:
            filter_mode = config["vehicle_filter_mode"]
            selection_mode = config["vehicle_selection_mode"]
            k = config["num_vehicles"]

            filter_parameters = [filter_mode, selection_mode, k]
        
        except KeyError as e:
            raise KeyError(f"Check Config, not all parameters for filtering vehicles are specified {e}\n")
    else:
        filter_parameters = None

    if filter_parameters is not None: print(f"Following filtering is applied: {filter_parameters}")

    return filter_parameters



def get_temporal_potential(sequence_len: Union[int, None], min_timesteps_seen: Union[int, None], dataset_path: List[str], dataset_name, loop: int, max_vehicles: int,
                           normalization: str, radius: int, filter_parameters: List) -> Tuple[int, int, int]:
    
    """
    Computes temporal potential and visibility statistics of vehicles in the dataset.

    Categories:
    - currently_visible: vehicles visible in the last timestep
    - past_visible: vehicles visible in past timesteps, but not currently
    - never_visible: vehicles never visible in the sequence


    Returns:
    Dictionary with counts, ratios and visibility metrics.
    """

    # Create datasets and data loaders
    dataset = SequenceTfcoDataset(
        dataset_path=[os.path.join(dataset_path, name) for name in dataset_name],
        sequence_len=sequence_len,
        max_vehicles=max_vehicles,
        min_timesteps_seen=min_timesteps_seen,
        radius=radius,
        normalization=normalization,
        loop=loop,
        filter=filter_parameters
    )

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

    if len(data_loader) == 0:
        raise ValueError(f'No data found for the given configuration: sequence_len={sequence_len}, min_timesteps_seen={min_timesteps_seen}, loop={loop}') 

    past_visible_count = 0
    never_visible_count = 0
    currently_visible_count = 0


    total_timesteps_seen = 0
    timesteps_visibility_sum = 0
    vehicle_visibility_ratios = list()
    all_timesteps_length = list()
    
    for batch in tqdm(data_loader):
        input_tensor, target_tensor, indexes = batch

        timesteps_seen_per_vehicle = torch.sum(input_tensor[:, :, :, 0] == 1, dim=1)
        total_timesteps_seen += timesteps_seen_per_vehicle.sum().item()

        vehicle_visibility_ratios.extend((timesteps_seen_per_vehicle.cpu().numpy() / sequence_len).flatten())
        
        timesteps_visibility = get_timesteps_visibility(input_tensor)
        all_timesteps_length.extend(timesteps_visibility)

        # Get the number of vehicles
        o = (target_tensor[:, :, 0] == 1).sum(dim=1)
        past_visible_count += o.sum().item()

        t = (target_tensor[:, :, 0] == 2).sum(dim=1)
        never_visible_count += t.sum().item()

        th = (target_tensor[:, :, 0] == 3).sum(dim=1)
        currently_visible_count += th.sum().item()
    
    total_vehicles = past_visible_count + never_visible_count + currently_visible_count
    mean_visibility = total_timesteps_seen / (total_vehicles * sequence_len)
    never_visibilty_ratio = never_visible_count / total_vehicles
    mean_timesteps_visibility = float(np.mean(all_timesteps_length)) if all_timesteps_length else 0.0
    std_timesteps_visibility = float(np.std(all_timesteps_length)) if all_timesteps_length else 0.0


    print(f"Shape of input tensor {input_tensor.shape} and target tensor {target_tensor.shape}")

    metrics = {
            "past_visible": past_visible_count,
            "never_visible": never_visible_count,
            "currently_visible": currently_visible_count,
            "total_vehicles": total_vehicles,

            "mean_visibility": mean_visibility,
            "std_visibilty": float(np.std(vehicle_visibility_ratios)),
            "median_visibility": float(np.median(vehicle_visibility_ratios)),
            "fully_visible_ratio": float(np.mean(np.array(vehicle_visibility_ratios) == 1.0)),
            "never_visible_ratio": never_visibilty_ratio,

            "mean_timesteps_visibility": mean_timesteps_visibility,
            "std_timesteps_visibility": std_timesteps_visibility,
            }
    
    plot_necessary = {
            "vehicle_visibility_ratios": vehicle_visibility_ratios,
            "timesteps_lengths": all_timesteps_length 
    }
    
    print(f"Metrics: {metrics}")

    return metrics, plot_necessary


def get_timesteps_visibility(input_tensor: torch.Tensor):
    
    """
    Computes the mean length of timesteps visibility for all vehicles.

    Args:
        input_tensor: torch.Tensor

    Returns:
        timesteps length across all vehicles in the batch
    """

    visibility = (input_tensor[:, :, :, 0] == 1).cpu().numpy() 
    timestep_streaks = []

    for batch_vis in visibility: 
        for veh_vis in batch_vis.T: 
            diffs = np.diff(np.concatenate(([0], veh_vis.astype(int), [0])))
            start = np.where(diffs == 1)[0]
            end = np.where(diffs == -1)[0]
            streak_lengths = end - start
            timestep_streaks.extend(streak_lengths.tolist())

    return timestep_streaks

        

def plot_temporal_potential(result_storage, sequence_length, min_timesteps_seen):
    
    """
    Plots the ratio of vehicles over sequence length.
    """

    plt.figure(figsize=(10, 6))
    for min_timesteps in min_timesteps_seen:
        y_values = []
        for seq_len in sequence_length:
            metrics = result_storage.get((seq_len, min_timesteps), (0, 0, 0))
            if metrics:
                ratio = metrics["past_visible"] / metrics["total_vehicles"]
            else:
                ratio = 0
            y_values.append(ratio)
        plt.plot(sequence_length, y_values, marker='o', label=f'min_timesteps_seen={min_timesteps}')

    plt.xlabel('Sequence Length')
    plt.ylabel('Ratio of Vehicles Seen in Past but Not in Current Timestep')
    plt.title('Temporal Potential Analysis')
    plt.legend()
    plt.savefig('temporal_potential.png', bbox_inches='tight')
    plt.close()



def plot_distributions(metrics: Dict, prefix: str):
    """
    Visualization of visibility ratios and timesteps lengths.
    """
    
    if len(metrics["vehicle_visibility_ratios"]) > 0:

        plt.figure(figsize=(8, 5))
        plt.hist(metrics["vehicle_visibility_ratios"], bins=20, color="blue", alpha=0.7)
        plt.xlabel("Visibility Ratio per Vehicle")
        plt.ylabel("Count")
        plt.title("Distribution of Vehicle Visibility Ratios")
        plt.savefig(f"{prefix}_visibility_ratios.png", bbox_inches="tight")
        plt.close()


        plt.figure(figsize=(6, 5))
        plt.boxplot(metrics["vehicle_visibility_ratios"], vert=True, patch_artist=True)
        plt.ylabel("Visibility Ratio")
        plt.title("Boxplot of Vehicle Visibility Ratios")
        plt.savefig(f"{prefix}_visibility_ratios_boxplot.png", bbox_inches="tight")
        plt.close()


    if len(metrics["timesteps_lengths"]) > 0:

        plt.figure(figsize=(8, 5))
        plt.hist(metrics["timesteps_lengths"], bins=20, color="green", alpha=0.7)
        plt.xlabel("Timesteps Lengths")
        plt.ylabel("Count")
        plt.title("Distribution of Visibility Timesteps Lengths")
        plt.savefig(f"{prefix}_timesteps_lengths.png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.boxplot(metrics["timesteps_lengths"], vert=True, patch_artist=True)
        plt.ylabel("Streak Lengths")
        plt.title("Boxplot of Visibility Timesteps Lengths")
        plt.savefig(f"{prefix}_timesteps_lengths_boxplot.png", bbox_inches="tight")
        plt.close()



def main(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    filter_parameters = extract_filter_parameters(config)
    
    result_storage = {}
    
    for loop in config['loops']:
        for seq_len in config['sequence_length']:
            for min_timesteps in config['min_timesteps_seen']:
                print(f'Loop: {loop}, Sequence Length: {seq_len}, Min Timesteps Seen: {min_timesteps}')
                results, plot_necessary = get_temporal_potential(
                    sequence_len=seq_len,
                    min_timesteps_seen=min_timesteps,
                    dataset_path=config['dataset_path'],
                    dataset_name=[config['dataset_name']],
                    max_vehicles=config["max_vehicles"],
                    radius=config["radius"],
                    normalization=config["normalization"],
                    filter_parameters=filter_parameters,
                    loop=loop
                )
                result_storage[(seq_len, min_timesteps)] = results


                prefix = f'r{config["radius"]}_loop{loop}_seq{seq_len}_min{min_timesteps}_'
                plot_distributions(plot_necessary, prefix=prefix)
                print('\n\n')
    

    plot_temporal_potential(
        result_storage=result_storage,
        sequence_length=config['sequence_length'],
        min_timesteps_seen=config['min_timesteps_seen']
    )


    

if __name__ == '__main__':
    main('configs/analysis_config.yaml')


