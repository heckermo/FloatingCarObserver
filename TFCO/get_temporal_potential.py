import logging
import os
import time
import yaml
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


from configs.config import network_configs, config as cfg
from utils.dataset_utils import TfcoDataset, SequenceTfcoDataset
# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

def get_temporal_potential(sequence_len: Union[int, None], min_timesteps_seen: Union[int, None], dataset_path: List[str], dataset_name, loop: int) -> Tuple[int, int, int]:
    if sequence_len is None:
        sequence_len = cfg['sequence_len']
    if min_timesteps_seen is None:
        min_timesteps_seen = cfg['min_timesteps_seen']

    # Create datasets and data loaders
    train_dataset = SequenceTfcoDataset(
        dataset_path=[os.path.join(dataset_path, name) for name in dataset_name],
        sequence_len=sequence_len,
        max_vehicles=200,
        min_timesteps_seen=min_timesteps_seen,
        loop=loop
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=8)  

    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    # 3: vehicles currently seen in the input
    # 2: vehicles not seen at any timestep
    # 1: vehicles seen at some point in the past but not in the current timestep
    for batch in tqdm(train_loader):
        input_tensor, target_tensor, indexes = batch
        last_input = input_tensor[:, -1, :]

        # Get the number of vehicles
        o = (target_tensor[:, :, 0] == 1).sum(dim=1)
        counter_1 += o.sum().item()

        t = (target_tensor[:, :, 0] == 2).sum(dim=1)
        counter_2 += t.sum().item()

        th = (target_tensor[:, :, 0] == 3).sum(dim=1)
        counter_3 += th.sum().item()

    print(f'Total vehicles: {counter_1 + counter_2 + counter_3}')
    print(f'Vehicles currently seen in the input: {counter_3} which is {counter_3 / (counter_1 + counter_2 + counter_3) * 100:.2f}%')
    print(f'Vehicles not seen at any timestep: {counter_2} which is {counter_2 / (counter_1 + counter_2 + counter_3) * 100:.2f}%')
    print(f'Vehicles seen at some point in the past but not in the current timestep: {counter_1} which is {counter_1 / (counter_1 + counter_2 + counter_3) * 100:.2f}%')

    return counter_1, counter_2, counter_3

def plot_temporal_potential(result_storage, sequence_length, min_timesteps_seen):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for min_timesteps in min_timesteps_seen:
        y_values = []
        for seq_len in sequence_length:
            counters = result_storage.get((seq_len, min_timesteps), (0, 0, 0))
            counter_1, counter_2, counter_3 = counters
            total = counter_1 + counter_2 + counter_3
            if total > 0:
                ratio = counter_1 / total
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

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    result_storage = {}
    
    for loop in config['loops']:
        for seq_len in config['sequence_length']:
            for min_timesteps in config['min_timesteps_seen']:
                print(f'Loop: {loop}, Sequence Length: {seq_len}, Min Timesteps Seen: {min_timesteps}')
                results = get_temporal_potential(
                    sequence_len=seq_len,
                    min_timesteps_seen=min_timesteps,
                    dataset_path=config['dataset_path'],
                    dataset_name=[config['dataset_name']],
                    loop=loop
                )
                result_storage[(seq_len, min_timesteps)] = results
                print('\n\n')
    
    plot_temporal_potential(
        result_storage=result_storage,
        sequence_length=config['sequence_length'],
        min_timesteps_seen=config['min_timesteps_seen']
    )


    
if __name__ == '__main__':
    main('configs/analysis_config.yaml')


