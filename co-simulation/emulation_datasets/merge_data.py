"""
The dataset generation generates dirs with {detection_type}_{town} to properly train the models later, we want to merge several of those with the same detection type and differetn towns into one dataset
"""
import os
import shutil
from typing import List

def merge_datasets(merged_dataset_name: str, detection_type: str, towns: List[str], base_path = "emulation_datasets") -> None:
    """
    Merge several datasets into one dataset.

    Args:
        merged_dataset_name (str): Name of the merged dataset.
        detection_type (str): Type of the detection algorithm.
        towns (List[str]): List of towns to merge.
    """
    merged_dataset_name = f'{detection_type}_{merged_dataset_name}'
    os.makedirs(os.path.join(base_path, merged_dataset_name), exist_ok=True)
    for town in towns:
        dataset_name = f'{detection_type}_{town}'
        for file in os.listdir(os.path.join(base_path, dataset_name)):
            new_file_name = f'{town}_{file}'
            shutil.copy(os.path.join(base_path, dataset_name, file), os.path.join(base_path, merged_dataset_name, new_file_name))
    
    print(f'Merged dataset {merged_dataset_name} created with towns {towns} and detection type {detection_type}')

if __name__ == '__main__':
    merge_datasets('CarlaSumo', 'monocon', ['Town01', 'Town02', 'Town03', 'KIVI'])