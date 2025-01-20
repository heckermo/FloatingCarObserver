import pandas as pd
import numpy as np
import os
import pickle
from typing import List
import shutil


"""
This script is used to merge a set of fco datasets into a large dataset for the solving occlusion training.
"""

def merge_fco_dataset(target_datasets: List[str], target_dataset_name):
    # Create the target dataset folder
    os.makedirs(os.path.join('data_fco', target_dataset_name), exist_ok=True)

    # copy the models and config folder from the first dataset to the target dataset
    shutil.copytree(os.path.join('data_fco', target_datasets[0], 'models'), os.path.join('data_fco', target_dataset_name, 'models'), dirs_exist_ok=True)
    shutil.copytree(os.path.join('data_fco', target_datasets[0], 'configs'), os.path.join('data_fco', target_dataset_name, 'configs'), dirs_exist_ok=True)

    # create folder for the complete_bev_images and detected_bev_images
    os.makedirs(os.path.join('data_fco', target_dataset_name, 'complete_bev_images'), exist_ok=True)
    os.makedirs(os.path.join('data_fco', target_dataset_name, 'detected_bev_images'), exist_ok=True)

    # merge all dataset.csv files into one
    dataset = pd.DataFrame()
    for dataset in target_datasets:
        dataset = pd.read_csv(os.path.join('data_fco', dataset, 'dataset.csv'))
        dataset = dataset.reset_index(drop=True)
        dataset = dataset.append(dataset, ignore_index=True)
    dataset.to_csv(os.path.join('data_fco', target_dataset_name, 'dataset.csv'))

    # move all images from complete_bev_images and detected_bev_images to the target dataset
    for dataset in target_datasets:
        for image in os.listdir(os.path.join('data_fco', dataset, 'complete_bev_images')):
            shutil.copy(os.path.join('data_fco', dataset, 'complete_bev_images', image), os.path.join('data_fco', target_dataset_name, 'complete_bev_images', image))
        for image in os.listdir(os.path.join('data_fco', dataset, 'detected_bev_images')):
            shutil.copy(os.path.join('data_fco', dataset, 'detected_bev_images', image), os.path.join('data_fco', target_dataset_name, 'detected_bev_images', image))


if __name__ == '__main__':
    target_datasets = ['i3040_10p_1', 'i3040_10p_2', 'i3040_10p_3', 'i3040_10p_4', 'i3040_10p_5', 'i3040_10p_6', 'i3040_10p_7', 'i3040_10p_8']

    target_dataset_name = 'i3040_10p_large'

    merge_fco_dataset(target_datasets, target_dataset_name)