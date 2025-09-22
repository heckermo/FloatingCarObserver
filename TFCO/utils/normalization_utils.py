import numpy as np
import csv
import os
from pathlib import Path
import re

base_root = Path(__file__).resolve().parents[3]


def extract_simulation_information(dataset_path):

    """
    Function that extract the relevant information of the simulation to load normalization stats. 
    """

    city_tag = dataset_path[0]
    radius = dataset_path[1]

    city_radius = city_tag + "_" + radius

    return city_tag, radius, city_radius


def denormalize(target_tensor, std, mean):
    """
    Function that denormalize the given target_tensor 
    """
    return target_tensor * std + mean 


def load_normalization_stats(path, dataset_path_conf:str = "", data_name:str = ""):

    """
    Function that return the right mean and standard deviation for the use dataset. 
    """


    if type(path) == list:
        for data_path in path:
            data_name = data_path.split(os.sep)[-1]

            overlap_portion = re.search(r"o([\d.]+)", data_name).group(1)
            storage_location = os.path.join(f"{os.sep}".join(data_path.split(os.sep)[:-1]), "storage_location.csv")

            dataset_path_split = data_name.split("_")

            city_tag, radius, city_radius = extract_simulation_information(dataset_path_split)


    else:

        dataset_path = path.split(os.sep)[-1]

        overlap_portion = re.search(r"o([\d.]+)", dataset_path).group(1)
        storage_location = os.path.join(base_root, dataset_path_conf, "storage_location.csv")

        dataset_path_split = dataset_path.split("_")

        city_tag, radius, city_radius = extract_simulation_information(dataset_path_split)

    if float(overlap_portion) > 0:
        #Overlap-Grid: Stats path are extracted from csv file

        stats_dict = dict()
        
        with open(storage_location, mode="r") as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                dataset_name = row[0].strip()
                path_csv = row[1].strip()
                stats_dict[dataset_name] = path_csv

        stats_path = stats_dict[data_name] 

        with open (os.path.join(base_root, stats_path, "mean.npy"), "rb") as m:
            mean = np.load(m)

        with open (os.path.join(base_root, stats_path, "std.npy"), "rb") as s:
            std = np.load(s)

    else:
        #Single-Grid: Stats path are extracted from data path 
        
        stats_path = os.path.join("data", "stats", city_tag, city_radius)
        
        try: 
            with open (os.path.join(base_root, stats_path, "mean.npy"), "rb") as m:
                mean = np.load(m)

            with open (os.path.join(base_root, stats_path, "std.npy"), "rb") as s:
                std = np.load(s)

        except Exception as e:
            print(f"Error: {e} - Check normalization stats for {stats_path}")
            return np.array([0.0, 0.0]), np.array([1.0, 1.0])
    
    print(f"loaded mean {mean} and std {std} from {stats_path}")
    
    return mean, std 
