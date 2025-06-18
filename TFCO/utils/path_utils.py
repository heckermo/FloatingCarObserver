import os
from datetime import datetime

def generate_file_name(sequenz_len, min_timesteps, dataset_name):

    assert len(dataset_name) < 2, "More than one dataset was given"

    dataset_name = dataset_name[0]
    information = dataset_name.split("_")

    city = information[0]
    radius = information[1]
    overlap = information[2]
    penetration_rate = information[3]
    point_of_interest = information[4]

    return f"{city}_{radius}_{overlap}_{penetration_rate}_seq{sequenz_len}_mint{min_timesteps}_{datetime.now().strftime('%d-%m_%H-%M-%S')}"