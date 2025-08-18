import os
from datetime import datetime

def generate_file_name(overlap_mode, sequenz_len, min_timesteps, dataset_name):

    assert len(dataset_name) < 2, "More than one dataset was given"

    dataset_name = dataset_name[0]
    information = dataset_name.split("_")

    if overlap_mode:
        city = information[0]
        radius = information[1]
        overlap = information[2]
        penetration_rate = information[3]
        point_of_interest_x = information[4]
        point_of_interest_y = information[5]
        number_grids = information[6]
        mode = information[7]
        number_rows = information[8]

        return f"{city}_{radius}_{overlap}_{penetration_rate}_seq{sequenz_len}_mint{min_timesteps}_{number_grids}_{mode}_{datetime.now().strftime('%d-%m_%H-%M-%S')}"
    else:
        print(f"Generate path without additional overlap information")
        city = information[0]
        radius = information[1]
        overlap = information[2]
        penetration_rate = information[3]
        point_of_interest_x = information[4]
        point_of_interest_y = information[5]
        number_rows= information[6]

        return f"{city}_{radius}_{overlap}_{penetration_rate}_seq{sequenz_len}_mint{min_timesteps}_{datetime.now().strftime('%d-%m_%H-%M-%S')}"

    