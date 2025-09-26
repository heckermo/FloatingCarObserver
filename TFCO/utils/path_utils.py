import os
from datetime import datetime



def extract_basic_information(information: list):

    """
    Function that extracts the basic information components from a dataset name.

    Returns:
        A tuple containing
            - city (str): Name of the sumo city
            - radius (str): Radius of the considered area
            - overlap (str): Overlap setting
            - penetration_rate (str): Penetration rate of fco
            - point_of_interest_x (str): X coordinate of poi
            - point_of_interest_y (str): Y coordinate of poi

    """

    city = information[0]
    radius = information[1]
    overlap = information[2]
    penetration_rate = information[3]
    point_of_interest_x = information[4]
    point_of_interest_y = information[5]

    return city, radius, overlap, penetration_rate, point_of_interest_x, point_of_interest_y



def generate_file_name(hierarchical_mode, overlap_mode, sequenz_len, min_timesteps, dataset_name):

    """
    Function that generates a standardized file name (for saving models) based on dataset parameters.

    Returns:
        str: Generated file name
    """
    
    assert len(dataset_name) == 1, "More/Less than one dataset was given"

    dataset_name = dataset_name[0]
    information = dataset_name.split("_")

    if overlap_mode:

        print(f"Generate model path with additional overlap informations\n")
        
        city, radius, overlap, penetration_rate, point_of_interest_x, point_of_interest_y = extract_basic_information(information)

        number_grids = information[6]
        mode = information[7]
        number_rows = information[8]

        return f"{city}_{radius}_{overlap}_{penetration_rate}_seq{sequenz_len}_mint{min_timesteps}_{number_grids}_{mode}_{datetime.now().strftime('%d-%m_%H-%M-%S')}"
    
    elif hierarchical_mode:

        print(f"Generate model path with additional hierarchical informations\n")

        city, radius, overlap, penetration_rate, point_of_interest_x, point_of_interest_y = extract_basic_information(information)
        number_grids = information[6]

        return f"{city}_{radius}_{overlap}_{penetration_rate}_hx{point_of_interest_x[3:]}_hy{point_of_interest_y}_{number_grids}_seq{sequenz_len}_mint{min_timesteps}_{number_grids}_{datetime.now().strftime('%d-%m_%H-%M-%S')}"
    
    else:

        print(f"Generate model path without additional overlap informations\n")
        
        city, radius, overlap, penetration_rate, point_of_interest_x, point_of_interest_y = extract_basic_information(information)

        number_rows= information[6]

        return f"{city}_{radius}_{overlap}_{penetration_rate}_seq{sequenz_len}_mint{min_timesteps}_{datetime.now().strftime('%d-%m_%H-%M-%S')}"

    