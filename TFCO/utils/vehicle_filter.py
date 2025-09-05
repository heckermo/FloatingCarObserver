import numpy as np 
import torch
import random 

#Set fixed seed for reproducibility
random.seed(13)  

def extract_x_y(vehicleposition):

    """
    Extract correct x and y position from the vehicle position.

    Args:
        vehicle_position (torch.Tensor | np.ndarray): Format  [flag, x, y]

    Returns:
        Float: x, y 
    """

    if isinstance(vehicleposition, torch.Tensor):

        if vehicleposition.numel() < 3:
            
            raise ValueError(f"Tensor has unexpected size")
        
        else:
            
            x = float(vehicleposition[1].detach().cpu().item())
            y = float(vehicleposition[2].detach().cpu().item())

        return x, y
    
    elif isinstance(vehicleposition, np.ndarray):

        if vehicleposition.size < 3:
            
            raise ValueError(f"Array has unexpected size")
        
        else:
            
            x = float(vehicleposition[1])
            y = float(vehicleposition[2])

        return x, y 
    
    else:
        raise TypeError(f"Vehicle position has an unsupported type")
    


def zero_padding_vehicles(num_vehicles, k, data_type):
    """
    Create zero-padded vehicle entries to reach k vehicles. 

    Args:
        num_vehicles (int): Current number of vehicles
        k (int): Desired number of vehicles
        data_type (type): torch.Tensor or np.ndarray

    Returns:
        dict: Padded vehicles 
    """ 

    num_zeros = k - num_vehicles
    padded_datapoint = dict()

    if data_type is torch.Tensor:
        padded_data = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    else:
        padded_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for i in range(num_zeros):
        padded_datapoint[f"f_pad_{i}"] = padded_data

    return padded_datapoint


def get_data_type(data):

    """
    Function that check the data types

    Args:
        data: First element of the dictionary

    Returns:
        type: data type 
    """
    
    if isinstance(data, torch.Tensor):
        return torch.Tensor
    
    elif isinstance(data, np.ndarray):
        return np.ndarray
    
    else:
        raise TypeError(f"Vehicle position has an unsupported type")



def get_nearest_vehicles(vehicle_information, mode, k):

    """
    Function that select k nearest vehicles to the origin (0,0).

    Args:
        vehicle_information (dict): {vehicle_id: position}
        mode (str): "exact" (pad to k), "below" (allow < k)
        k (int): Number of vehicles to select

    Returns:
        dict: Filtered (and optionally padded) vehicles
    """

    datapoint_len = len(vehicle_information)

    if mode == "exact" and datapoint_len < k:    

        if vehicle_information:
            data_type = get_data_type(next(iter(vehicle_information.values())))
        else:
            data_type = torch.Tensor
        
        padded_vehicles = zero_padding_vehicles(len(vehicle_information), k, data_type)
        
        return {**vehicle_information, **padded_vehicles} 

    elif mode == "below" and datapoint_len < k:
        
        return vehicle_information
    
    distances = list()
    filtered_datapoint = dict()
    
    for vehicle_id, vehicle_position in vehicle_information.items():
        
        x,y = extract_x_y(vehicle_position)

        distance = np.hypot(x, y)
        distances.append((vehicle_id, distance))

    distances.sort(key=lambda x: x[1])
    distances = distances[:k]

    for vehicle in distances:
        key = vehicle[0]
        value = vehicle_information.get(key)
        filtered_datapoint[key] = value


    return filtered_datapoint



def get_furthest_vehicles(vehicle_information, mode, k):

    """
    Function that select k furthest vehicles to the origin (0,0).

    Args:
        vehicle_information (dict): {vehicle_id: position}
        mode (str): "exact" (pad to k), "below" (allow < k)
        k (int): Number of vehicles to select

    Returns:
        dict: Filtered (and optionally padded) vehicles
    """

    datapoint_len = len(vehicle_information)

    if mode == "exact" and datapoint_len < k:
        
        if vehicle_information:
            data_type = get_data_type(next(iter(vehicle_information.values())))
        else:
            data_type = torch.Tensor
        
        padded_vehicles = zero_padding_vehicles(len(vehicle_information), k, data_type)
        
        return {**vehicle_information, **padded_vehicles} 

    elif mode == "below" and datapoint_len < k:
        return vehicle_information

    distances = list()
    filtered_datapoint = dict()
    
    for vehicle_id, vehicle_position in vehicle_information.items():
        
        x,y = extract_x_y(vehicle_position)

        distance = np.hypot(x, y)
        distances.append((vehicle_id, distance))
        

    distances.sort(key=lambda x: x[1])
    distances = distances[-k:]

    for vehicle in distances:
        key = vehicle[0]
        value = vehicle_information.get(key)
        
        filtered_datapoint[key] = value

    return filtered_datapoint



def get_random_vehicles(vehicle_information, mode, k):

    """
    Funtion that select k random vehicles

    Args:
        vehicle_information (dict): {vehicle_id: position}
        mode (str): "exact" (pad to k), "below" (allow < k)
        k (int): Number of vehicles to select

    Returns:
        dict: Filtered (and optionally padded) vehicles
    """

    datapoint_len = len(vehicle_information)
 
    if mode == "exact" and datapoint_len < k:
        
        if vehicle_information:
            data_type = get_data_type(next(iter(vehicle_information.values())))
        else:
            data_type = torch.Tensor
        
        padded_vehicles = zero_padding_vehicles(len(vehicle_information), k, data_type)
        
        return {**vehicle_information, **padded_vehicles} 

    elif mode == "below" and datapoint_len < k:
        return vehicle_information
        
    filtered_datapoint = dict()
    vehicles = list(vehicle_information.keys())
    random_vehicles_id = random.sample(vehicles, k)

    for key in random_vehicles_id:
        value = vehicle_information.get(key)
        filtered_datapoint[key] = value
        
    return filtered_datapoint
    

if __name__ == "__main__":
    
    #test = {"test_1": torch.tensor([1, 0.9991, 0.032]),
    #       "test_2": torch.tensor([1, 0.673, 0.2137]),
    #        "test_3": torch.tensor([1, -0.40134044891455906, 0]),
    #        "test_4": torch.tensor([1, -1, 1])}
    
    #test_results = (get_nearest_vehicles(test, "exact", 15))
    #print(test_results)
    #print(f"Type {type(next(iter(test_results.values())))}")
    
    raise NotImplementedError("This script is not meant to be executed directly")