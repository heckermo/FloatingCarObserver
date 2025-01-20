import numpy as np
import traci
from typing import List, Tuple

def get_current_fcos(fco_penetration):
    all_vehicles = traci.vehicle.getIDList()
    fcos = list()
    for v in all_vehicles:
        if traci.vehicle.getVehicleClass(v) == 'passenger':
            if np.random.rand() < fco_penetration:
                fcos.append(v)
    return fcos

def update_fcos(fcos, fco_penetration):
    # delete the fcos that are not in the simulation anymore
    current_vehicles = traci.vehicle.getIDList()
    for v in fcos:
        if v not in current_vehicles:
            fcos.remove(v)
    # add new fcos
    all_vehicles = traci.vehicle.getIDList()
    for v in all_vehicles:
        if traci.vehicle.getVehicleClass(v) == 'passenger':
            if v not in fcos and np.random.rand() < fco_penetration:
                fcos.append(v)
    return fcos


def filter_intersection_fcos(fcos: List[str], intersection_center: Tuple[float, float], intersection_radius: float):
    current_fcos = list()
    for v in fcos:
        try:
            v_pos = traci.vehicle.getPosition(v)
        except:
            continue
        if np.linalg.norm(np.array(v_pos) - np.array(intersection_center)) < intersection_radius:
            current_fcos.append(v)
    return current_fcos