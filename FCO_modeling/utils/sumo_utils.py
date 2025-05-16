import xml.etree.ElementTree as ET
from sumolib import checkBinary
import os
import traci
from typing import List
import numpy as np
from typing import Optional 


def get_all_vehicles(path_to_xml) -> list:
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    vehicle_ids = [vehicle.get('id') for vehicle in root.iter('vehicle')]
    return vehicle_ids

def get_index_after_start(start_time: float, path_to_xml: str) -> int:
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    vehicle_start_times = [float(vehicle.get('depart')) for vehicle in root.iter('vehicle')]
    for i, time in enumerate(vehicle_start_times):
        if time > start_time:
            return i
    raise ValueError('No vehicle found after start time')

def configure_sumo(config_file: str, gui: bool = False, max_steps: int=36000, seed: int = 42) -> List[str]:
    """
    Configure various parameters of SUMO.
    """
    # Setting the cmd mode or the visual mode
    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    # Setting the cmd command to run sumo at simulation time
    sumo_cmd = [
        sumo_binary, "-c", config_file, "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps), "--xml-validation", "never", "--start", "--quit-on-end", "--seed", str(seed)
    ]

    return sumo_cmd

def update_sumocfg(path_to_cfg: str, net_file: str, route_files: str, start_value: int, end_value: Optional[int] = None):
    # Parse the XML file
    tree = ET.parse(path_to_cfg)
    root = tree.getroot()
    if net_file is not None:
        # Update net-file value
        root.find(".//net-file").set("value", net_file)

    # Update route-files value (excluding any None values)
    valid_route_files = [rf for rf in route_files if rf is not None]

    root.find(".//route-files").set("value", "".join(valid_route_files[5:]))

    # Update begin and end values
    root.find(".//begin").set("value", str(start_value))
    if end_value is None:
        end_value = start_value + 10000
    root.find(".//end").set("value", str(end_value))

    # Write the changes back to the file
    tree.write(path_to_cfg)
    
def update_modal_split(path_to_xml: str, modal_split: dict):
    print(modal_split)
    # Load the XML file
    tree = ET.parse(path_to_xml)
    root = tree.getroot()

    # Find the vTypeDistribution element
    vTypeDistribution = root.find('vTypeDistribution')

    # Iterate over the vType elements
    for vType in vTypeDistribution.findall('vType'):
        vTypeId = vType.get('id')
        if vTypeId in modal_split:
            # Update the probability attribute
            vType.set('probability', str(modal_split[vTypeId]))

    # Save the modified XML file
    tree.write(path_to_xml)

def simulate():
    for _ in range(1):
        traci.simulationStep()

def variate_traffic(path_routes: str, seed: int=42, mean=0, std=30):
    """
    This function is used to variate the traffic in the simulation using a normal distribution.
    """
    # set random seed for numpy
    np.random.seed(seed)
    tree = ET.parse(path_routes)
    root = tree.getroot()
    if variate_traffic:
        # Find all trip elements
        trips = root.findall('trip')
        flows = root.findall('flow')

        for trip in trips:
            # Generate a new departure time
            depart_time = np.random.normal(mean, std)

            # Ensure the departure time is non-negative
            depart_time_add = max(depart_time, 0)

            # Get the current departure time
            depart_time = float(trip.get('depart'))

            # Update the departure time
            trip.set('depart', str(depart_time + depart_time_add))

    # Sort routes by their departure(trip) or begin(flow) time

    total_routes = trips + flows
    routes_sorted = sorted(total_routes, key=lambda v: float(v.get('depart')) if v.get('color') == 'cyan' else float(v.get('begin')))

    # Remove old trips from root and add sorted routes
    for route in total_routes:
        root.remove(route)

    for routes in routes_sorted:
        root.append(routes)

    # Save the modified XML to a new file
    tree.write(path_routes)

