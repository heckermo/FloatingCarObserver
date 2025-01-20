import traci
import sys
import os
sys.path.append(os.path.join(os.path.expanduser('~'), 'sumo_detector'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'sumo_detector', 'SUMO_detector_plus'))
from SUMO_detector_plus.detector import Detector
from SUMO_detector_plus.utils_detector.create_box import create_nn_input
import numpy as np
import math

# TODO import configs
class EmulationDetector():
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.current_time = None
        self.detector = Detector(mode = 'nn', model_path = weights_path)
    
    def run_emulation_inference(self, fco, current_time):
        if self.current_time != current_time:
            self.vehicle_dict = create_vehicle_dict()
            self.current_time = current_time
        try: 
            all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians = self.detector.detect(fco, vehicle_dict=self.vehicle_dict)
        except:
            all_vehicles, all_cyclists, all_pedestrians, detected_vehicles, detected_cyclists, detected_pedestrians = [], [], [], [], [], []
        results_dict = {
            'all_vehicles': all_vehicles,
            'all_cyclists': all_cyclists,
            'all_pedestrians': all_pedestrians,
            'detected': detected_vehicles,
            'detected_cyclists': detected_cyclists,
            'detected_pedestrians': detected_pedestrians
        }
        return results_dict
    
def create_vehicle_dict() -> dict:
    """
    Create a dictionary of all vehicles in the simulation with their position and angle
    using the sumo subscriptions
    """
    vehicle_dict = dict()
    vehicles = traci.vehicle.getIDList()

    for vehicle in vehicles:
        traci.vehicle.subscribe(vehicle, [traci.constants.VAR_POSITION, traci.constants.VAR_ANGLE, traci.constants.VAR_TYPE, traci.constants.VAR_WIDTH, traci.constants.VAR_LENGTH])
    subscription_results = traci.vehicle.getAllSubscriptionResults()
    for vehicle, data in subscription_results.items():
        vehicle_dict[vehicle] = {
            'pos_x': data[traci.constants.VAR_POSITION][0],
            'pos_y': data[traci.constants.VAR_POSITION][1],
            'angle': data[traci.constants.VAR_ANGLE],
            'type': data[traci.constants.VAR_TYPE], 
            'width': data[traci.constants.VAR_WIDTH],
            'length': data[traci.constants.VAR_LENGTH]
        }
    # print(f'subscription took {time.time() - t}')
    return vehicle_dict


def create_dataset_entry(fco: str, detection_results, simtime, radius = 50): 
    assert 'detected' in detection_results
    img_base_path = os.path.join('emulation_dataset', 'images')
    if not os.path.exists(img_base_path):
        os.makedirs(img_base_path)
    current_dict = {} 
    current_vehicle_dict = create_vehicle_dict()
    ego_pos_x = current_vehicle_dict[fco]['pos_x']
    ego_pos_y = current_vehicle_dict[fco]['pos_y']
    ego_angle = current_vehicle_dict[fco]['angle']
    ego_pos = [ego_pos_x, ego_pos_y, ego_angle]
    img_name = f'ego_{fco}_time_{simtime}'
    _, _, filename = create_nn_input(fco, current_vehicle_dict.keys(), [], ego_pos, 50, [], path = os.path.join(img_base_path, f'{img_name}.npy'), image_size = 400, vehicle_dict = current_vehicle_dict)
    radius_vehicles = []
    for vehicle in current_vehicle_dict:
        pos_x = current_vehicle_dict[vehicle]['pos_x']
        pos_y = current_vehicle_dict[vehicle]['pos_y']
        calculated_distance = np.sqrt((ego_pos_x - pos_x)**2 + (ego_pos_y - pos_y)**2)
        if calculated_distance < radius:
            if vehicle != fco:
                radius_vehicles.append(vehicle)
    for vehicle in radius_vehicles:
        entry_name = f'ego_{fco}_time_{simtime}_vehicle_{vehicle}'
        current_dict[entry_name] = {}
        current_dict[entry_name]['image'] = filename
        current_dict[entry_name]['detected'] = True if vehicle in detection_results['detected'] else False
        base_vector = [current_vehicle_dict[vehicle]['pos_x'] - ego_pos_x, current_vehicle_dict[vehicle]['pos_y'] - ego_pos_y]
        theta = math.radians(ego_angle)
        rotated_vector = [base_vector[0] * math.cos(theta) - base_vector[1] * math.sin(theta), base_vector[0] * math.sin(theta) + base_vector[1] * math.cos(theta)]
        current_dict[entry_name]['vector'] = rotated_vector
    return current_dict

if __name__ == '__main__':
    emulation_detector = EmulationDetector(weights_path='path/to/weights')