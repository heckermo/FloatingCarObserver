import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import libsumo as traci

from utils.sumo_utils import configure_sumo
from detector import detector_factory  # Updated import
from utils.visualize import visualize_fco

def main():
    # Configuration parameters
    fco_id = 'pv_12_7892_1'  # Define the ID of the vehicle that should be the FCO
    sumocfg_file = '24h_sim.sumocfg'  # Define the path to the sumocfg file
    sumo_max_steps = 3600  # Define the maximum number of steps for the SUMO simulation
    save_path = 'visualization'  # Define the path to save the visualization
    detection_mode = '2d-raytracing'  # Detection mode ('3d-raytracing', 'emulation', '2d-raytracing')
    model_path = None  # Path to the trained model for 'nn' mode, if applicable

    # Initialize the detector using the factory function
    detector = detector_factory(mode=detection_mode, model_path=model_path, building_polygons=None)

    # Configure SUMO
    sumo_cmd = configure_sumo(sumocfg_file, show_gui, sumo_max_steps)
    traci.start(sumo_cmd)

    # Delete the save_path if it exists and create a new one
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if fco_id in traci.vehicle.getIDList():
                detected_vehicles = detector.detect_vehicles([fco_id])

                print(f'Detected vehicles: {detected_vehicles[fco_id]}')

                # Visualize the FCO and detected objects
                visualize_fco(
                    all_vehicles=traci.vehicle.getIDList(),
                    detected_vehicles=detected_vehicles[fco_id],
                    fco_ids=[fco_id],
                    fco_position=list(traci.vehicle.getPosition(fco_id)) + [0],
                    show=True,
                    save=False,
                    save_path=save_path
                )
    except traci.exceptions.TraciException as e:
        print(f"An error occurred during simulation: {e}")
    finally:
        traci.close()
        sys.exit("Simulation ended")

if __name__ == "__main__":
    main()

