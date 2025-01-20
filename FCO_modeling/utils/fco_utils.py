import random
import libsumo as traci


class FcoMonitor():
    def __init__(self, penetration_rate: float):
        self.penetration_rate = penetration_rate
        self.fco_vehicles = []
        self.non_fco_vehicles = []
    
    def update(self):
        all_vehicles = traci.vehicle.getIDList()
        
        # Delete the vehilces that are not in the simulation anymore
        self.fco_vehicles = [v for v in self.fco_vehicles if v in all_vehicles]
        self.non_fco_vehicles = [v for v in self.non_fco_vehicles if v in all_vehicles]

        # Randomly add vehilces to the fco_vehicles and non_fco_vehicles according to the penetration rate
        new_vehicles = [v for v in all_vehicles if v not in self.fco_vehicles and v not in self.non_fco_vehicles]
        for v in new_vehicles:
            if random.random() < self.penetration_rate:
                self.fco_vehicles.append(v)
            else:
                self.non_fco_vehicles.append(v)

