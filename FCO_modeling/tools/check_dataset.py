"""
This script can be used to check if the list of intersections and their radius intersect with each other
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from configs.config_dataset import DATASET_INTERSECTION

def check_dataset(intersections_dict):
    """
    This function checks if any of the intersections in 'intersections_dict' overlap with each other.
    It prints out any pairs of intersections that overlap.
    """
    overlapping_pairs = []
    num_intersections = len(intersections_dict)
    
    for i in range(num_intersections):
        intersection1 = intersections_dict[i]
        name1 = intersection1['INTERSECTION_NAME']
        center1 = intersection1['INTERSECTION_CENTER']
        radius1 = intersection1['INTERSECTION_RADIUS']
        
        for j in range(i + 1, num_intersections):
            intersection2 = intersections_dict[j]
            name2 = intersection2['INTERSECTION_NAME']
            center2 = intersection2['INTERSECTION_CENTER']
            radius2 = intersection2['INTERSECTION_RADIUS']
            
            # Calculate the distance between centers
            dx = center1[0] - center2[0]
            dy = center1[1] - center2[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            
            # Check if the distance is less than the sum of radii
            if distance < (radius1 + radius2):
                overlapping_pairs.append((name1, name2))
    
    if overlapping_pairs:
        print("Overlapping intersections found:")
        for pair in overlapping_pairs:
            print(f" - {pair[0]} overlaps with {pair[1]}")
    else:
        print("No overlapping intersections found.")
    




if __name__ == "__main__":
    check_dataset(DATASET_INTERSECTION)