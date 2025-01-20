import logging
from typing import Dict, Tuple, Optional, List
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate, translate
import libsumo as traci
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def create_vehicle_polygon(
    vehicle_id: str, 
    angle: Optional[float] = None, 
    x: Optional[float] = None, 
    y: Optional[float] = None, 
    length: Optional[float] = None, 
    width: Optional[float] = None, 
    representation: str = 'box'
) -> Polygon:
    """
    Creates a Shapely Polygon representing a vehicle.

    Args:
        vehicle_id (str): The ID of the vehicle.
        angle (Optional[float]): The angle of the vehicle in degrees. If None, it is fetched from the simulation.
        x (Optional[float]): The x-coordinate of the vehicle's position. If None, it is fetched from the simulation.
        y (Optional[float]): The y-coordinate of the vehicle's position. If None, it is fetched from the simulation.
        length (Optional[float]): The length of the vehicle. If None, it is fetched from the simulation.
        width (Optional[float]): The width of the vehicle. If None, it is fetched from the simulation.
        representation (str): The representation of the vehicle. Default is 'box'.

    Returns:
        Polygon: A Shapely Polygon representing the vehicle.
    """
    if angle is None:
        angle = traci.vehicle.getAngle(vehicle_id)
    if x is None or y is None:
        x, y = traci.vehicle.getPosition(vehicle_id)
    if length is None:
        length = traci.vehicle.getLength(vehicle_id)
    if width is None:
        width = traci.vehicle.getWidth(vehicle_id)

    adjusted_angle = (-angle) % 360

    if representation == 'box':
        rect = Polygon([
            (-width / 2, -length / 2), 
            (-width / 2, length / 2), 
            (width / 2, length / 2), 
            (width / 2, -length / 2)
        ])
    else:
        raise ValueError(f"Invalid vehicle representation: {representation}")

    rotated_rect = rotate(rect, adjusted_angle, use_radians=False, origin=(0, 0))
    translated_rect = translate(rotated_rect, xoff=x, yoff=y)
    
    return translated_rect


def detect_intersections(ray: tuple, objects: Dict[str,Polygon]) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    """
    Adapting the function such that the object that got hit by the ray is returned alongside the intersection point.
    Intersection points should only be considered if it is the closest intersection point to the ray's origin.
    """
    closest_hit_coordinate = None
    closest_hit_object = None
    min_distance = float('inf')
    ray_line = LineString(ray)  # Create a LineString object from the ray

    for key, obj in objects.items():
        if not ray_line.intersects(obj):
            continue

        intersection_point = ray_line.intersection(obj)
        if intersection_point.is_empty:
            continue

        if intersection_point.geom_type.startswith('Multi'):
            coords = [coord for part in intersection_point.geoms if hasattr(part, 'coords') for coord in part.coords]
        else:
            coords = intersection_point.coords if hasattr(intersection_point, 'coords') else []

        for coord in coords:
            distance = Point(ray[0]).distance(Point(coord))
            if distance < min_distance:
                min_distance = distance
                closest_hit_coordinate = coord  # Update the closest intersection point
                closest_hit_object = key  # Update the object that got hit

    return closest_hit_object, closest_hit_coordinate

def generate_rays(center: Tuple[float, float], num_rays: int=360, radius: float=50) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Generates a list of rays originating from a central point.
    Each ray is represented as a tuple containing the start point (center) and the end point, 
    which is calculated based on the specified radius and angle.
    Args:
        center (Tuple[float, float]): The central point from which rays are generated.
        num_rays (int, optional): The number of rays to generate. Default is 360.
        radius (float, optional): The length of each ray. Default is 50.
    Returns:
        List[Tuple[Tuple[float, float], Tuple[float, float]]]: A list of tuples, each containing 
        the start and end points of a ray.
    """
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    rays = [(center, (float(center[0] + np.cos(angle) * radius), float(center[1] + np.sin(angle) * radius))) for angle in angles]
    return rays

def save_polygons_and_rays_to_file(polygons: List[Polygon], rays: Optional[List[Tuple]]=None, filename='polygons_rays_plot.png', dpi=300):
    """
    Plots a list of Polygon objects and Rays (lines) and saves the plot to a file with a specified resolution.

    Parameters:
    polygons (list): List of shapely.geometry.Polygon objects.
    rays (list): List of tuples, where each tuple contains two points (start and target coordinates) representing a ray.
    filename (str): The name of the file to save the plot (default is 'polygons_rays_plot.png').
    dpi (int): The resolution in dots per inch (default is 300).

    Returns:
    None
    """
    # Create a plot
    fig, ax = plt.subplots()

    # Plot each polygon
    for poly in polygons:
        x, y = poly.exterior.xy  # Get the x and y coordinates of the polygon's exterior
        ax.fill(x, y, alpha=0.5, fc='black', edgecolor='black', linewidth=0.1)

    # Plot each ray
    if rays is not None:
        for ray in rays:
            start, target = ray
            start_x, start_y = start
            target_x, target_y = target
            ax.plot([start_x, target_x], [start_y, target_y], color='red', linewidth=0.1, linestyle='--')  # Plot ray as a red dashed line

    # Set equal scaling for both axes
    ax.set_aspect('equal', 'box')

    # Save the plot to file with specified dpi (resolution)
    plt.savefig(filename, dpi=dpi)

    # Close the plot to free memory
    plt.close()


def parse_polygons_from_xml(file_path: str) -> List[Polygon]:
    """
    Parse an XML file containing 'poly' elements and convert them into Shapely Polygons.
    
    Parameters:
        file_path (str): The path to the XML file.
        
    Returns:
        List[Polygon]: A list of Shapely Polygon objects.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize an empty list to store the polygons
    polygons = []

    # Loop through each 'poly' element in the XML
    for poly in root.findall('poly'):
        shape_str = poly.get('shape')
        
        # Convert the shape string into a list of tuples (coordinates)
        coordinates = [tuple(map(float, point.split(','))) for point in shape_str.split()]
        if len(coordinates) < 3:
            continue
        
        # Create a Shapely Polygon and add it to the list
        polygon = Polygon(coordinates)
        polygons.append(polygon)

    return polygons