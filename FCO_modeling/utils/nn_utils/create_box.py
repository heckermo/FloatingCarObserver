import os
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PIL import Image, ImageDraw
from typing import List, Tuple
import torch
from torchvision import transforms
import traci

from configs.config_simulation import TRAFFIC
from sumo_sim.extract_poly_cords import filter_buildings

def create_box_pillow(width, length, x_pos, y_pos, angle, ego_info):
    ego_x, ego_y, ego_angle = ego_info

    # Define vertices of the box
    vertices = np.array([
        [-width / 2, 0],
        [width / 2, 0],
        [width / 2, -length],
        [-width / 2, -length]
    ])

    # Calculate angle in radians and adjust with ego's angle
    angle_rad = np.radians(-angle)

    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Rotate vertices using rotation matrix
    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    # Translate rotated vertices by relative position of the box to the ego vehicle
    global_vertices = rotated_vertices + np.array([x_pos, y_pos])

    # Convert global vertices to ego coordinate system
    theta = np.radians(ego_angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # rotation matrix

    t = np.array([ego_x, ego_y])  # translation vector

    # Apply transformation
    transformed_vertices = np.dot(global_vertices[:, :2] - t, R.T)

    # Convert numpy array to list of tuples for Pillow
    polygon_vertices = [(x, y) for x, y in transformed_vertices]

    return polygon_vertices


def create_box(width, length, x_pos, y_pos, angle, ego_info=None):
    ego_x, ego_y, ego_angle = ego_info

    # Calculate center_point
    center_point = np.array([0, length / 2])

    # Define vertices of the box
    vertices = np.array([
        [-width / 2, 0],
        [width / 2, 0],
        [width / 2, -length],
        [-width / 2, -length]
    ])

    # Calculate angle in radians and adjust with ego's angle
    angle_rad = np.radians(-angle)

    # Create rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Rotate vertices using rotation matrix
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    rotated_center_point = np.dot(center_point, rotation_matrix.T)

    # Translate rotated vertices by relative position of the box to the ego vehicle
    global_vertices = rotated_vertices + np.array([x_pos, y_pos])
    #print(f'global_vertices: {global_vertices}')

    # Convert global vertices to ego coordinate system
    theta = np.radians(ego_angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # rotation matrix

    t = np.array([ego_x, ego_y])  # translation vector

    # Apply transformation
    global_vertices[:, :2] = np.dot(global_vertices[:, :2] - t, R.T)

    # Log the final position of the vertices
    #print(f'global_vertices: {global_vertices}')

    # Create a polygon using the final vertices
    polygon = patches.Polygon(global_vertices, closed=True, fill=True, edgecolor='black', facecolor='black')

    return polygon

def create_building_pillow(global_vertices, ego_info):
    ego_x, ego_y, ego_angle = ego_info

    # Convert global vertices to ego coordinate system
    theta = np.radians(ego_angle)

    # Create the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Define the translation vector
    t = np.array([ego_x, ego_y])

    # Convert global_vertices to numpy array for operations
    global_vertices = np.array(global_vertices)

    # Apply the transformation
    transformed_vertices = np.dot(global_vertices[:, :2] - t, R.T)

    # Convert numpy array to list of tuples for Pillow
    polygon_vertices = [(x, y) for x, y in transformed_vertices]

    return polygon_vertices


def create_building(global_vertices, ego_info):
    ego_x, ego_y, ego_angle = ego_info

    # Convert global vertices to ego coordinate system
    theta = np.radians(ego_angle)

    # Create the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Define the translation vector
    t = np.array([ego_x, ego_y])

    # Convert global_vertices to numpy array for operations
    global_vertices = np.array(global_vertices)

    # Apply the transformation
    global_vertices[:, :2] = np.dot(global_vertices[:, :2] - t, R.T)

    # Create a polygon using the final vertices
    polygon = patches.Polygon(global_vertices, closed=True, fill=True, edgecolor='black', facecolor='black')

    return polygon

def get_object_dimensions(vehicle_id):
    vehicle_type = traci.vehicle.getVehicleClass(vehicle_id)
    if vehicle_type == 'passenger':
        if vehicle_type == 'passenger/van':
            width = TRAFFIC['LARGE_CAR']['WIDTH']
            length = TRAFFIC['LARGE_CAR']['LENGTH']
        else:
            width = TRAFFIC['SMALL_CAR']['WIDTH']
            length = TRAFFIC['SMALL_CAR']['LENGTH']
    elif vehicle_type == 'truck':
        width = TRAFFIC['TRUCK']['WIDTH']
        length = TRAFFIC['TRUCK']['LENGTH']
    elif vehicle_type == 'bus':
        width = TRAFFIC['BUS']['WIDTH']
        length = TRAFFIC['BUS']['LENGTH']
    elif vehicle_type == 'bicycle':
        width = TRAFFIC['BIKE']['WIDTH']
        length = TRAFFIC['BIKE']['LENGTH']
    elif vehicle_type == 'delivery':
        width = TRAFFIC['DELIVERY']['WIDTH']
        length = TRAFFIC['DELIVERY']['LENGTH']
    else:
        raise ValueError(f'Vehicle type {vehicle_type} not recognized')

    return width, length, None

def plot_vehicles_buildings(vehicles, pedestrians, buildings, ego_info, radius, path=None, image_size=None):
    if image_size is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=(image_size/100, image_size/100), dpi=100)
    for box in vehicles:
        #get the vehicle type with traci to create the box with correct dimensions
        width, length, _ = get_object_dimensions(box)

        # Create a polygon
        polygon = create_box(width, length,
                             vehicles[box][0], vehicles[box][1],
                             vehicles[box][2], ego_info)
        # Add the polygon to the axis
        ax.add_patch(polygon)

    for box in pedestrians:
        # Create a polygon
        polygon = create_box(TRAFFIC['PERSON']['WIDTH'], TRAFFIC['PERSON']['LENGTH'],
                             pedestrians[box][0], pedestrians[box][1],
                             pedestrians[box][2], ego_info)
        # Add the polygon to the axis
        ax.add_patch(polygon)

    for building in buildings:
        polygon = create_building(buildings[building], ego_info)
        ax.add_patch(polygon)

        # Set the limits of the plot based on the coordinates of the points
        # Adjust the values according to your needs
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Hide the axis
    ax.axis('off')
    # Show the plot
    # create tmp folder if it does not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if path is None:
        filename = f'image_{uuid.uuid4().hex}.jpg'
        plt.savefig(os.path.join('tmp', filename), bbox_inches='tight', pad_inches=0)
    else:
        filename = path
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return filename

def plot_vehicles_buildings_pillow(vehicles, pedestrians, buildings, ego_info, radius):
    # Create an image with white background
    size = (radius * 2, radius * 2)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)

    # Draw vehicles
    for box in vehicles:
        width, length, _ = get_object_dimensions(box)
        polygon = create_box_pillow(width, length, vehicles[box][0], vehicles[box][1], vehicles[box][2], ego_info)
        draw.polygon(polygon, fill="black")

    # Draw pedestrians
    for box in pedestrians:
        polygon = create_box_pillow(TRAFFIC['PERSON']['WIDTH'], TRAFFIC['PERSON']['LENGTH'], pedestrians[box][0], pedestrians[box][1], pedestrians[box][2], ego_info)
        draw.polygon(polygon, fill="black")  # Change color as needed

    # Draw buildings
    for building in buildings:
        polygon = create_building_pillow(buildings[building], ego_info)
        draw.polygon(polygon, fill="black")  # Change color as needed

    # Save the image
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    filename = f'image_{uuid.uuid4().hex}.jpg'
    image.save(os.path.join('tmp', filename))

    return filename

def get_empty_plot():
    fig, ax = plt.subplots()
    ax.set_axis_off()  # Hide the axes
    fig.set_facecolor('white')  # Set background color to white
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding and margins
    return fig

def plot_boxes(vehicles: dict, radius: int, ego_info: List[float], vehicle_size: List[float], target_size: int = 400, img: bool = False, figsize: int=100) -> torch.Tensor:
    # Create a figure and axis
    if img:
        fig, ax = plt.subplots(figsize=(2.56, 2.56))
    else:
        fig, ax = plt.subplots()

    for box in vehicles:
        # Create a polygon
        vehicles[box]['width'] = vehicle_size[0]
        vehicles[box]['length'] = vehicle_size[1]
        polygon = create_box(vehicles[box]['width'], vehicles[box]['length'],
                             vehicles[box]['pos_x'], vehicles[box]['pos_y'],
                             vehicles[box]['angle'], ego_info)
        # Add the polygon to the axis
        ax.add_patch(polygon)

    # Set the limits of the plot based on the coordinates of the points
    # Adjust the values according to your needs
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    # Set aspect ratio to be equal
    ax.set_aspect('equal')

    # Hide the axis
    ax.axis('off')

    if img:
        return fig

    # Show the plot
    # plt.show()
    plt.savefig('tmp/tmp.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Load the saved image
    image = Image.open('tmp/tmp.png')

    # Convert the image to grayscale (single channel)
    gray_image = image.convert('L')

    # Resize the image to 400x400
    resized_image = gray_image.resize((target_size, target_size))

    # Convert the image to a numpy array
    image_array = np.array(resized_image)

    # Convert the numpy array to a Torch tensor
    tensor_image = torch.from_numpy(image_array)

    # Add an additional dimension to represent the single channel
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def create_nn_input(ego: str, vehicles: List[str], pedestrians: List[str], ego_pos: List[float], radius: float, raw_buildings, vehicle_dict = None, pedestrians_dict = None, path=None, image_size=None) -> Tuple[dict, dict]:
    t = time.time()
    radius_vehicles = {}
    radius_pedestrians = {}
    for v in vehicles:
        if vehicle_dict is None:
            v_pos = traci.vehicle.getPosition(v)
            v_angle = traci.vehicle.getAngle(v)
        else:
            v_pos = [vehicle_dict[v]['pos_x'], vehicle_dict[v]['pos_y']]
            v_angle = vehicle_dict[v]['angle']
        # check if i vehicle is within radius
        if np.linalg.norm(np.array(v_pos[0:1]) - np.array(ego_pos[0:1])) < radius:
            radius_vehicles[v] = [v_pos[0], v_pos[1], v_angle]
    for p in pedestrians:
        if pedestrians_dict is None:
            p_pos = traci.person.getPosition(p)
            p_angle = traci.person.getAngle(p)
        else:
            p_pos = pedestrians_dict[p]['position']
            p_angle = pedestrians_dict[p]['angle']
        # check if i vehicle is within radius
        if np.linalg.norm(np.array(p_pos) - np.array(ego_pos)) < radius:
            radius_pedestrians[p] = [p_pos[0], p_pos[1], p_angle]
    # get the relevant buildings
    close_buildings = list(filter_buildings(raw_buildings, ego_pos[0], ego_pos[1], 100))
    c_buildings = {}
    for b in raw_buildings:
        if b in close_buildings:
            c_buildings[b] = raw_buildings[b]
    tmp_filename = plot_vehicles_buildings(radius_vehicles, radius_pedestrians, c_buildings,
                            [ego_pos[0], ego_pos[1], ego_pos[2]], radius, path=path, image_size=image_size)

    return radius_vehicles, radius_pedestrians, tmp_filename

def get_tmp_image_tensor(tmp_filename: str = 'image.jpg', transform: transforms = None):
    img = Image.open(f'tmp/{tmp_filename}').convert('L')
    tensor = transform(img)
    # delete the tmp file
    os.remove(f'tmp/{tmp_filename}')
    return tensor
