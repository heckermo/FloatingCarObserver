import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import math
def extract_building_coordinates(poly_xml_file):
    # Parse the XML file
    tree = ET.parse(poly_xml_file)
    root = tree.getroot()

    # Extract the shape attribute for each polygon element and parse the coordinates
    building_coords = dict()
    for poly in root.iter('poly'):
        if poly.get('type') == 'building':
            shape = poly.get('shape')
            name = poly.get('id')
            coords = [tuple(map(float, coord.split(','))) for coord in shape.split()]
            building_coords[name]= coords

    return building_coords

def plot_lines(points):
    x, y = zip(*points)

    plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def interpolate_points(points, density=10, height = 10):
    interpolated_points = []

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        x_values = np.linspace(x0, x1, int(density * distance))
        y_values = np.linspace(y0, y1, int(density * distance))

        interpolated_points.extend(list(zip(x_values, y_values)))

    # Add the z=0 coordinate and convert to a numpy array
    interpolated_points_3d = np.array([(x, y, 0) for (x, y) in interpolated_points])

    return interpolated_points_3d

def extrude_points(points, z_start=0, z_end=10, density=10):
    extruded_points = []

    for x, y, z in points:
        z_values = np.linspace(z_start, z_end, int(density * (z_end - z_start)))
        extruded_points.extend([(x, y, z_val) for z_val in z_values])

    return np.array(extruded_points)

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def filter_buildings(points_dict, x_ego, y_ego, d):
    ego_point = (x_ego, y_ego)
    filtered_points = {
        key: point_set
        for key, point_set in points_dict.items()
        if all(euclidean_distance(ego_point, point) <= d for point in point_set)
    }
    return filtered_points

def get_centroid(points):
    # Calculate the average x and y coordinates
    x_sum = 0
    y_sum = 0
    for point in points:
        x_sum += point[0]
        y_sum += point[1]

    centroid_x = x_sum / len(points)
    centroid_y = y_sum / len(points)

    centroid = (centroid_x, centroid_y)

    return centroid



if __name__ == "__main__":
    poly_xml_file = 'sumo_sim/buildings.poly.xml'
    building_coords_dict = extract_building_coordinates(poly_xml_file)
    #plot_lines(building_coords_list[0])
    all_points = list()
    for building_coords in building_coords_list:
        interpolated_points = interpolate_points(building_coords, density=2)
        points_3d = extrude_points(interpolated_points, z_start=0, z_end=10, density=2)
        all_points.append(points_3d)
    all_points = np.concatenate(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    o3d.visualization.draw_geometries([pcd])