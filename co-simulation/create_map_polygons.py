"""
This file creates a set of polygons that represent the map of teh simulation environment.
"""
import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union, polygonize
import tqdm
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import xml.dom.minidom as minidom

def merge_pointclouds(map_name: str, base_path: str = 'measurements', min_height: float = 1.0, max_height: float = 20.0) -> np.ndarray:
    """
    Merge all pointclouds from the given map into a single pointcloud, applying
    the appropriate yaw rotation and translation to each pointcloud.

    Args:
        map_name (str): Name of the map to merge the pointclouds from.
        base_path (str): Base path to the measurements folder.
        min_height (float): Minimum height to filter points.
        max_height (float): Maximum height to filter points.

    Returns:
        np.ndarray: Merged and filtered pointcloud.
    """
    # Load the information file
    info_path = os.path.join(base_path, map_name, 'recording_infos.pkl')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f'Information file not found at {info_path}')

    with open(info_path, 'rb') as f:
        recording_infos = pickle.load(f)  # dict with keys 'timestep', 'position', and 'rotation'

    # Ensure that 'rotation' is present in recording_infos
    if 'carla_rotation' not in recording_infos:
        raise KeyError("'rotation' key not found in recording_infos.")

    fco = recording_infos['fco']

    # get the sumo2carla offset
    offset = recording_infos['sumo_carla_offset']
    print(f'Offset: {offset}')

    all_points = []
    for timestep, position, rotation in zip(recording_infos['timestep'], recording_infos['carla_position'], recording_infos['carla_rotation']):
        yaw_d, pitch, roll = rotation
        # Load the PCD file for the current timestep

        timestep_str = str(timestep).replace('.', '_')
        lidar_path = os.path.join(base_path, map_name, timestep_str, fco, 'front', 'lidar_points.pcd') 
        if not os.path.exists(lidar_path):
            # stop this iteration if the lidar pointcloud is not found
            continue
            #raise FileNotFoundError(f'Could not find the lidar pointcloud for timestep {timestep_str}')
        if abs(pitch) > 1:
            continue
        print(f'Processing timestep {timestep} with position {position} and yaw {yaw_d}, pitch {pitch}, roll {roll}')
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(lidar_path)
        points = np.asarray(pcd.points)[:, :3]
        points[:,1] = -points[:,1] # flip the y axis to go back to the right handed coordinate system

        yaw_rad = np.deg2rad(yaw_d)

        R = np.array([
        [ np.cos(yaw_rad), -np.sin(yaw_rad), 0 ],
        [ np.sin(yaw_rad),  np.cos(yaw_rad), 0 ],
        [                0,                0, 1 ]
        ])
        T = np.array([position[0], position[1], position[2]])

        # Apply rotation and translation
        transformed_points = (R @ points.T).T + np.array(T)

        all_points.append(transformed_points)

        print(f'Loaded and transformed pointcloud for timestep {timestep_str}, points shape: {transformed_points.shape}')

    # Concatenate all points into a single array
    all_points = np.vstack(all_points)
    print(f'Initial merged pointcloud shape: {all_points.shape}')

    # Get the min value in the z axis
    min_z = np.min(all_points[:, 2])

    # Move all points to the ground plane
    all_points[:, 2] -= min_z

    # Filter out points that are too high or too low
    filtered_points = all_points[(all_points[:, 2] > min_height) & (all_points[:, 2] < max_height)]
    print(f'Merged pointcloud shape after filtering: {filtered_points.shape}')

    # Save the merged pointcloud
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    output_pcd_path = os.path.join(base_path, map_name, 'all_points.pcd')
    o3d.io.write_point_cloud(output_pcd_path, merged_pcd)
    print(f'Saved merged pointcloud to {output_pcd_path}')

    return filtered_points

def map_points_to_groundplane(all_points, save_plot: bool = True, plot_name: str = 'map.png'):
    """
    Map all points to the ground plane by setting the z value to 0.

    Args:
        all_points (np.array): Array of points to map to the ground plane.
        save_plot (bool): Whether to save a scatter plot of the ground plane.
        plot_name (str): Filename for the saved plot.

    Returns:
        np.array: Array of points mapped to the ground plane.
    """
    # Since points are already moved to the ground plane in merge_pointclouds,
    # we simply set z to 0.
    ground_plane_points = all_points.copy()
    ground_plane_points[:, 2] = 0

    if save_plot:
        plt.figure(figsize=(10, 10))
        plt.scatter(ground_plane_points[:, 0], ground_plane_points[:, 1], s=1, c='blue', alpha=0.5)
        plt.title('Ground Plane Point Cloud')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(plot_name, dpi=300)
        plt.close()
        print(f'Saved ground plane plot to {plot_name}')

    return ground_plane_points

def cluster_and_create_polygons(all_points, eps=2.0, min_samples=5, alpha=1.5, save_plot: bool = True, plot_name: str = 'clusters.png'):
    """
    Cluster the 2D points and create polygons for each cluster.

    Args:
        all_points (np.array): Array of points mapped to the ground plane.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood (DBSCAN parameter).
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN parameter).
        alpha (float): Alpha parameter for the alpha shape (concave hull).
        save_plot (bool): Whether to save the cluster and polygon plot.
        plot_name (str): Filename for the saved plot.

    Returns:
        list of shapely.geometry.Polygon: List of polygons representing each cluster.
        np.array: Array of cluster labels corresponding to all_points.
    """
    from sklearn.cluster import DBSCAN
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union, polygonize

    # Extract 2D coordinates
    points_2d = all_points[:, :2]

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_2d)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    print(f'Number of clusters found: {len(unique_labels)}')

    clusters = {label: points_2d[labels == label] for label in unique_labels}
    polygons = []

    for label, cluster_points in tqdm.tqdm(clusters.items(), desc='Creating polygons'):
        if len(cluster_points) < 750:
            print(f'Cluster {label} has less than 1000 points, skipping polygon creation.')
            continue

        # Create convex hull
        try:
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            polygon = Polygon(hull_points)
            polygons.append(polygon)
        except Exception as e:
            print(f'Could not create convex hull for cluster {label}: {e}')

    print(f'Created {len(polygons)} polygons from clusters.')

    if save_plot:
        plt.figure(figsize=(12, 12))
        #plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, cmap='tab20', s=1, alpha=0.5, marker='o')
        for polygon in polygons:
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=2)
        plt.title('Clusters and Their Convex Hulls')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(plot_name, dpi=300)
        plt.close()
        print(f'Saved clusters and polygons plot to {plot_name}')
    
    # save the polygons to a file in sumo format
    save_polygons_to_custom_xml(polygons, filename='polygons.add.xml', polygon_type='building')

    return polygons, labels

def create_alpha_shape(points, alpha):
    """
    Create an alpha shape (concave hull) for a set of 2D points.

    Args:
        points (np.array): Nx2 array of points.
        alpha (float): Alpha parameter.

    Returns:
        shapely.geometry.Polygon or MultiPolygon: The alpha shape polygon.
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union, polygonize

    if len(points) < 4:
        return Polygon(points).convex_hull

    tri = Delaunay(points)
    edges = set()
    edge_length = lambda a, b: np.linalg.norm(a - b)

    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            edges.add(edge)

    edge_points = []
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        if edge_length(p1, p2) < alpha:
            edge_points.append((tuple(p1), tuple(p2)))

    m = MultiPoint([pt for edge in edge_points for pt in edge])
    return unary_union(list(polygonize(m)))

def save_polygons_to_custom_xml(polygons, filename='polygons.add.xml', 
                                polygon_type='forest', color='140,196,107', 
                                fill='1', layer='-3.00', id_prefix='poly'):
    """
    Save a list of shapely Polygons to a custom SUMO .add.xml file with <poly> elements.

    Args:
        polygons (list of shapely.geometry.Polygon): List of polygons to save.
        filename (str): The filename for the output .add.xml file.
        polygon_type (str): The type attribute for each polygon (e.g., 'forest', 'building').
        color (str): The color attribute in "R,G,B" format (e.g., '140,196,107').
        fill (str): The fill attribute, typically '1' for filled polygons.
        layer (str): The layer attribute indicating rendering order.
        id_prefix (str): Prefix for the polygon IDs to ensure uniqueness.
    """
    # Create the root 'additional' element
    additional = ET.Element("additional")

    for i, polygon in enumerate(polygons):
        # Generate a unique ID for each polygon
        poly_id = f"{id_prefix}_{i}"

        # Create a 'poly' element with the required attributes
        poly_element = ET.SubElement(
            additional, "poly", 
            id=poly_id, 
            type=polygon_type, 
            color=color, 
            fill=fill, 
            layer=layer
        )

        # Extract coordinates in the format "x1,y1 x2,y2 x3,y3 ..."
        coords = " ".join([f"{x},{y}" for x, y in polygon.exterior.coords])

        # Set the 'shape' attribute with the coordinates
        poly_element.set("shape", coords)

    # Convert the ElementTree to a string
    rough_string = ET.tostring(additional, 'utf-8')

    # Parse the string using minidom for pretty printing
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Write the pretty XML to the specified file
    with open(filename, 'w') as f:
        f.write(pretty_xml)

    print(f"Saved {len(polygons)} polygons to {filename}")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Merge point clouds, map to ground plane, and create polygons.')
    parser.add_argument('--map_name', type=str, default='Town03', help='Name of the map to process.')
    parser.add_argument('--base_path', type=str, default='measurements', help='Base path to the measurements folder.')
    parser.add_argument('--min_height', type=float, default=4.5, help='Minimum height to filter points.')
    parser.add_argument('--max_height', type=float, default=6, help='Maximum height to filter points.')
    parser.add_argument('--cluster_eps', type=float, default=1.0, help='DBSCAN eps parameter.')
    parser.add_argument('--cluster_min_samples', type=int, default=100, help='DBSCAN min_samples parameter.')
    parser.add_argument('--alpha', type=float, default=1.5, help='Alpha parameter for alpha shapes.')
    parser.add_argument('--save_plots', action='store_true', help='Whether to save plots.')
    args = parser.parse_args()

    # Merge point clouds
    all_points = merge_pointclouds(
        map_name=args.map_name,
        base_path=args.base_path,
        min_height=args.min_height,
        max_height=args.max_height
    )

    # Map points to ground plane
    all_ground_points = map_points_to_groundplane(
        all_points,
        save_plot=True,
        plot_name='ground_plane_map.png'
    )

    # Cluster and create polygons
    polygons, labels = cluster_and_create_polygons(
        all_ground_points,
        eps=args.cluster_eps,
        min_samples=args.cluster_min_samples,
        alpha=args.alpha,
        save_plot=True,
        plot_name='clusters_polygons.png'
    )

