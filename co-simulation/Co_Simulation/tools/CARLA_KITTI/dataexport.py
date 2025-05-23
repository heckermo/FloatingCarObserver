"""
This file contains all the methods responsible for saving the generated data in the correct output format.

"""
import cv2
import numpy as np
import os
import logging
import math

def save_groundplanes(planes_fname, player_transform, lidar_height):
    from math import cos, sin
    """ Saves the groundplane vector of the current frame.
        The format of the ground plane file is first three lines describing the file (number of parameters).
        The next line is the three parameters of the normal vector, and the last is the height of the normal vector,
        which is the same as the distance to the camera in meters.
    """
    rotation = player_transform.rotation
    pitch, roll = rotation.pitch, rotation.roll
    # Since measurements are in degrees, convert to radians
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    # Rotate normal vector (y) wrt. pitch and yaw
    normal_vector = [cos(pitch) * sin(roll),
                     -cos(pitch) * cos(roll),
                     sin(pitch)
                     ]
    normal_vector = map(str, normal_vector)
    with open(planes_fname, 'w') as f:
        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Height 1\n")
        f.write("{} {}\n".format(" ".join(normal_vector), lidar_height))
    logging.info("Wrote plane data to %s", planes_fname)


def save_ref_files(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')
        logging.info("Wrote reference files to %s", path)


def save_image_data(filename, image):
    logging.info("Wrote image data to %s", filename)
    # Convert to correct color format
    color_fmt = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, color_fmt)


def save_lidar_data(filename, point_cloud):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the lidar coordinate system from KITTI is defined as
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.

        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
    """
    logging.info("Wrote lidar data to %s", filename)
    point_cloud = np.copy(point_cloud)
    point_cloud[:, 1] = -point_cloud[:, 1]
    lidar_array = np.array(point_cloud).astype(np.float32)
    lidar_array.tofile(filename)


def save_kitti_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)


def save_calibration_matrices(filename, intrinsic_mat, lidar_cam_mat):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)
    R0 = np.identity(3)

    # TODO Remove the bellow hardcoded rotation matrix and write it based on the lidar_cam_mat matrix.
    # Currently it is assumed that the lidar has the same rotation as the camera but can vary in location (see below)
    #  We need to convert left-hand camera frame to right-hand kitti camera frame.

    R_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])

    # Add translation vector from velo to camera.
    T_velodyne = np.array([lidar_cam_mat[1, 3], -lidar_cam_mat[2, 3], lidar_cam_mat[0, 3]])
    TR_velodyne = np.column_stack((R_velodyne, T_velodyne))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    logging.info("Wrote all calibration matrices to %s", filename)
