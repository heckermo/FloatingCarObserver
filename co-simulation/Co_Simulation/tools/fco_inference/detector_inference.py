import os
import sys
import torch
import argparse
import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import traci
import carla
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../D3-Detection/algorithms/monocon-pytorch"))
from model.detector.monocon_detector import MonoConDetector
from monocon_utils.data_classes import KITTICalibration
from monocon_utils.geometry_ops import extract_corners_from_bboxes_3d, points_cam2img
from transforms.default_transforms import Normalize, Pad, ToTensor
from torchvision.transforms import Compose
import math
from Co_Simulation.tools.CARLA_KITTI.dataexport import save_calibration_matrices

from tools.fco_inference.detector_wrappers import DetectorWrapper
from tools.fco_inference.detector_wrappers import MonoConDetectorWrapper, OpenPCDetDetectorWrapper

CLASSES = ['Car']
CLASS_IDX_TO_COLOR = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
}

# Define lines for drawing 3D bounding boxes
LINE_INDICES = (
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5),
    (3, 2), (3, 7), (4, 5), (4, 7),
    (2, 6), (5, 6), (6, 7)
)

ALLOWED_DETECTOR_TYPES = ["monocon"]

class CoSimulationDetector:
    def __init__(
        self, 
        detector_type: str, 
        weights_path: str, 
        device: str = 'cuda',
        cfg_file: str = None,  # Only required for OpenPCDet
        save_path: str = 'inference_results',  # Only required for OpenPCDet
        iou_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initializes the CoSimulationDetector with the specified detector type.

        Args:
            detector_type (str): The type of detector to use (e.g., 'monocon', 'openpcdet').
            weights_path (str): The path to the model weights or checkpoint.
            device (str): The computation device (e.g., 'cpu', 'cuda').
            visualize (bool): Whether to visualize detection results.
            cfg_file (str, optional): Path to the config file (required for 'openpcdet').
            save_path (str, optional): Path to save inference results (optional for 'openpcdet').
            **kwargs: Additional arguments for specific detectors.
        """
        ALLOWED_DETECTOR_TYPES = ["monocon", "openpcdet"]  # Updated detector types

        assert detector_type in ALLOWED_DETECTOR_TYPES, \
            f"Detector type {detector_type} not supported. Supported types are {ALLOWED_DETECTOR_TYPES}"
        
        self.detector_type = detector_type
        self.device = device
        self.visualize = True

        # Initialize the appropriate detector wrapper
        self.detector = self._initialize_detector_wrapper(
            detector_type=detector_type, 
            weights_path=weights_path, 
            device=device, 
            cfg_file=cfg_file,
            save_path=save_path,
            **kwargs
        )

        self.iou_threshold = iou_threshold

        # crrate an inference kitti dir
        os.makedirs('inference_kitti', exist_ok=True)
        os.makedirs('inference_kitti/image_2', exist_ok=True)
        os.makedirs('inference_kitti/label_2', exist_ok=True)
        os.makedirs('inference_kitti/calib', exist_ok=True)
        os.makedirs('inference_kitti/velodyne', exist_ok=True)
        self.kitti_mapping = {'front': '000001', 'left': '000002', 'right': '000003', 'back': '000004'}

    def _initialize_detector_wrapper(self, detector_type: str, weights_path: str, device: str, visualize: bool, cfg_file: str, save_path: str, **kwargs) -> DetectorWrapper:
        if detector_type == "monocon":
            return MonoConDetectorWrapper(
                weights_path=weights_path,
                device=device,
                visualize=visualize
            )
        elif detector_type == "openpcdet":
            if cfg_file is None:
                raise ValueError("cfg_file must be provided for OpenPCDet detector.")
            return OpenPCDetDetectorWrapper(
                cfg_file=cfg_file,
                ckpt_file=weights_path,  # Assuming weights_path refers to checkpoint file
                device=device,
                visualize=visualize,
                save_path=save_path
            )
        else:
            raise ValueError(f"Detector type {detector_type} not supported.")


    def detect(self, current_rgb_results, sensor_data, sensor_config, fco, simulation, sumo_carla_mapping, sensor_manager):
        """
        Runs object detection on the current RGB images from sensors.

        Args:
            current_rgb_results (dict): Current RGB images from sensors.
            sensor_config (dict): Configuration of the sensors.
            fco: First controlled object (e.g., ego vehicle).
            simulation: CoSimulation object.
            sumo_carla_mapping (dict): Mapping between SUMO and CARLA IDs.

        Returns:
            dict: A dictionary containing detection results and metrics.
        """
        # Initialize result dictionaries
        sensor_results_dict = dict.fromkeys(current_rgb_results.keys(), {})
        fco_results_dict = {'detected':[], 'undetected':[]}
        bbox_storage = {}
        img_storage = {}

        for sensor in current_rgb_results.keys():
            orientation = sensor.split('.')[-1]
            img = current_rgb_results[sensor]['image']

            gt_bboxes_3d = extract_n_7(current_rgb_results[sensor]['kitti_datapoint'])

            sumo_ids = current_rgb_results[sensor]['sumo_detected_ids']

            # Load LiDAR points and adjust to kitti coordinate system
            lidar_sensor = f'sensor.lidar.{orientation}'
            points = sensor_data[lidar_sensor]['points']
            points[:, 1] = -points[:, 1]
            points = np.array(points, dtype=np.float32)

            # Save the current calibration matrices
            save_calibration_matrices(os.path.join('tmp', 'calib.txt'), sensor_manager.fco_cameramanager_mapping[fco].sensors.get(sensor).get('calibration'), 
                    sensor_manager.fco_cameramanager_mapping[fco].sensors.get(lidar_sensor).get('lidar_cam_mat').get(sensor))
            calib_dir = "tmp/calib.txt"
            # copy the calibration matrices to the kitti directory
            os.system(f'cp tmp/calib.txt inference_kitti/calib/{self.kitti_mapping[orientation]}.txt')
            # save the lidar points to the kitti directory in bin format
            points.tofile(f'inference_kitti/velodyne/{self.kitti_mapping[orientation]}.bin')
            lidar_cam_mat = sensor_manager.fco_cameramanager_mapping[fco].sensors.get(lidar_sensor).get('lidar_cam_mat').get(sensor)

            detection_data = {
                'img' : img,
                'points' : points,
                'calib_dir' : calib_dir}

            # Run detection on the image
            pred_bboxes_3d = self.detector.detect(detection_data) # (N, 7)
            pred_bboxes_3d[:, 4] = 1.89
            pred_bboxes_3d[:, 5] = 4.71
            pred_labels_3d = [0 for _ in range(pred_bboxes_3d.shape[0])]

            # copy the current inference results (i.e. label) to the kitti directory they are stored in inference_results/0.txt
            os.system(f'cp inference_results/0.txt inference_kitti/label_2/{self.kitti_mapping[orientation]}.txt')

            # find the indices in gt_bbox3d that have an high iou with the pred_bboxes_3d
            high_iou_indices = find_gt_high_iou_indices(gt_bboxes_3d, pred_bboxes_3d, iou_threshold=self.iou_threshold)
            print(f"Indices of predictions with IoU >= {self.iou_threshold}:", high_iou_indices)

            detected_sumo_ids = [sumo_ids[i] for i in high_iou_indices]
            undetected_sumo_ids = [sumo_ids[i] for i in range(len(sumo_ids)) if i not in high_iou_indices]
            fco_results_dict['detected'].extend(detected_sumo_ids)
            fco_results_dict['undetected'].extend(undetected_sumo_ids)
 
            plot_bev(gt_bboxes_3d, pred_bboxes_3d, high_iou_indices, orientation)

            plot_and_save_bboxes(pred_bboxes_3d=pred_bboxes_3d,
                                 pred_labels_3d=pred_labels_3d,
                                 lidar_cam_mat=lidar_cam_mat,
                                 calib_dir=calib_dir,
                                 class_idx_to_color=CLASS_IDX_TO_COLOR,
                                 line_indices=LINE_INDICES,
                                 img=img,
                                 output_path=f'bboxes_{orientation}.png')

        return fco_results_dict

def extract_n_7(kitti_descriptors):
    """
    Extracts bounding box information from a list of KITTI descriptors and returns
    an N x 7 NumPy array in the format [x, y, z, h, w, l, ry].

    Parameters:
    ----------
    kitti_descriptors : list of dict
        A list where each element is a dictionary representing a KITTI descriptor.
        Each dictionary must contain the keys:
            - 'location': str, space-separated "x y z" coordinates.
            - 'dimensions': str, space-separated "h w l" dimensions.
            - 'rotation_y': float, rotation around the y-axis.

    Returns:
    -------
    bounding_boxes : np.ndarray
        A NumPy array of shape (N, 7), where each row represents a bounding box
        in the format [x, y, z, h, w, l, ry].

    Raises:
    ------
    ValueError:
        If any descriptor is missing required fields or if parsing fails.
    """
    bounding_boxes_list = []

    for idx, descriptor in enumerate(kitti_descriptors):
        try:
            # Extract 'location', 'dimensions', and 'rotation_y'
            location_str = descriptor.location
            dimensions_str = descriptor.dimensions
            rotation_y = descriptor.rotation_y

            # Parse the 'location' string into floats
            x, y, z = map(float, location_str.strip().split())
            
            # Parse the 'dimensions' string into floats
            h, w, l = map(float, dimensions_str.strip().split())
            
            # Append the bounding box to the list
            bounding_boxes_list.append([x, y, z, h, w, l, rotation_y])

        except KeyError as e:
            raise ValueError(f"Descriptor at index {idx} is missing key: {e}")
        except ValueError as e:
            raise ValueError(f"Error parsing descriptor at index {idx}: {e}")

    # Convert the list to a NumPy array
    bounding_boxes = np.array(bounding_boxes_list)

    return bounding_boxes

def box3d_to_bev_polygon(box3d):
    """
    Converts a 3D bounding box in format (x, y, z, h, w, l, ry)
    into a shapely Polygon in the BEV (bird's-eye view) plane.
    The BEV polygon is centered at (x, z) with dimensions (w, l),
    rotated by ry around the y-axis.

    Parameters:
    -----------
    box3d : array-like, shape = (7,)
        [x, y, z, h, w, l, ry]

    Returns:
    --------
    polygon : shapely.geometry.Polygon
        2D polygon representing the projection of the 3D box on the XZ-plane.
    """
    # Unpack the box3d
    x, y, z, h, w, l, ry = box3d

    # For BEV, we only care about x, z, w, l, and ry.
    # Create a rectangle centered at the origin (0,0) with width=w and length=l
    # The corners (in local box coords) will be:
    #   ( w/2,  l/2)
    #   ( w/2, -l/2)
    #   (-w/2, -l/2)
    #   (-w/2,  l/2)
    # We’re constructing this rectangle in shapely, then we will rotate & translate.
    half_w = w / 2.0
    half_l = l / 2.0

    corners = [
        ( half_w,  half_l),
        ( half_w, -half_l),
        (-half_w, -half_l),
        (-half_w,  half_l)
    ]

    poly = Polygon(corners)  # rectangle centered at (0,0) in XZ-plane

    # Rotate around (0,0) by ry (in degrees for shapely, so convert from radians)
    # shapely's rotate expects degrees, so we convert ry (radians) -> degrees
    deg = np.degrees(ry)
    poly = rotate(poly, deg, origin=(0, 0), use_radians=False)

    # Translate polygon so that its center is at (x, z)
    poly = translate(poly, xoff=x, yoff=z)

    return poly

def bev_iou(box_a, box_b):
    """
    Computes the BEV IoU for two boxes in format (x, y, z, h, w, l, ry).

    Returns:
    --------
    iou_value : float
        IoU between 0.0 and 1.0.
    """
    poly_a = box3d_to_bev_polygon(box_a)
    poly_b = box3d_to_bev_polygon(box_b)

    inter_area = poly_a.intersection(poly_b).area
    union_area = poly_a.union(poly_b).area

    if union_area == 0.0:
        return 0.0
    return inter_area / union_area

def find_gt_high_iou_indices(gt_bbox3d, pred_bbox3d, iou_threshold=0.7):
    """
    Finds indices of ground-truth bounding boxes that have a BEV IoU >= iou_threshold
    with at least one predicted bounding box.

    Parameters:
    -----------
    gt_bbox3d : np.ndarray, shape = (N,7)
        Ground-truth bounding boxes in [x, y, z, h, w, l, ry] format.
    pred_bbox3d : np.ndarray, shape = (M,7)
        Predicted bounding boxes in [x, y, z, h, w, l, ry] format.
    iou_threshold : float
        The IoU threshold to consider a match.

    Returns:
    --------
    high_iou_gt_indices : list of int
        Indices in gt_bbox3d that have IoU >= iou_threshold with at least one predicted box.
    """
    high_iou_gt_indices = []

    for gt_idx, gt_box in enumerate(gt_bbox3d):
        for pred_idx, pred_box in enumerate(pred_bbox3d):
            iou_val = bev_iou(gt_box, pred_box)
            if iou_val >= iou_threshold:
                high_iou_gt_indices.append(gt_idx)
                break  # No need to check other predictions for this GT box

    return high_iou_gt_indices

def plot_bev(gt_bbox3d, pred_bbox3d, high_iou_indices, orientation):
    """
    Plots GT and predicted bounding boxes in BEV, highlighting overlaps.

    Parameters:
    -----------
    gt_bbox3d : np.ndarray (N,7)
        Ground-truth boxes
    pred_bbox3d : np.ndarray (M,7)
        Predicted boxes
    high_iou_indices : list of int
        Indices into pred_bbox3d with IoU >= threshold
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Plot GT boxes (blue)
    for box in gt_bbox3d:
        poly_gt = box3d_to_bev_polygon(box)
        x, y = poly_gt.exterior.xy
        ax.fill(x, y, alpha=0.3, fc='blue', ec='blue', label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else "")

    # 2. Plot predicted boxes (red or green if high IoU)
    for idx, box in enumerate(pred_bbox3d):
        poly_pred = box3d_to_bev_polygon(box)
        x, y = poly_pred.exterior.xy

        if idx in high_iou_indices:
            ax.fill(x, y, alpha=0.4, fc='green', ec='green', label='Pred High IoU' if 'Pred High IoU' not in ax.get_legend_handles_labels()[1] else "")
        else:
            ax.fill(x, y, alpha=0.4, fc='red', ec='red', label='Pred' if 'Pred' not in ax.get_legend_handles_labels()[1] else "")

    # Make sure we only show each label once in the legend
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_aspect('equal', 'box')  # Ensure square axes for proper scaling

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    plt.title("BEV Overlap of GT and Pred BBoxes for Orientation {}".format(orientation))
    plt.savefig(f'bev_overlap_{orientation}.png')

def plot_and_save_bboxes(
    pred_bboxes_3d: torch.Tensor,
    pred_labels_3d: List[int],
    lidar_cam_mat: np.ndarray,
    calib_dir: str,
    img: np.ndarray,
    class_idx_to_color: Dict[int, Tuple[int, int, int]],
    line_indices: List[Tuple[int, int]],
    output_path: str
) -> np.ndarray:
    """
    Plots 3D bounding boxes on a 2D image and saves the result.

    Parameters:
    ----------
    pred_bboxes_3d : torch.Tensor
        Predicted bounding boxes, shape (N, 7), where each box is [x, y, z, h, w, l, ry].
    pred_labels_3d : List[int]
        Predicted labels corresponding to each bounding box.
    lidar_cam_mat : np.ndarray
        Transformation matrix from LiDAR to camera coordinates, shape (4, 4).
    calib_dir : str
        Directory path to the calibration file for KITTICalibration.
    img : np.ndarray
        The image on which to draw bounding boxes, shape (H, W, 3).
    class_idx_to_color : Dict[int, Tuple[int, int, int]]
        Mapping from class indices to BGR color tuples for drawing.
    line_indices : List[Tuple[int, int]]
        List of tuples indicating which corners to connect with lines.
    output_path : str
        File path to save the image with drawn bounding boxes.

    Returns:
    -------
    img_bbox : np.ndarray
        The image with drawn bounding boxes.
    """
    # Initialize lists to store corner projections if needed
    corners_list = []
    proj_corners_list = []

    # Make a copy of the image to draw on
    img_bbox = img.copy()

    # Load calibration data
    calib = KITTICalibration(calib_dir)  # Ensure KITTICalibration is properly defined

    # Process each predicted bounding box
    for bbox_3d, label_3d in zip(pred_bboxes_3d, pred_labels_3d):
        # Ensure bbox_3d is a torch.Tensor
        if not isinstance(bbox_3d, torch.Tensor):
            bbox_3d = torch.tensor(bbox_3d)

        # Extract 8 corners of the 3D bounding box
        corners = extract_corners_from_bboxes_3d(bbox_3d.unsqueeze(0))[0]  # Shape: (8, 3)
        corners_list.append(corners)

        # Transform corners to camera coordinate system
        # Assuming lidar_cam_mat is a (4,4) transformation matrix
        # Convert to homogeneous coordinates for transformation
        corners_hom = np.hstack((corners.numpy(), np.ones((corners.shape[0], 1))))
        transformed_corners = (lidar_cam_mat @ corners_hom.T).T[:, :3]  # Shape: (8, 3)

        # Project transformed corners onto 2D image plane
        proj_corners = points_cam2img(transformed_corners, calib.P2)  # Shape: (8, 2)
        proj_corners_list.append(proj_corners)

        # Adjust corner coordinates for image scaling if necessary
        # Here, 's' is a scaling factor; adjust based on your specific calibration
        s = np.reciprocal(np.array([1.0, 1.0])[::-1])  # Example scaling
        proj_corners_scaled = ((proj_corners - 1).round() * s).astype(np.int32)

        # Get color for the current label
        color = class_idx_to_color.get(label_3d, (0, 255, 0))  # Default to green if label not found

        # Draw 3D bounding box on the image by connecting the corners
        for start, end in line_indices:
            pt1 = tuple(proj_corners_scaled[start])
            pt2 = tuple(proj_corners_scaled[end])
            img_bbox = cv2.line(
                img_bbox,
                pt1,
                pt2,
                color,
                thickness=2,
                lineType=cv2.LINE_AA
            )

    # Save the image with bounding boxes
    cv2.imwrite(output_path, cv2.cvtColor(img_bbox, cv2.COLOR_RGB2BGR))

    return img_bbox

if __name__ == "__main__":
    cfg = None
    weights_path = "3D-Detection/algorithms/monocon-pytorch/checkpoints/best.pth"
    detector = CoSimulationDetector("monocon", weights_path, True)
    img_path = '/data/public_datasets/KITTI/training/image_2/000000.png'
    calib_path = '/data/public_datasets/KITTI/training/calib/000000.txt'

    detector.detect(img_path, calib_path)
 
