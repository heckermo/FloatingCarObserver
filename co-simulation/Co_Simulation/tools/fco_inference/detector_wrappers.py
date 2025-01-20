from abc import ABC, abstractmethod
import os
import torch
import cv2
import numpy as np
import logging
from torchvision.transforms import Compose
from pathlib import Path
import sys

# Monocon imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../D3-Detection/algorithms/monocon-pytorch"))
from model.detector.monocon_detector import MonoConDetector
from monocon_utils.data_classes import KITTICalibration
from monocon_utils.geometry_ops import extract_corners_from_bboxes_3d, points_cam2img
from transforms.default_transforms import Normalize, Pad, ToTensor
from torchvision.transforms import Compose
import math
from Co_Simulation.tools.CARLA_KITTI.dataexport import save_calibration_matrices

# OpenPCDet imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../D3-Detection/algorithms/OpenPCDet"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../D3-Detection/algorithms/OpenPCDet/tools"))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
from pcdet.utils.calibration_kitti import Calibration


DEFAULT_TEST_TRANSFORMS = [
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    Pad(size_divisor=32),
    ToTensor(),
]


class DetectorWrapper(ABC):
    @abstractmethod
    def __init__(self, weights_path: str, device: str, **kwargs):
        """
        Initialize the detector wrapper.

        Args:
            weights_path (str): Path to the model weights or checkpoint.
            device (str): Computation device ('cpu' or 'cuda').
            visualize (bool): Whether to visualize detection results.
            **kwargs: Additional arguments specific to the detector.
        """
        self.weights_path = weights_path
        self.device = device

        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        
        # Create temporary directory for any necessary files
        if not os.path.exists('tmp'):
            os.makedirs('tmp')

    @abstractmethod
    def detect(self, data: dict) -> dict:
        """
        Run detection on the provided data.

        Args:
            data (dict): A dictionary containing necessary inputs for detection.

        Returns:
            dict: Detection results.
        """
        pass


class MonoConDetectorWrapper(DetectorWrapper):
    def __init__(self, weights_path: str, device: str, **kwargs):
        """
        Initialize the MonoConDetectorWrapper.

        Args:
            weights_path (str): Path to the MonoCon model weights.
            device (str): Computation device ('cpu' or 'cuda').
            visualize (bool): Whether to visualize detection results.
            **kwargs: Additional arguments (unused).
        """
        super().__init__(weights_path, device, **kwargs)
        
        # Initialize the detector
        self.detector = MonoConDetector()
        self.detector.to(self.device)
        self.detector.eval()
        
        # Load weights
        self._load_weights()
        
        # Define image transformations
        self.transforms = Compose(DEFAULT_TEST_TRANSFORMS)
    
    def _load_weights(self):
        """
        Loads the MonoCon model weights from the specified path.
        """
        engine_dict = torch.load(self.weights_path, map_location=self.device)
        
        # Load engine attributes
        attrs = engine_dict.get('engine_attrs', {})
        for attr_k, attr_v in attrs.items():
            setattr(self, attr_k, attr_v)
        
        state_dict = engine_dict.get('state_dict', {})
        
        # Load model state dictionary
        self.detector.load_state_dict(state_dict.get('model', {}))
    
    def detect(self, data: dict) -> dict:
        """
        Run detection using the MonoCon detector.

        Args:
            data (dict): Dictionary containing 'img' (np.ndarray) and 'calib' (str or Calibration object).

        Returns:
            dict: Detection results in evaluation format.
        """
        img = data.get('img')
        calib = data.get('calib_dir')

        if isinstance(calib, str):
            calib = KITTICalibration(calib)  # Assuming KITTICalibration can be initialized from a file path
        elif not isinstance(calib, KITTICalibration):
            raise TypeError("Calibration data must be a file path or KITTICalibration instance.")

        # Prepare image metadata
        img_metas = {
            'idx': 0,
            'split': 'test',
            'sample_idx': 0,
            'ori_shape': img.shape[:2]
        }

        # Convert image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data_dict = {
            'img': img_rgb,
            'img_metas': img_metas,
            'calib': calib,
            'label': {}
        }

        # Apply transformations
        data_dict = self.transforms(data_dict)

        # Prepare data for the model
        data_dict['img'] = data_dict['img'].unsqueeze(0).to(self.device)
        data_dict['img_metas']['pad_shape'] = [data_dict['img_metas']['pad_shape']]
        data_dict['calib'] = [data_dict['calib']]

        # Run detection model
        with torch.no_grad():
            pred_dict = self.detector(data_dict, return_loss=False)

        # Adapt to eval format
        eval_format = self.detector.head._get_eval_formats(
            data_dict, pred_dict, get_vis_format=True
        )

        # Extract detection results
        vis_container = []
        vis_container.extend(eval_format)
        _pred_bbox_3d = [f['img_bbox'] for f in vis_container]
        pred_bboxes_3d = _pred_bbox_3d[0]['boxes_3d']  # (N, 7)

        return pred_bboxes_3d

class OpenPCDetDetectorWrapper(DetectorWrapper):
    def __init__(self, cfg_file: str, ckpt_file: str, device: str, visualize: bool, save_path: str = 'inference_results', **kwargs):
        """
        Initialize the OpenPCDetDetectorWrapper.

        Args:
            cfg_file (str): Path to the model configuration file.
            ckpt_file (str): Path to the model checkpoint file.
            device (str): Computation device ('cpu' or 'cuda').
            visualize (bool): Whether to visualize detection results.
            save_path (str): Directory to save inference results.
            **kwargs: Additional arguments (unused).
        """
        super().__init__(weights_path=ckpt_file, device=device, visualize=visualize, **kwargs)
        self.cfg_file = cfg_file
        self.ckpt_file = ckpt_file
        self.save_path = save_path

        # Initialize the OpenPCDet detector
        self.detector = OpenPCDetInMemoryDetector(
            cfg_file=self.cfg_file,
            ckpt_file=self.ckpt_file,
            save_path=self.save_path
        )

    def detect(self, data: dict) -> dict:
        """
        Run detection using the OpenPCDet detector.

        Args:
            data (dict): Dictionary containing 'points' (torch.Tensor), 'calib' (Calibration), 'image' (np.ndarray), and 'frame' (int).

        Returns:
            dict: Detection results.
        """
        points = data.get('points')        # torch.Tensor of shape (N, 4)
        calib = data.get('calib_dir')          # Calibration object
        calib = Calibration(calib)
        image = data.get('img')          # np.ndarray
        frame = data.get('frame', 0)       # int

        if points is None or calib is None or image is None:
            self.logger.error("Missing required data for OpenPCDet detection.")
            raise ValueError("Data dictionary must contain 'points', 'calib_dir', and 'img' keys.")

        # Perform detection
        pred_bbox_3d = self.detector.detect(points, calib, image, frame)

        return pred_bbox_3d


class OpenPCDetInMemoryDetector:
    def __init__(self, cfg_file: str, ckpt_file: str, save_path: str = 'inference_results'):
        """
        Initialize the OpenPCDetInMemoryDetector.

        Args:
            cfg_file (str): Path to the model configuration file.
            ckpt_file (str): Path to the model checkpoint file.
            save_path (str): Directory to save inference results.
        """
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        # Load configuration
        self.logger.info(f"Loading configuration from {cfg_file}")
        self.cfg = {}
        cfg_from_yaml_file(cfg_file, self.cfg)  # Assuming this function populates self.cfg

        # Create a dummy dataset to provide class_names
        self.logger.info("Initializing dataset")
        self.dataset = KittiDataset(
            dataset_cfg=self.cfg['DATA_CONFIG'],
            class_names=self.cfg['CLASS_NAMES'],
            training=False
        )

        # Build and load the model
        self.model = build_network(
            model_cfg=self.cfg['MODEL'], 
            num_class=len(self.cfg['CLASS_NAMES']), 
            dataset=self.dataset
        )
        self.model.load_params_from_file(ckpt_file, logger=self.logger, to_cpu=False)
        self.model.to('cuda')  # Assuming CUDA is available; adjust as needed
        self.model.eval()

        self.save_path = Path(save_path)
        if not self.save_path.exists():
            os.makedirs(self.save_path, exist_ok=True)

    def detect(self, points: torch.Tensor, calib: Calibration, image: np.ndarray, frame: int) -> list:
        """
        Run object detection on given point cloud data.

        Args:
            points (torch.Tensor): Point cloud data of shape (N, 4).
            calib (Calibration): Calibration data.
            image (np.ndarray): Image data as a NumPy array.
            frame (int): Frame index or identifier.

        Returns:
            list: Detected bounding boxes and associated attributes.
        """
        img_shape = image.shape[:2]

        # Convert Calibration object to rect coordinates
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        pts_rect = torch.from_numpy(pts_rect).float()
        
        # Get FOV flag
        fov_flag = self.dataset.get_fov_flag(pts_rect, img_shape, calib)
        points_fov = points[fov_flag]

        # Preprocess the point cloud
        input_dict = {
            'points': points_fov,   
            'frame_id': frame,
            'calib': calib,
            'image_shape': img_shape,
        }
        data_dict = self.dataset.prepare_data(input_dict)
        data_dict = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        # Perform inference
        with torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        # Generate prediction annotations
        annos = self.dataset.generate_prediction_dicts(
            data_dict, pred_dicts, class_names=self.cfg['CLASS_NAMES'],
            output_path=self.save_path
        )

        pred_bbox_3d = annos[0]['boxes_lidar']
        rotation = annos[0]['rotation_y']
        dimensions = annos[0]['dimensions']
        location = annos[0]['location']
        # location[:, 0] = -location[:, 0]
        n_7_bboxes = np.hstack((location, dimensions, rotation[:, np.newaxis]))

        return n_7_bboxes

