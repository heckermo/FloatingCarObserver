import os
import logging
import numpy as np
import torch
import cv2
from collections import defaultdict
from pathlib import Path
import sys
sys.path.append('/home/jeremias/OpenPCDet')

# Import necessary OpenPCDet modules
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
from pcdet.utils.calibration_kitti import Calibration

class OpenPCDetDetector:
    def __init__(self, cfg_file, ckpt_file, save_path='inference_results'):
        """
        Initialize the OpenPCDetDetector.
        
        Args:
            cfg_file (str): Path to the model configuration file.
            ckpt_file (str): Path to the model checkpoint file.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Load configuration
        self.logger.info(f"Loading configuration from {cfg_file}")
        cfg_from_yaml_file(cfg_file, cfg)
        self.cfg = cfg

        # Create a dummy dataset to provide class_names
        self.logger.info("Initializing dataset")
        self.dataset = KittiDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False
        )

        # Build and load the model
        self.model = build_network(
            model_cfg=cfg.MODEL, 
            num_class=len(cfg.CLASS_NAMES), 
            dataset=self.dataset
        )
        self.model.load_params_from_file(ckpt_file, logger=self.logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

        self.save_path = Path(save_path)
        if not self.save_path.exists():
            os.makedirs(self.save_path, exist_ok=True)

    def detect(self, point_cloud_file, calib_file, image_file, frame):
        """
        Run object detection on a given point cloud file.
        
        Args:
            point_cloud_file (str): Path to the .bin file containing the point cloud.
            calib_file (str): Path to the calibration file.
            image_file (str): Path to the image file.
        
        Returns:
            list: Detected bounding boxes and associated attributes (e.g., class labels, scores).
        """
        # Load calibration data
        calib = self._load_calib(calib_file)

        # Load image data
        image = self._load_image(image_file)
        img_shape = image.shape[:2]

        # Load the point cloud data
        points = self._load_point_cloud(point_cloud_file)
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = self.dataset.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]

        # Preprocess the point cloud
        input_dict = {
            'points': points,   
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
            data_dict, pred_dicts, class_names=self.cfg.CLASS_NAMES,
            output_path=Path('inferece_results')
        )

        return annos

    def _load_point_cloud(self, point_cloud_file):
        if not os.path.isfile(point_cloud_file):
            self.logger.error(f"Point cloud file {point_cloud_file} does not exist.")
            raise FileNotFoundError(f"Point cloud file {point_cloud_file} not found.")
        point_cloud = np.fromfile(point_cloud_file, dtype=np.float32).reshape(-1, 4)
        return torch.tensor(point_cloud, dtype=torch.float32)

    def _load_calib(self, calib_file):
        if not os.path.isfile(calib_file):
            self.logger.error(f"Calibration file {calib_file} does not exist.")
            raise FileNotFoundError(f"Calibration file {calib_file} not found.")
        return Calibration(calib_file)

    def _load_image(self, image_file):
        if not os.path.isfile(image_file):
            self.logger.error(f"Image file {image_file} does not exist.")
            raise FileNotFoundError(f"Image file {image_file} not found.")
        image = cv2.imread(image_file)
        if image is None:
            self.logger.error(f"Failed to load image {image_file}.")
            raise ValueError(f"Image file {image_file} could not be loaded.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def save_predictions(self, pred_dicts, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for pred in pred_dicts:
                boxes = pred['pred_boxes'].cpu().numpy()
                scores = pred['pred_scores'].cpu().numpy()
                labels = pred['pred_labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    class_name = self.cfg.CLASS_NAMES[label - 1]
                    f.write(f"{class_name} {score} {box.tolist()}\n")
        self.logger.info(f"Saved predictions to {output_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="OpenPCDet Detector")
    parser.add_argument('--cfg_file', type=str, required=False, default="/home/jeremias/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml")
    parser.add_argument('--ckpt_file', type=str, required=False, default="/home/jeremias/OpenPCDet/output/home/jeremias/OpenPCDet/tools/cfgs/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_9.pth")
    parser.add_argument('--velodyne_dir', type=str, required=False, default='/home/jeremias/OpenPCDet/data/kitti_synthetic/testing/velodyne')
    parser.add_argument('--calib_dir', type=str, required=False, default='/home/jeremias/OpenPCDet/data/kitti_synthetic/testing/calib')
    parser.add_argument('--image_dir', type=str, required=False, default='/home/jeremias/OpenPCDet/data/kitti_synthetic/testing/image_2')
    parser.add_argument('--output_dir', type=str, required=False, default=None)
    args = parser.parse_args()

    detector = OpenPCDetDetector(args.cfg_file, args.ckpt_file)

    velodyne_files = {f[:-4]: os.path.join(args.velodyne_dir, f) for f in os.listdir(args.velodyne_dir) if f.endswith('.bin')}
    calib_files = {f[:-4]: os.path.join(args.calib_dir, f) for f in os.listdir(args.calib_dir) if f.endswith('.txt')}
    image_files = {f[:-4]: os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg'))}

    common_keys = velodyne_files.keys() & calib_files.keys() & image_files.keys()
    sorted_keys = sorted(common_keys, key=lambda x: int(x))  # Convert strings to integers for sorting


    for key in sorted_keys:
        frame = velodyne_files[key].split('/')[-1].split('.')[0]
        pred_dicts = detector.detect(velodyne_files[key], calib_files[key], image_files[key], frame)
        