import tqdm
import os
import shutil
import random
from typing import List, Tuple
import sys
import time

import os
import shutil
import random
from typing import List, Tuple
import tqdm
import time
import numpy as np
from Co_Simulation.tools.utils import config_to_dict
import yaml

import os
import shutil
from typing import List, Tuple
import tqdm
import open3d as o3d
from easydict import EasyDict
from pathlib import Path

def convert_pcd_to_kitti(pcd_file, output_bin_file):
    """
    Convert a PCD file to KITTI Velodyne format (.bin).
    
    Args:
        pcd_file (str): Path to the input PCD file.
        output_bin_file (str): Path to save the output KITTI .bin file.
    """
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    
    # Check if the intensity channel exists
    if pcd.has_colors():
        intensity = np.mean(np.asarray(pcd.colors), axis=1)  # Use RGB average as intensity
    else:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)  # Default intensity to 0
    
    # Combine x, y, z, and intensity
    kitti_points = np.hstack((points, intensity.reshape(-1, 1))).astype(np.float32)
    
    # Save to KITTI .bin format
    kitti_points.tofile(output_bin_file)

def prepare_dataset(
    train_dataset_paths: List[str],
    test_dataset_paths: List[str],
    copy_lidar: bool = False,
    copy_bounding_box_image: bool = False,
    create_metadata: bool = False,
    keep_empty: bool = False,
    target_dir: str = 'tmp_kitti'
) -> None:
    """
    Prepares the dataset by flattening multiple training and testing datasets into a single
    KITTI-compatible format. All entries from the provided training datasets are combined 
    into a single training set, and similarly for the testing datasets.

    Args:
        train_dataset_paths (List[str]): A list of root directories for training datasets.
        test_dataset_paths (List[str]): A list of root directories for testing datasets.
        copy_lidar (bool): Whether to copy LiDAR data as well.
        keep_empty (bool): Whether to keep data points with empty label files.
        target_dir (str): Directory to save the prepared dataset.

    Each dataset in train_dataset_paths/test_dataset_paths is expected to have a structure:
        dataset_path/
            timestep/
                vehicle/
                    camera/
                        calib.txt
                        rgb_image.png
                        bounding_box_image.png
                        kitti_datapoint.txt
                        lidar_points.pcd

    This function flattens all given datasets and organizes them into a KITTI-compatible format:
    target_dir/
        training/
            calib/
            image_2/
            label_2/
            bounding_box_image/ (optional)
            velodyne/ (optional)
        testing/
            calib/
            image_2/
            label_2/
            bounding_box_image/ (optional)
            velodyne/ (optional)
    """

    # Remove existing target_dir directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Create required directories
    for split in ['training', 'testing']:
        for folder in ['calib', 'image_2', 'label_2']:
            os.makedirs(os.path.join(target_dir, split, folder), exist_ok=True)
        if copy_lidar:
            os.makedirs(os.path.join(target_dir, split, 'velodyne'), exist_ok=True)  
        if copy_bounding_box_image:
            os.makedirs(os.path.join(target_dir, split, 'bounding_box_image'), exist_ok=True)
        if create_metadata:
            os.makedirs(os.path.join(target_dir, split, 'metadata'), exist_ok=True)

    # Helper to collect entries from a single dataset path
    def collect_entries(dataset_path: str, front_only=False) -> List[Tuple[str, str, str, str]]:
        entries = []
        timestep_dirs = sorted(os.listdir(dataset_path))
        for timestep in timestep_dirs:
            timestep_path = os.path.join(dataset_path, timestep)
            if os.path.isdir(timestep_path):
                for vehicle in os.listdir(timestep_path):
                    vehicle_path = os.path.join(timestep_path, vehicle)
                    if os.path.isdir(vehicle_path):
                        for camera in os.listdir(vehicle_path):
                            if front_only and camera != 'front':
                                continue
                            camera_path = os.path.join(vehicle_path, camera)
                            if os.path.isdir(camera_path):
                                # Store the dataset path as well to know where to copy from
                                entries.append((dataset_path, timestep, vehicle, camera))
        return entries

    # Collect training entries from all training dataset paths
    training_entries = []
    for path in train_dataset_paths:
        training_entries.extend(collect_entries(path))

    # Collect testing entries from all testing dataset paths
    testing_entries = []
    for path in test_dataset_paths:
        testing_entries.extend(collect_entries(path))

    # Helper function to copy data entries to target directories
    def copy_entries(entries: List[Tuple[str, str, str, str]], split: str):
        counter = 1  # Start counter at 1 for each split
        for (dataset_path, timestep, vehicle, orientation) in tqdm.tqdm(entries, desc=f'Copying {split} data'):
            source_dir = os.path.join(dataset_path, timestep, vehicle, orientation)
            if not os.path.exists(source_dir):
                continue

            label_src = os.path.join(source_dir, 'kitti_datapoint.txt')
            if not os.path.exists(label_src):
                # Skip if no label file
                continue

            # Check emptiness / 'DontCare' conditions
            if not keep_empty:
                with open(label_src, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    lines = content.split('\n')
                    labels = [line.split(' ')[0] for line in lines]
                    if all(label == 'DontCare' for label in labels):
                        continue

            filename = f'{counter:06d}'

            # Copy calibration file
            calib_src = os.path.join(source_dir, 'calib.txt')
            calib_dst = os.path.join(target_dir, split, 'calib', f'{filename}.txt')
            if os.path.exists(calib_src):
                shutil.copy(calib_src, calib_dst)

            # Copy image file
            image_src = os.path.join(source_dir, 'rgb_image.png')
            image_dst = os.path.join(target_dir, split, 'image_2', f'{filename}.png')
            if os.path.exists(image_src):
                shutil.copy(image_src, image_dst)

            # Copy label file
            label_dst = os.path.join(target_dir, split, 'label_2', f'{filename}.txt')
            shutil.copy(label_src, label_dst)

            # Copy LiDAR file if required (one directory up from camera)
            if copy_lidar:
                lidar_src = os.path.join(source_dir, 'lidar_points.pcd')
                if os.path.exists(lidar_src):
                    lidar_dst = os.path.join(target_dir, split, 'velodyne', f'{filename}.bin')
                    convert_pcd_to_kitti(lidar_src, lidar_dst)
            
            if copy_bounding_box_image:
                bb_image_src = os.path.join(source_dir, 'bounding_box_image.png')
                bb_image_dst = os.path.join(target_dir, split, 'bounding_box_image', f'{filename}.png')
                if os.path.exists(bb_image_src):
                    shutil.copy(bb_image_src, bb_image_dst)
            
            if create_metadata:
                metadata_dst = os.path.join(target_dir, split, 'metadata', f'{filename}.txt')
                with open(metadata_dst, 'w') as f:
                    f.write(f'{dataset_path}\n{timestep}\n{vehicle}\n{orientation}\n')

            counter += 1  # Increment counter

    # Copy training data from aggregated entries
    copy_entries(training_entries, 'training')

    # Copy testing data from aggregated entries
    copy_entries(testing_entries, 'testing')



def create_imagessets(target_path: str = 'tmp_kitti/ImageSets', mode: str = 'training', dataset_path: str = 'tmp_kitti'):
    """
    Creates ImageSets files required by some 3D detection algorithms for training and testing.
    This function reads the converted dataset, extracts image identifiers, and saves them
    in a .txt file within the target_path's ImageSets folder.

    Args:
        target_path (str): Path to save the ImageSets files.
        mode (str): Mode of the dataset, either 'training' or 'testing'.
    """
    # Ensure valid mode
    assert mode in ['training', 'testing'], f"Invalid mode: {mode}"

    # Ensure ImageSets directory exists
    os.makedirs(target_path, exist_ok=True)

    # First delete the existing ImageSets file
    existing_file_path = os.path.join(target_path, f'{mode}.txt')
    if os.path.exists(existing_file_path):
        os.remove(existing_file_path)

    # Get sorted list of file identifiers without extensions
    image_files = os.listdir(os.path.join(dataset_path, mode, 'calib'))
    image_ids = sorted([file.split('.')[0] for file in image_files])

    # Write image IDs to the ImageSets file
    output_file_path = os.path.join(target_path, f'{mode}.txt')
    with open(output_file_path, 'w') as f:
        for file_id in image_ids:
            f.write(file_id + '\n')

    print(f"{mode.capitalize()} ImageSets file created at: {output_file_path}")

class BaseDetectionWrapper:
    def __init__(self, config, dataset_root, run_name):
        self.config = config
        self.dataset_root = dataset_root
        self.run_name = run_name

    def prepare_training(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

class MonoconDetectionWrapper(BaseDetectionWrapper):
    def __init__(self, config, dataset_root, run_name):
        super().__init__(config, dataset_root, run_name)
        self.do_imports()
        self.cfg = self.get_default_cfg()
        self.prepare_training()
        import wandb
    
    def do_imports(self):
        sys.path.append("D3-Detection/algorithms/monocon-pytorch")
        from engine.monocon_engine import MonoconEngine
        from utils.engine_utils import get_default_cfg
        self.MonoconEngine = MonoconEngine
        self.get_default_cfg = get_default_cfg

    def prepare_training(self):
        self.cfg.DATA.ROOT = os.path.abspath(self.dataset_root)
        self.cfg.SOLVER.OPTIM.NUM_EPOCHS = self.config['training']['num_epochs']
        self.cfg.MODEL.LOAD_MODEL = self.config['training']['pre_training_path']
        self.cfg.OUTPUT_DIR = os.path.join('d3_detection_trainings', self.run_name)
        self.engine = self.MonoconEngine(self.cfg, auto_resume=False)
    
    def start_wandb(self):
        wandb.init(project='monocon', name=self.run_name, config=self.cfg, mode='disabled')

    def train(self):
        self.start_wandb()
        self.engine.train()

class PointPillarDetectionWrapper(BaseDetectionWrapper):
    def __init__(self, config, dataset_root, run_name):
        super().__init__(config, dataset_root, run_name)
        self.do_imports()
        self.prepare_training()
    
    def do_imports(self):
        sys.path.append("D3-Detection/algorithms/OpenPCDet")
        sys.path.insert(0, "D3-Detection/algorithms/OpenPCDet/tools")
        from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
        self.create_kitti_infos = create_kitti_infos
        from tools.train import main
        self.train = main
        self.model_cfg_path = os.path.join('D3-Detection/algorithms/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml')
        self.dataset_cfg_path = os.path.join('D3-Detection/algorithms/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml')

    def prepare_training(self): 
        sys.argv.extend([
            '--pretrained_model', self.config['training']['pre_training_path'],
            '--cfg_file', self.model_cfg_path,
            '--epochs', str(self.config['training']['num_epochs']),
            '--output_dir', os.path.join('d3_detection_trainings', self.run_name),
        ])

        # run the data preparation script
        self.create_kitti_infos(
            EasyDict(yaml.safe_load(open(self.dataset_cfg_path))),
            class_names=['Car'],
            data_path=Path('tmp_kitti'),
            save_path=Path('tmp_kitti')
        )


    def train(self):
        self.train()

WRAPPER_MAPPINGS = {
    'monocon': MonoconDetectionWrapper,
    'pointpillar': PointPillarDetectionWrapper  
}

def main():
    config_path = os.path.join('configs', 'd3_detection.yaml')
    config = config_to_dict(config_path)
    prepare_dataset(
        config['dataset']['train_paths'], 
        config['dataset']['test_paths'], 
        copy_lidar=True, 
        keep_empty=False, 
        copy_bounding_box_image=True, 
        create_metadata=True
    )
    create_imagessets('tmp_kitti/ImageSets', mode='training')
    create_imagessets('tmp_kitti/ImageSets', mode='testing')

    training_name = f'{config["3d_detector"]}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'

    # Create a new directory in d3_detection_trainings with the training name
    training_dir = os.path.join('d3_detection_trainings', training_name)
    os.makedirs(training_dir, exist_ok=True)

    detector_wrapper = WRAPPER_MAPPINGS[config['3d_detector']](config, 'tmp_kitti', training_name)
    detector_wrapper.train()



if __name__ == "__main__":
    # This script currently only supports the monocon detector, we currently are in teh developemnt of a wrapper that also allows the openpcdet (pointpillar) detector
    main()
