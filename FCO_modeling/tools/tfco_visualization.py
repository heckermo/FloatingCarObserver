import argparse
import os
import pickle
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import tqdm
import imageio

# Add the parent directory to sys.path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils.polygon_to_tensor import create_bev_tensor

"""
This script generates a plot of the dataset created with the TfcoDatasetGenerator class.
"""

def main(
    dataset_path: str,
    image_size: int = 256,
    fco_color: Tuple[int, int, int] = (255, 128, 0),
    detected_color: Tuple[int, int, int] = (0, 255, 0),
    undetected_color: Tuple[int, int, int] = (0, 0, 255),
    output_file: str = 'output.mp4',
    fps: int = 10
) -> None:
    """
    Generate a plot for the dataset.

    Args:
        dataset_path (str): Path to the dataset.
        image_size (int, optional): Size of the image. Defaults to 256.
        fco_color (Tuple[int, int, int], optional): Color for FCO vehicles. Defaults to (255, 128, 0).
        detected_color (Tuple[int, int, int], optional): Color for detected vehicles. Defaults to (0, 255, 0).
        undetected_color (Tuple[int, int, int], optional): Color for undetected vehicles. Defaults to (0, 0, 255).
        output_file (str, optional): Output video file name. Defaults to 'output.mp4'.
        fps (int, optional): Frames per second for the output video. Defaults to 10.
    """
    # Load the dataset
    dataset_pkl_path = os.path.join(dataset_path, 'dataset.pkl')
    dataset = pd.read_pickle(dataset_pkl_path)

    # Load the config file
    config_pkl_path = os.path.join(dataset_path, 'config.pkl')
    with open(config_pkl_path, 'rb') as f:
        config = pickle.load(f)

    # Filter to only include the first loop
    dataset = dataset[dataset['loop'] == 0]

    # Initialize list to store images
    images = []

    # Iterate through the dataset and process each row
    for _, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset)):
        current_vehicle_info = pd.DataFrame(row.vehicle_information).T

        # Process detected vehicles
        detected_vehicles = current_vehicle_info[
            (current_vehicle_info['detected_label'] == 1) &
            (current_vehicle_info['fco_label'] == 0)
        ]
        detected_vehicles_tensor = create_vehicle_tensor(
            detected_vehicles, image_size, config
        )
        detected_rgb_mask = create_rgb_mask(
            detected_vehicles_tensor, detected_color
        )

        # Process FCO vehicles
        fco_vehicles = current_vehicle_info[
            current_vehicle_info['fco_label'] == 1
        ]
        fco_vehicles_tensor = create_vehicle_tensor(
            fco_vehicles, image_size, config
        )
        fco_rgb_mask = create_rgb_mask(
            fco_vehicles_tensor, fco_color
        )

        # Process undetected vehicles
        undetected_vehicles = current_vehicle_info[
            (current_vehicle_info['detected_label'] == 0) &
            (current_vehicle_info['fco_label'] == 0)
        ]
        undetected_vehicles_tensor = create_vehicle_tensor(
            undetected_vehicles, image_size, config
        )
        undetected_rgb_mask = create_rgb_mask(
            undetected_vehicles_tensor, undetected_color
        )

        # Combine all masks and create an RGB image
        rgb_image_tensor = torch.clamp(
            detected_rgb_mask + fco_rgb_mask + undetected_rgb_mask, 0, 255
        ).byte()
        rgb_image = Image.fromarray(rgb_image_tensor.numpy())

        # Store the image
        images.append(rgb_image)

    # Convert images to frames and save as video
    frames = [np.array(img) for img in images]
    imageio.mimsave(output_file, frames, fps=fps, format='FFMPEG')


def create_vehicle_tensor(
    vehicles: pd.DataFrame,
    image_size: int,
    config: dict
) -> torch.Tensor:
    """
    Create a BEV tensor for the given vehicles.

    Args:
        vehicles (pd.DataFrame): Vehicle data.
        image_size (int): Size of the image.
        config (dict): Configuration dictionary.

    Returns:
        torch.Tensor: BEV tensor.
    """
    if not vehicles.empty:
        return create_bev_tensor(
            {},
            vehicles.T.to_dict(),
            'box',
            image_size,
            config['CENTER_POINT'][0],
            config['CENTER_POINT'][1],
            0,
            config['RADIUS']
        )
    else:
        return torch.zeros((1, image_size, image_size))


def create_rgb_mask(
    vehicle_tensor: torch.Tensor,
    color: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Create an RGB mask for the vehicle tensor.

    Args:
        vehicle_tensor (torch.Tensor): Vehicle tensor.
        color (Tuple[int, int, int]): RGB color.

    Returns:
        torch.Tensor: RGB mask.
    """
    return torch.stack(
        [vehicle_tensor.squeeze(0) * c for c in color],
        dim=2
    ).cpu()


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """
    Parse a color string into a tuple of integers.

    Args:
        color_str (str): Color string in the format 'R,G,B'.

    Returns:
        Tuple[int, int, int]: Tuple representing the RGB color.
    """
    try:
        color = tuple(map(int, color_str.strip().split(',')))
        if len(color) != 3:
            raise ValueError
        return color
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid color value: '{color_str}'. Expected format 'R,G,B'."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a plot for the dataset.")
    #parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--image_size', type=int, default=512, help='Size of the image')
    parser.add_argument(
        '--fco_color', type=str, default='255,128,0',
        help='Color for FCO vehicles as comma-separated RGB values (e.g., "255,128,0")'
    )
    parser.add_argument(
        '--detected_color', type=str, default='0,255,0',
        help='Color for detected vehicles as comma-separated RGB values (e.g., "0,255,0")'
    )
    parser.add_argument(
        '--undetected_color', type=str, default='0,0,255',
        help='Color for undetected vehicles as comma-separated RGB values (e.g., "0,0,255")'
    )
    parser.add_argument(
        '--output_file', type=str, default='output.mp4',
        help='Output video file name'
    )
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the output video')

    args = parser.parse_args()

    main(
        dataset_path="/home/jeremias/sumo_detector/SUMO_detector_plus/tfco_datasets/test",
        #dataset_path=args.dataset_path,
        image_size=args.image_size,
        fco_color=parse_color(args.fco_color),
        detected_color=parse_color(args.detected_color),
        undetected_color=parse_color(args.undetected_color),
        output_file=args.output_file,
        fps=args.fps
    )
