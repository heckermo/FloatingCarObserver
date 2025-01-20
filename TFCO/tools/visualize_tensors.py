import torch
from PIL import Image, ImageDraw, ImageColor
import numpy as np
from typing import Optional

def compare_tensors(
    output_tensor: torch.Tensor,
    target_tensor: Optional[torch.Tensor] = None,
    path: Optional[str] = None,
    image_size: int = 512,
    output_circle_size: int = 4,
    output_color: str = 'green',
    target_circle_size: int = 4,
    target_color: str = 'red'
) -> None:
    """
    Creates a visualization that compares the output tensor with the target tensor.
    Both can be either BEV (Bird's Eye View) tensors or raw tensors but should contain only a single sequence.
    If `target_tensor` is not provided, only the `output_tensor` will be visualized.
    """
    # Ensure output_tensor is on CPU
    output_tensor = output_tensor.detach().cpu()

    # Ensure output_tensor is a single sequence
    if output_tensor.dim() > 3:
        raise ValueError("Output tensor should not have more than 3 dimensions (use a single item in batch and sequence).")
    
    # Determine if output_tensor is a BEV tensor (channels, height, width)
    is_output_bev = output_tensor.dim() == 3

    # Use image_size from tensor if BEV tensor
    if is_output_bev:
        image_size = output_tensor.shape[-1]

    # Convert output_tensor to BEV image
    if not is_output_bev:
        output_image = create_bev_from_rawtensor(
            raw_tensor=output_tensor,
            image_size=image_size,
            flag='output',
            size=output_circle_size,
            color=output_color
        )
    else:
        output_image = create_bev_from_bevtensor(
            bev_tensor=output_tensor,
            color=output_color
        )

    # If target_tensor is provided, process it
    if target_tensor is not None:
        # Ensure target_tensor is on CPU
        target_tensor = target_tensor.detach().cpu()

        if target_tensor.dim() > 3:
            raise ValueError("Target tensor should not have more than 3 dimensions (use a single item in batch and sequence).")
        
        is_target_bev = target_tensor.dim() == 3

        # Ensure image sizes match
        if is_target_bev and target_tensor.shape[-1] != image_size:
            raise ValueError("Image size of target_tensor does not match output_tensor.")

        # Convert target_tensor to BEV image
        if not is_target_bev:
            target_image = create_bev_from_rawtensor(
                raw_tensor=target_tensor,
                image_size=image_size,
                flag='target',
                size=target_circle_size,
                color=target_color
            )
        else:
            target_image = create_bev_from_bevtensor(
                bev_tensor=target_tensor,
                color=target_color
            )

        # Combine the two images for comparison
        combined_image = Image.blend(
            output_image.convert('RGBA'),
            target_image.convert('RGBA'),
            alpha=0.5
        )
    else:
        # If no target_tensor, use the output_image
        combined_image = output_image

    # Save or display the image
    if path:
        combined_image.save(path)
    else:
        combined_image.save('output_comparison.png')

def create_output_target_comparison(output_tensor, target_tensor, path=None):
    output_tensor = output_tensor.detach().cpu()
    target_tensor = target_tensor.detach().cpu()

    relevant_mask = target_tensor[:, 0] == 1
    target_tensor = target_tensor[relevant_mask]
    output_tensor = output_tensor[relevant_mask]

    output_image = create_bev_from_rawtensor(
        raw_tensor=output_tensor,
        image_size=512,
        flag='output',
        size=4,
        label_to_color={1: 'pink'}
    )

    target_image = create_bev_from_rawtensor(
        raw_tensor=target_tensor,
        image_size=512,
        flag='target',
        size=4,
    )

    combined_image = Image.blend(
        output_image.convert('RGBA'),
        target_image.convert('RGBA'),
        alpha=0.5
    )

    if path:
        combined_image.save(path)

def create_bev_from_rawtensor(
    raw_tensor: torch.Tensor,
    image_size: int,
    flag: str,
    size: int = 4,
    label_to_color: dict = {0: 'red', 1: 'blue', 2: 'green'},
    save_path: Optional[str] = None
) -> Image.Image:
    """
    Converts a raw tensor to a BEV image by plotting points on an image.

    The tensor values are expected to be in the range [-1, 1] for both x and y,
    with (0, 0) representing the center of the image.

    Args:
        raw_tensor (torch.Tensor): The input tensor containing point data.
        image_size (int): The size (width and height) of the output image.
        flag (str): Indicates the type of tensor ('input', 'target', or 'output').
        size (int, optional): The radius of the circles to draw for each point. Defaults to 4.
        color (str, optional): The color of the points. Defaults to 'red'.

    Returns:
        Image.Image: The generated BEV image.
    """
    assert flag in ['input', 'target', 'output'], "The flag should be 'input', 'output', or 'target'."

    # If the flag is 'output', apply sigmoid and threshold at 0.5
    if flag == 'output':
        raw_tensor[:, 0] = torch.sigmoid(raw_tensor[:, 0])
        raw_tensor[:, 0] = (raw_tensor[:, 0] > 0.5).float()
    
    if flag == 'input':
        raw_tensor[raw_tensor[:, 0] == -1] = 0

    # Remove rows where index 0 is 0
    raw_tensor = raw_tensor[raw_tensor[:, 0] != 0]

    # Extract x and y coordinates based on the flag
    x = raw_tensor[:, 1].numpy()
    y = raw_tensor[:, 2].numpy()
    labels = raw_tensor[:, 0].numpy()

    # Map coordinates from [-1, 1] to image pixel coordinates [0, image_size - 1]
    # Since (0,0) is at the center of the image, we shift and scale accordingly
    xi = ((x + 1) / 2) * (image_size - 1)
    yi = ((1 - y) / 2) * (image_size - 1)  # Invert y-axis to match image coordinate system

    # Create an empty image with white background
    image = Image.new('RGB', (image_size, image_size), 'white')
    draw = ImageDraw.Draw(image)

    # Draw filled circles at each point
    for x_pixel, y_pixel, label in zip(xi, yi, labels):
        # Ensure coordinates are within image bounds
        x_pixel = max(0, min(int(round(x_pixel)), image_size - 1))
        y_pixel = max(0, min(int(round(y_pixel)), image_size - 1))
        draw.ellipse(
            [(x_pixel - size, y_pixel - size), (x_pixel + size, y_pixel + size)],
            fill=label_to_color[int(label)]
        )
    
    if save_path:
        image.save(save_path)

    return image

def create_bev_from_bevtensor(
    bev_tensor: torch.Tensor,
    color: str = 'green'
) -> Image.Image:
    """
    Converts a BEV tensor to an image by mapping tensor values to pixel colors.
    """
    # Convert the tensor to a NumPy array
    bev_array = bev_tensor.numpy()

    # Sum over channels if the tensor has multiple channels
    if bev_array.ndim == 3:
        bev_array = bev_array.sum(axis=0)

    # Create a mask of where the tensor has values greater than zero
    mask = bev_array > 0

    # Define colors
    color_rgb = ImageColor.getrgb(color)
    background_rgb = (255, 255, 255)  # White background

    # Initialize an RGB image array with the background color
    height, width = bev_array.shape
    image_array = np.full((height, width, 3), background_rgb, dtype=np.uint8)

    # Set pixel colors based on the mask
    image_array[mask] = color_rgb

    # Convert the NumPy array to a PIL image
    image = Image.fromarray(image_array)
    return image