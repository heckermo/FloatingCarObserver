import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from typing import Optional


def visualize_frame(image_tensor: torch.Tensor, vector_tensor: torch.Tensor, radius: int, save_path: Optional[str] = None) -> Optional[Image.Image]:
    channels, height, width = image_tensor.shape
    image_tensor = image_tensor.squeeze().cpu()
    vector_tensor = vector_tensor.cpu()
    # convert the vector tensor to the image coordinate system
    # the origin is at the center of the image and the width and height are the radius in m
    m_to_px = width / (2 * radius) # --> 1m = m_to_px px
    vector_tensor = vector_tensor * m_to_px
    dx, dy = vector_tensor[0].item(), vector_tensor[1].item()
    end_x = width // 2 + dx
    end_y = height // 2 + dy # PIL coordinate systems starts at the top left corner
    # create the PIL image from the tensor
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(image_tensor)
    # add the vector to the image
    draw = ImageDraw.Draw(image_pil)
    draw.line([(width // 2, height // 2), (end_x, end_y)], fill='red', width=2)
    # save the image
    if save_path:
        image_pil.save(save_path)
        return None
    return image_pil