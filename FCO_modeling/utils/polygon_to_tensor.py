import torch
from shapely.geometry import Polygon
import time
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Optional
import sys
from utils.raytracing_utils.raytracing import create_vehicle_polygon
import torch
import PIL

def world_to_pixel(vx, vy, x_min, x_max, y_min, y_max, width, height):
    vx_scaled = (vx - x_min) / (x_max - x_min) * (width - 1)
    vy_scaled = (vy - y_min) / (y_max - y_min) * (height - 1)
    return vx_scaled, vy_scaled

def point_in_polygons(xv, yv, vx_batch, vy_batch, valid_vertices, valid_polygons):
    """
    xv, yv: Tensors of shape (H, W)
    vx_batch, vy_batch: Tensors of shape (num_groups, P_max, K_max)
    valid_vertices: Tensor of shape (num_groups, P_max, K_max), boolean
    valid_polygons: Tensor of shape (num_groups, P_max), boolean
    """
    num_groups, P_max, K_max = vx_batch.shape

    # Shift vertices to get edges
    vx_next = torch.roll(vx_batch, shifts=-1, dims=2)
    vy_next = torch.roll(vy_batch, shifts=-1, dims=2)

    # Shift valid_vertices to get valid edges
    valid_vertices_next = torch.roll(valid_vertices, shifts=-1, dims=2)
    # Edge is valid if both current and next vertices are valid
    edge_valid = valid_vertices & valid_vertices_next

    # Prepare tensors for vectorized computation
    # Unsqueeze to match dimensions for broadcasting
    vx = vx_batch.unsqueeze(2).unsqueeze(2)       # Shape: (num_groups, P_max, 1, 1, K_max)
    vy = vy_batch.unsqueeze(2).unsqueeze(2)
    vx_next = vx_next.unsqueeze(2).unsqueeze(2)
    vy_next = vy_next.unsqueeze(2).unsqueeze(2)
    edge_valid = edge_valid.unsqueeze(2).unsqueeze(2)  # Shape: (num_groups, P_max, 1, 1, K_max)

    # Unsqueeze xv and yv to match dimensions
    xv = xv.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # Shape: (1, 1, H, W, 1)
    yv = yv.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # Shape: (1, 1, H, W, 1)

    # Compute conditions for all edges simultaneously
    cond1 = ((vy <= yv) & (vy_next > yv)) | ((vy_next <= yv) & (vy > yv))
    cond1 = cond1 & edge_valid  # Mask invalid edges

    slope = (vx_next - vx) / (vy_next - vy + 1e-6)
    cond2 = xv < slope * (yv - vy) + vx

    # Compute crossings
    crossings = (cond1 & cond2).sum(dim=-1) % 2  # Sum over K_max
    inside_polygons = crossings > 0  # Shape: (num_groups, P_max, H, W)

    # Mask invalid polygons
    valid_polygons = valid_polygons.unsqueeze(2).unsqueeze(2)  # Shape: (num_groups, P_max, 1, 1)
    inside_polygons = inside_polygons & valid_polygons  # Mask invalid polygons

    # Compute inside mask for each group
    inside = inside_polygons.any(dim=1)  # OR over P_max dimension
    # inside: shape (num_groups, H, W)

    return inside  # Shape: (num_groups, H, W)

def polygons_to_tensor(polygons: List[List[Polygon]], height: int, width: int, x_min: float, x_max: float, y_min: float, y_max: float):
    """
    polygons: List of lists of shapely Polygons. Each sublist represents a group.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create grid coordinates
    xv, yv = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing='xy'
    )

    t = time.time()
    num_groups = len(polygons)
    num_polygons = [len(group) for group in polygons]  # Number of polygons in each group
    P_max = max(num_polygons)  # Maximum number of polygons in any group

    # Get the number of vertices for each polygon in each group
    num_vertices = []
    for group in polygons:
        group_num_vertices = [len(polygon.exterior.coords.xy[0]) for polygon in group]
        num_vertices.append(group_num_vertices)

    K_max = max([max(group_num_vertices) for group_num_vertices in num_vertices])  # Max vertices

    # Initialize tensors
    vx_padded = torch.zeros((num_groups, P_max, K_max), device=device)
    vy_padded = torch.zeros((num_groups, P_max, K_max), device=device)
    valid_vertices = torch.zeros((num_groups, P_max, K_max), dtype=torch.bool, device=device)
    valid_polygons = torch.zeros((num_groups, P_max), dtype=torch.bool, device=device)

    # Fill tensors
    for i, group in enumerate(polygons):
        num_p = len(group)
        valid_polygons[i, :num_p] = True
        for j, polygon in enumerate(group):
            vx, vy = polygon.exterior.coords.xy
            K = len(vx)
            vx_padded[i, j, :K] = torch.tensor(vx, device=device)
            vy_padded[i, j, :K] = torch.tensor(vy, device=device)
            valid_vertices[i, j, :K] = True

    # Scale the vertex coordinates
    vx_scaled, vy_scaled = world_to_pixel(vx_padded, vy_padded, x_min, x_max, y_min, y_max, width, height)

    # Compute inside masks
    mask = point_in_polygons(xv, yv, vx_scaled, vy_scaled, valid_vertices, valid_polygons)
    # mask: shape (num_groups, height, width)

    # print(f'polygon_to_tensor: point_in_polygons took {time.time() - t} seconds for {len(polygons)} groups')

    # Save images for debugging
    # to_pil = transforms.ToPILImage()
    # mask_tensor = mask.float()
    # for i in range(mask_tensor.shape[0]):
    #     mask_image = mask_tensor[i]  # Shape: (height, width)
    #     pil_image = to_pil(mask_image.cpu())
    #     pil_image.save(f"output_image_group_{i}.png")

    return mask.float()  # Shape: (num_groups, height, width)

def create_bev_tensor(
    building_polygons: Optional[Dict[str, Polygon]], 
    vehicle_infos: Dict[str, Dict], 
    vehicle_representation: str, 
    image_size: int, 
    x_offset: int, 
    y_offset: int,
    rotation_angle: int = 0,
    radius_covered: int = 50
) -> torch.Tensor:
    assert vehicle_representation in ['box'], "Invalid vehicle representation"

    # Create the vehicle polygons
    vehicle_polygons = {}
    for v in vehicle_infos:
        polygon = create_vehicle_polygon(
            v,
            angle=vehicle_infos[v]['angle'],
            x=vehicle_infos[v]['position'][0],
            y=vehicle_infos[v]['position'][1],
            width=vehicle_infos[v]['width'],
            length=vehicle_infos[v]['length'],
            representation=vehicle_representation
        )
        vehicle_polygons[v] = polygon

    # Merge all polygons
    all_polygons = {**building_polygons, **vehicle_polygons}

    # Convert the polygons to tensor
    return polygons_to_tensor(
        [list(all_polygons.values())],
        height=image_size,
        width=image_size,
        x_min=x_offset - radius_covered,
        x_max=x_offset + radius_covered,
        y_min=y_offset - radius_covered,
        y_max=y_offset + radius_covered
    )


