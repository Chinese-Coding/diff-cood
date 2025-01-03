import numpy as np
import torch
from torchvision import transforms


def img_transform():
    augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform(imgs):
        return [augmentations(img.convert("RGB")) for img in imgs]

    return transform


def _project_points_to_bev_map(cav_lidar_range, points, ratio=0.1):
    """从 opencood 中的 BasePreprocessor 中拿过来的函数 (方便变成函数). 用于将点云投影为 BEV 图"""
    L1, W1, H1, L2, W2, H2 = cav_lidar_range
    img_row = int((L2 - L1) / ratio)
    img_col = int((W2 - W1) / ratio)
    bev_map = np.zeros((img_row, img_col))
    bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
    # (N, 3)
    indices = ((points[:, :3] - bev_origin) / ratio).astype(int)
    mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
    mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0, indices[:, 1] < img_col))
    indices = indices[mask, :]
    bev_map[indices[:, 0], indices[:, 1]] = 1
    return bev_map


def pcd_transform(cav_lidar_range):
    def transform(pcd):
        bev_map = _project_points_to_bev_map(cav_lidar_range, pcd)
        bev_map_tensor = torch.tensor(bev_map)
        return bev_map_tensor.unsqueeze(0).repeat(3, 1, 1)  # 返回一个添加了一个维度的 bev_map

    return transform
