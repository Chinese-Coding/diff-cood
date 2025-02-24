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


def _project_points_to_bev_map(cav_lidar_range, points, ratio=0.1) -> np.ndarray:
    """
    从 opencood 中的 BasePreprocessor 中拿过来的函数 (方便变成函数). 用于将点云投影为 BEV 图
    :return shape: [H, W]
    """
    L1, W1, H1, L2, W2, H2 = cav_lidar_range
    img_row, img_col = _get_resolution(cav_lidar_range, ratio)
    bev_map = np.zeros((img_row, img_col))
    bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
    # (N, 3)
    indices = ((points[:, :3] - bev_origin) / ratio).astype(int)
    mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
    mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0, indices[:, 1] < img_col))
    indices = indices[mask, :]
    bev_map[indices[:, 0], indices[:, 1]] = 1
    # 注意此处对 bev 图做了旋转 90 的处理 (不知道是逆时针还是顺时针), 只知道旋转之后可以和可视化时的 bev 图 (含有 gt 的)方向保持一致
    # 理论上来说这一点应该不影响模型的训练结果
    bev_map = np.rot90(bev_map)
    bev_map = np.flip(bev_map, axis=0)
    return bev_map


def _get_resolution(cav_lidar_range, ratio=0.1):
    L1, W1, _, L2, W2, _ = cav_lidar_range
    return int((L2 - L1) / ratio), int((W2 - W1) / ratio)


def pcd_transform(cav_lidar_range, ratio=0.1):
    def transform(pcd):
        if not isinstance(pcd, list):  # 确保 pcd 是一个列表，统一处理
            pcd = [pcd]

        # TODO: 这里有点问题, 转换完的 bev 图还需要 * 255 吗? 就是达到可视化的效果
        # 加一个 copy() 以解决 numpy 说的负步长的问题, 加了旋转代码后出现的问题, 不清楚具体逻辑
        bev_maps = [
            torch.tensor(_project_points_to_bev_map(cav_lidar_range, i, ratio).copy()).unsqueeze(0).repeat(3, 1, 1) for i in pcd
        ]

        # 如果是单个元素的列表，直接返回
        return bev_maps[0] if len(bev_maps) == 1 else bev_maps

    return transform
