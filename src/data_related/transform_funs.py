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


def _project_points_to_bev_depth_map(cav_lidar_range, points: np.ndarray, ratio=0.1) -> np.ndarray:
    """
    从 BEV 视角投影点云数据生成深度图。
    :param cav_lidar_range: 点云的范围 [L1, W1, H1, L2, W2, H2] (L、W、H分别表示范围的最小和最大值)
    :param points: 点云数据，shape 为 [N, 3]，每个点为 (x, y, z) 坐标
    :param ratio: BEV 分辨率与点云范围之间的缩放比例
    :return: 深度图，shape 为 [H, W]
    """
    L1, W1, H1, L2, W2, H2 = cav_lidar_range
    img_row, img_col = _get_resolution(cav_lidar_range, ratio)  # 获取 BEV 图像的行列数

    # BEV 原点（坐标系原点）设置为 [L1, W1, H1]，可以根据需要调整
    bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
    # 将点云的 (x, y, z) 投影到 BEV 图像坐标系
    indices = ((points[:, :3] - bev_origin) / ratio).astype(int)
    # 生成有效像素的掩码，只选择在图像尺寸范围内的点
    mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
    mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0, indices[:, 1] < img_col))

    valid_indices, valid_points = indices[mask], points[mask]
    # 初始化深度图为非常小的值（负无穷），这样可以确保任何有效的深度值都会覆盖
    depth_map = np.full((img_row, img_col), -np.inf)  # 初始化为负无穷
    # 向量化更新深度图，使用 np.maximum.at 直接更新深度图，选择每个像素的最大深度
    np.maximum.at(depth_map, (valid_indices[:, 0], valid_indices[:, 1]), valid_points[:, 2])

    # 对 BEV 图像做旋转和翻转操作，确保方向一致
    depth_map = np.rot90(depth_map)
    depth_map = np.flip(depth_map, axis=0)

    return depth_map


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
