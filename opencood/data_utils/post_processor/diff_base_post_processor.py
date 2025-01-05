# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Template for AnchorGenerator
"""

from typing import Literal, Dict

import cv2
import numpy as np
from omegaconf import DictConfig

from opencood.utils import box_utils


class DiffBasePostProcessor:
    """`diff`: diff-cood 项目专用"""

    def __init__(self, order: Literal["lwh", "hwl"], max_num, cav_lidar_range, gt_range, train=True):
        self.order = order
        self.gt_range = gt_range
        self.max_num = max_num
        self.cav_lidar_range = cav_lidar_range
        self.train = train

    def generate_gt_bbx(self, object_bbx_data, transformation_matrix=np.eye(4)):
        """
        TODO: 证明一下这里使用单位矩阵的正确性
        :param transformation_matrix: 两辆不同车之间的转换矩阵, 因为没有协同, 所以用单位矩阵就可以了吧, 对应的 shape: (4, 4)
        """
        object_bbx_center = object_bbx_data.object_bbx_center
        object_bbx_mask = object_bbx_data.object_bbx_mask
        object_ids = object_bbx_data.object_ids

        object_bbx_center = object_bbx_center[object_bbx_mask == 1]
        # convert center to corner
        object_bbx_corner: np.ndarray = box_utils.boxes_to_corners_3d(object_bbx_center, self.order)
        projected_object_bbx_corner = box_utils.project_box3d(object_bbx_corner, transformation_matrix)
        selected_indices = [object_ids.index(x) for x in set(object_ids)]
        gt_box = projected_object_bbx_corner[selected_indices]
        return box_utils.mask_boxes_outside_range_numpy(gt_box, self.gt_range, order=None)

    def generate_object_center_lidar(self, cav_data: Dict, ref_lidar_pose, enlarge_z=False):
        """使用 lidar 传感器时, 对应的 object center"""
        vehicles = cav_data.cav_info["vehicles"]
        return self._generate_object_center(vehicles, ref_lidar_pose, enlarge_z)

    def generate_object_center_camera(self, cav_data, ref_lidar_pose, enlarge_z=False):
        """获取使用 camera 传感器时, 对应的 object center"""
        vehicles = cav_data.cav_info["vehicles"]
        inf_filter_range = [-1e5, -1e5, -1e5, 1e5, 1e5, 1e5]
        visibility_map = np.asarray(cv2.cvtColor(cav_data.bev_img, cv2.COLOR_BGR2GRAY))
        # TODO: 注意这里使用的是 `lidar_pose` 而并非 `lidar_pose_clean` 也就是说这里暂时不考虑噪声的问题
        ego_lidar_pose = cav_data.cav_info.lidar_pose
        output_dict = box_utils.diff_project_world_objects(vehicles, ego_lidar_pose, inf_filter_range, self.order, enlarge_z)

        updated_vehicles = {
            k: v for k, v in vehicles.items() if k in output_dict and box_utils.box_is_visible(output_dict[k], visibility_map)
        }  # 选出那些在 BEV 中可见的物体 (object, vehicles)
        return self._generate_object_center(updated_vehicles, ref_lidar_pose, enlarge_z)

    def _generate_object_center(self, vehicles: Dict, ref_lidar_pose, enlarge_z=False):
        filter_range = self.cav_lidar_range if self.train else self.gt_range

        output_dict = box_utils.diff_project_world_objects(vehicles, ref_lidar_pose, filter_range, self.order, enlarge_z)

        object_np = np.zeros((self.max_num, 7))
        mask = np.zeros(self.max_num)
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids
