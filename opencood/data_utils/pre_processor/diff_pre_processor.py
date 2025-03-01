import os
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer
import torch


def _get_resolution(cav_lidar_range, ratio=0.1):
    L1, W1, _, L2, W2, _ = cav_lidar_range
    return int((L2 - L1) / ratio), int((W2 - W1) / ratio)


class DiffPreProcessor:
    def __init__(self, preprocessor_args, train=True):
        self.cav_lidar_range = preprocessor_args.cav_lidar_range
        self.ratio = preprocessor_args.ratio
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.expanduser(preprocessor_args.pretrained_model),
            subfolder="tokenizer",
            revision=preprocessor_args.revision,
            use_fast=False,
        )

    def preprocess(self, pcd: np.ndarray):
        if not isinstance(pcd, list):  # 确保 pcd 是一个列表，统一处理
            pcd = [pcd]

        bev_maps, dep_maps = [], []
        for points in pcd:
            L1, W1, H1, L2, W2, H2 = self.cav_lidar_range
            img_row, img_col = _get_resolution(self.cav_lidar_range, self.ratio)
            bev_map = np.zeros((img_row, img_col))
            bev_origin = np.array([L1, W1, H1]).reshape(1, -1)
            # (N, 3)
            indices = ((points[:, :3] - bev_origin) / self.ratio).astype(int)
            mask = np.logical_and(indices[:, 0] > 0, indices[:, 0] < img_row)
            mask = np.logical_and(mask, np.logical_and(indices[:, 1] > 0, indices[:, 1] < img_col))

            valid_indices, valid_points = indices[mask], points[mask]
            # 初始化深度图为非常小的值（负无穷），这样可以确保任何有效的深度值都会覆盖
            dep_map = np.full((img_row, img_col), -np.inf)  # 使用一个非常大的数初始化
            # 向量化更新深度图，使用 np.maximum.at 直接更新深度图，选择每个像素的最大深度
            np.maximum.at(dep_map, (valid_indices[:, 0], valid_indices[:, 1]), valid_points[:, 2])
            dep_map = np.nan_to_num(dep_map, nan=0.0, posinf=0.0, neginf=0.0)

            indices = indices[mask, :]
            bev_map[indices[:, 0], indices[:, 1]] = 1

            # 对 BEV 图像做旋转和翻转操作，确保方向一致
            dep_map = np.rot90(dep_map)
            dep_map = np.flip(dep_map, axis=0)
            dep_map = np.repeat(dep_map[np.newaxis, :, :], 3, axis=0)

            bev_map = np.rot90(bev_map)
            bev_map = np.flip(bev_map, axis=0)
            bev_map = np.repeat(bev_map[np.newaxis, :, :], 3, axis=0)

            dep_maps.append(dep_map)
            bev_maps.append(bev_map)
        if len(pcd) == 1:
            bev_maps, dep_maps, pcd = bev_maps[0], dep_maps[0], pcd[0]
        return {"bev_maps": bev_maps, "dep_maps": dep_maps, "origin_lidar": pcd, "pcd_inputs_ids": self._get_inputs_ids("")}

    def _get_inputs_ids(self, captions):
        return self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def collate_batch(batch: Dict[str, List[np.ndarray | torch.Tensor]]):
        bev_maps = batch["bev_maps"]
        dep_maps = batch["dep_maps"]
        origin_lidar_list = batch["origin_lidar"]
        pcd_inputs_ids = batch["pcd_inputs_ids"]

        return {
            "bev_maps": torch.tensor(np.stack(bev_maps)),
            "dep_maps": torch.tensor(np.stack(dep_maps)),
            "origin_lidar_list": [torch.tensor(origin_lidar) for origin_lidar in origin_lidar_list],
            "pcd_inputs_ids": torch.stack(pcd_inputs_ids),
        }
