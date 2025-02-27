from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, SkipValidation


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_data: SkipValidation[List[Image.Image]]
    lidar_np: np.ndarray[np.float64]  # 存放完整的点云

    # 新添加的字段
    cav_info: Dict  # 同时刻下的 yaml 文件里面的各种信息
    bev_img: np.ndarray  # 同时刻下, 对应的 bev 图像, 用于推理阶段的目标检测
    origin_lidar: np.ndarray  # 同时刻下, 对应的 lidar 数据, 用于绘制图像

    file_path: Path  # 记录当前数据的加载源 (为了老师今早的任务特地添加的参数)


class PFTimestampData(BaseModel):
    lidar: str
    cameras: List[Path]

    # 新添加字段
    yaml: Path
    bev: Path


class ObjectBbxData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    object_bbx_center: np.ndarray
    object_bbx_mask: np.ndarray
    object_ids: List[int]


class LiftSplatShootParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    imgs: torch.Tensor
    rots: torch.Tensor
    trans: torch.Tensor
    intrins: torch.Tensor
    post_rots: torch.Tensor
    post_trans: torch.Tensor

    @staticmethod
    def collate_fn(batch_lss_inputs: List):
        return LiftSplatShootParams(
            imgs=torch.stack([x.imgs for x in batch_lss_inputs]),
            rots=torch.stack([x.rots for x in batch_lss_inputs]),
            trans=torch.stack([x.trans for x in batch_lss_inputs]),
            intrins=torch.stack([x.intrins for x in batch_lss_inputs]),
            post_rots=torch.stack([x.post_rots for x in batch_lss_inputs]),
            post_trans=torch.stack([x.post_trans for x in batch_lss_inputs]),
        )

    def to(self, device: torch.device):
        for attr, value in self.__dict__.items():
            # 将所有 torch.Tensor 类型的属性搬运到指定设备
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
            # 对于可能包含多个 torch.Tensor 的类型（如 Tuple 或 Dict），递归搬运
            elif isinstance(value, (tuple, list)):
                setattr(self, attr, type(value)(v.to(device) if isinstance(v, torch.Tensor) else v for v in value))
            elif isinstance(value, dict):
                setattr(self, attr, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()})
