from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, SkipValidation


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_data: SkipValidation[List[Image.Image]]
    lidar_np: np.ndarray[np.float64]  # 存放完整的点云

    # 新添加的字段
    cav_info: Dict  # 同时刻下的 yaml 文件里面的各种信息
    bev_img: np.ndarray  # 同时刻下, 对应的 bev 图像, 用于推理阶段的目标检测
    lidar_splitted: List[np.ndarray]  # 存放分割后的点云, 因为点云的尺寸可能不同, 所以不能堆叠在一起


class PFTimestampData(BaseModel):
    lidar: str
    cameras: List[Path]

    # 新添加字段
    yaml: Path
    bev: Path
    lidar_splitted: List[str]


class ObjectBbxData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    object_bbx_center: np.ndarray
    object_bbx_mask: np.ndarray
    object_ids: List[int]
