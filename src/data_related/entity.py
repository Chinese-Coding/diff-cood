from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from omegaconf import DictConfig, ListConfig
from pydantic import BaseModel, ConfigDict, SkipValidation


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_data: SkipValidation[List[Image.Image]]
    lidar_np: np.ndarray[np.float64]

    # 新添加的字段
    cav_info: DictConfig | ListConfig  # 同时刻下的 yaml 文件里面的各种信息
    bev_img: np.ndarray  # 同时刻下, 对应的 bev 图像, 用于推理阶段的目标检测


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
