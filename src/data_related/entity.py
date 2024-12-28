from pydantic import ConfigDict, BaseModel, SkipValidation
from typing import List
import numpy as np
from PIL import Image
from pathlib import Path


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_data: SkipValidation[List[Image.Image]]
    lidar_np: np.ndarray[np.float64]


class PFTimestampData(BaseModel):
    lidar: str
    cameras: List[Path]
