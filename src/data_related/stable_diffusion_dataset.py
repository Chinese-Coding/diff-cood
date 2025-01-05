import os
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from PIL import Image
from loguru import logger
from torch.utils.data import Dataset

from data_related.entity import CAVData, PFTimestampData

loader = yaml.Loader
loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
|\\.[0-9_]+(?:[eE][-+][0-9]+)?
|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
|[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _load_yaml(file):
    with open(file, "r") as f:
        return yaml.load(f, Loader=loader)


def _load_camera_data(camera_files: List[Path], preLoad=True):
    return (
        [Image.open(camera_file).copy() for camera_file in camera_files]
        if preLoad
        else [Image.open(camera_file) for camera_file in camera_files]
    )


def _extract_values(d: Dict):
    result = []
    for value in d.values():
        if isinstance(value, dict):
            result.extend(_extract_values(value))  # 如果值是字典，则递归处理
        else:
            result.append(value)  # 否则直接添加值
    return result


def _replace_with_additional(file_path: str):
    """Replace the main folder with 'additional' if file is not found."""
    return (
        file_path.replace("train", "additional/train")
        .replace("validate", "additional/validate")
        .replace("test", "additional/test")
    )


def _get_timestamp_data_path(cav_path: Path, timestamp: str):
    """
    获取某一时间戳下数据的路径 (考虑到灵活性, 这里还是返回元组, 用哪个就取哪个)

    :param cav_path 汽车所在路径
    :param timestamp 时间戳
    """
    yaml_file, lidar_file = cav_path / f"{timestamp}.yaml", os.path.join(cav_path, f"{timestamp}.pcd")
    camera_files = [cav_path / f"{timestamp}_camera{i}.png" for i in range(4)]
    cav_path_dpt = Path(str(cav_path).replace("OPV2V", "OPV2V_Hetero"))
    depth_files = [cav_path_dpt / f"{timestamp}_depth{i}.png" for i in range(4)]
    cav_path_bev = Path(_replace_with_additional(str(cav_path))) / f"{timestamp}_bev_visibility.png"
    lidar_splitted_files = [os.path.join(cav_path, f"{timestamp}_camera{i}.pcd") for i in range(4)]
    return yaml_file, lidar_file, camera_files, depth_files, cav_path_bev, lidar_splitted_files


def _pcd_to_np(pcd_file: str, need_color=True):
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    if not need_color:
        return xyz
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float64)  # 这里改成 np.float64 为了方便后续计算的统一


class StableDiffusionDataset(Dataset):
    def __init__(self, root_dir):
        logger.success(f"从 {root_dir} 中加载数据")
        self.scenario_folders: List[Path] = sorted(folder for folder in Path(root_dir).iterdir() if folder.is_dir())

        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database: List[Dict[str, Dict[str, PFTimestampData]]] = []
        self.flattened_database = []
        # 输入图片的 0, 1, 2, 3 序号照片的提示词 (从 0 ~ 3: 前, 左, 右, 后)
        self.img_captions = [
            "A front view taken by an overhead camera of a moving vehicle",
            "A left view taken by a camera on top of a moving vehicle",
            "A right view taken by a camera on top of a moving vehicle",
            "A rear view taken by an overhead camera of a moving vehicle",
        ]
        self.pcd_captions = [
            "Bird's-eye view projection of the front view of the point cloud of a moving vehicle scanned with LiDAR",
            "Bird's-eye view projection of the left view of the point cloud image of a moving vehicle scanned by LiDAR",
            "Bird's-eye view projection of the right view of the point cloud image of a moving vehicle scanned by LiDAR",
            "Bird's-eye view projection of the rear view of the point cloud of a moving vehicle scanned with LiDAR",
        ]

    def reinitialize(self):
        # 每次初始化的时候记得清空之前存储的东西 (如果是第一次初始化可能不需要, 但是为了统一写法就不做判断了)
        self.scenario_database.clear()
        # 定义一个新变量用于存储加载数据的方法, 这样写能缩短代码的长度, 其实也

        # loop over all scenarioscount = 0
        for i, scenario_folder in enumerate(self.scenario_folders):
            self.scenario_database.append({})
            # 判断写在 `append` 之后, 因为下面有代码 `self.scenario_database[i][cav_id] = outputs` 要用到 i, 所以无论如何都要 `append`
            # 以免出现 `IndexError: list index out of range` (为了这么点问题, 又多写了那么多行注释)
            if scenario_folder.parts[-1] == "2021_09_09_13_20_58":  # 这个时刻下的数据都只有三个 camera.
                continue
            # 这三个文件夹下的点云还没有分割完
            if scenario_folder.parts[-1] in ["2021_09_09_22_21_11", "2021_09_09_23_21_21", "2021_09_10_12_07_11"]:
                continue
            # at least 1 cav should show up
            # 用三元运算符来简化判断 (使用 sample 函数代替原先的 shuffle 函数, 因为sample函数有返回值写起来比较统一, 不知道应不影响性能)
            cav_list: List[str] = [cav.name for cav in scenario_folder.iterdir() if cav.is_dir()]

            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):
                # save all yaml files to the dictionary
                cav_path = scenario_folder / cav_id
                outputs, _ = self._load_data_paths(cav_path)
                self.scenario_database[i][cav_id] = outputs
        for scenario in self.scenario_database:
            self.flattened_database.extend(_extract_values(scenario))

        logger.success(f"数据总长度: {len(self.flattened_database)}")

    def __getitem__(self, idx):
        """
        Given the index, return the corresponding data.

        :param idx: Index given by dataloader
        :return: The dictionary contains loaded yaml params and lidar data for each cav.
        """
        pathes: PFTimestampData = self.flattened_database[idx]
        return CAVData(
            cav_info=_load_yaml(pathes.yaml),
            camera_data=_load_camera_data(pathes.cameras),
            lidar_np=_pcd_to_np(pathes.lidar, need_color=False),
            bev_img=cv2.imread(pathes.bev),
            lidar_splitted=[_pcd_to_np(file, False) for file in pathes.lidar_splitted],
        )

    def __len__(self):
        return len(self.flattened_database)

    def _get_inputs_ids(self, captions):
        return self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

    def collate_fn(self, batches: List[CAVData]):
        camera_data, lidar_np, img_inputs_ids, pcd_inputs_ids = [], [], [], []
        lidar_splitted_list = []
        for batch in batches:
            camera_data.append(torch.stack(self.img_transform(batch.camera_data)))
            lidar_np.append(self.pcd_transform(batch.lidar_np))
            img_inputs_ids.append(self._get_inputs_ids(self.img_captions))
            pcd_inputs_ids.append(self._get_inputs_ids(self.pcd_captions))
            lidar_splitted_list.append(torch.stack(self.pcd_transform(batch.lidar_splitted)))

        return {
            # camera shape: (batch, 4, 3, W, H), 这个 4 是每个车有四个相机
            "img": torch.stack(camera_data),
            # pcd shape: (batch, 4, 3, W', H'), 4 是分割之后的点云, 3 将点云投影为 BEV 之后通过重复 3 扩展成通道数为 3 的 BEV
            "pcd": torch.stack(lidar_splitted_list),
            "img_inputs_ids": torch.stack(img_inputs_ids),
            "pcd_inputs_ids": torch.stack(pcd_inputs_ids),
        }

    def set_transform(self, img_transform, pcd_transform):
        self.img_transform = img_transform
        self.pcd_transform = pcd_transform

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _load_data_paths(self, cav_path: Path):
        outputs = {}
        yaml_files: List[Path] = sorted(file for file in cav_path.glob("*.yaml") if "additional" not in file.stem)
        # this timestamp is not ready
        yaml_files = [
            x for x in yaml_files
            if not (("2021_08_20_21_10_24" in (
                path_str := str(x)) and "000265" in path_str) or "2021_09_09_13_20_58" in path_str)  # fmt: skip
        ]
        timestamps = [file.stem for file in yaml_files]  # 来自GPT: 把提取 timestamp 函数删掉了 (一行代码完事)

        for timestamp in timestamps:
            # 将加载数据路径的函数, 移到了一个单独的函数中 (如果因为后面的代码还需要 `lidar_file` 我一定会让 `_GetTimestampDataPath` 函数返回一个字典)
            yaml_file, lidar_file, camera_files, depth_files, bev_file, lidar_splitted_files = _get_timestamp_data_path(
                cav_path, timestamp
            )
            pfTimestampData = PFTimestampData(
                yaml=yaml_file, lidar=lidar_file, cameras=camera_files, bev=bev_file, lidar_splitted=lidar_splitted_files
            )
            outputs[timestamp] = pfTimestampData
        return outputs, len(timestamps)
