import os
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from data_related.entity import CAVData, LiftSplatShootParams, ObjectBbxData, PFTimestampData
from opencood.data_utils.post_processor.diff_post_processor import DiffPostProcessor
from opencood.utils.camera_utils import img_to_tensor  # 如果以后添加对深度图的处理, 这个函数会用到, 因此先不删除
from opencood.utils.camera_utils import img_transform, normalize_img, sample_augmentation
from opencood.utils.transformation_utils import x1_to_x2

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
    def __init__(self, args):
        self.mode = args.mode  # 模式指的是训练 diffusion 还是 detection
        logger.success(f"从 {args.root_dir} 中加载数据")
        self.scenario_folders: List[Path] = sorted(folder for folder in Path(args.root_dir).iterdir() if folder.is_dir())

        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database: List[Dict[str, Dict[str, PFTimestampData]]] = []
        self.flattened_database = []
        # 输入图片的 0, 1, 2, 3 序号照片的提示词 (从 0 ~ 3: 前, 左, 右, 后)
        self.img_captions = [""]
        self.pcd_captions = [""]

        if self.mode == "detection":
            self.postprocessor = DiffPostProcessor(args.postprocess_args)
            self.anchor_boxes = self.postprocessor.generate_anchor_boxes()
            self.anchor_boxes_tensor = torch.tensor(self.anchor_boxes)

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
            lidar_np=_pcd_to_np(pathes.lidar, False),
            bev_img=cv2.imread(pathes.bev),
            origin_lidar=_pcd_to_np(pathes.lidar),
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
        """增加了一个 `mode` 参数, 这个函数里面为了增加了很多对于这个变量的判断,"""
        camera_data, lidar_np, img_inputs_ids, pcd_inputs_ids = [], [], [], []
        batch_lss_params = []

        for batch in batches:
            camera_data.append(torch.stack(self.img_transform(batch.camera_data)))
            lidar_np.append(self.pcd_transform(batch.lidar_np))
            img_inputs_ids.append(self._get_inputs_ids(self.img_captions))
            pcd_inputs_ids.append(self._get_inputs_ids(self.pcd_captions))
            batch_lss_params.append(self.get_lift_splat_shoot_inputs(self.data_aug_conf, batch, False))

        ret = {
            "img": torch.stack(camera_data),
            "pcd": torch.stack(lidar_np),
            "img_inputs_ids": torch.stack(img_inputs_ids),
            "pcd_inputs_ids": torch.stack(pcd_inputs_ids),
            "lss_params": LiftSplatShootParams.collate_fn(batch_lss_params),
        }

        if self.mode == "detection":
            pos_equal_one_list, neg_equal_one_list, targets_list, gt_bbx_list = [], [], [], []
            origin_lidar_list = []
            for batch in batches:
                # 目标检测所需的参数
                object_np, mask, object_ids = self.postprocessor.generate_object_center_lidar(
                    batch, batch.cav_info["lidar_pose"]
                )
                pos_equal_one, neg_equal_one, targets = self.postprocessor.generate_label(
                    object_np, self.anchor_boxes, mask, return_dict=False
                )
                object_bbx_data = ObjectBbxData(object_bbx_center=object_np, object_bbx_mask=mask, object_ids=object_ids)
                gt_bbx = self.postprocessor.generate_gt_bbx(object_bbx_data)

                pos_equal_one_list.append(torch.tensor(pos_equal_one))
                neg_equal_one_list.append(torch.tensor(neg_equal_one))
                targets_list.append(torch.tensor(targets))
                gt_bbx_list.append(torch.tensor(gt_bbx))
                origin_lidar_list.append(torch.tensor(batch.origin_lidar))

            ret["pos_equal_one"] = torch.stack(pos_equal_one_list)
            ret["neg_equal_one"] = torch.stack(neg_equal_one_list)
            ret["targets"] = torch.stack(targets_list)
            # 每个场景下 gt_bbx 的大小不同, 没法直接对 gt_bbx 做 stack 操作, 所以只能用 list 来进行存储
            # 一般来说 infer 的时候 batch size 是 1, 所以使用 torch.stack(gt_bbx_list) 的时候不会报错
            ret["gt_bbx_list"] = gt_bbx_list  # infer 的时候会用到
            ret["origin_lidar_list"] = origin_lidar_list
        return ret

    def set_transform(self, img_transform, pcd_transform):
        self.img_transform = img_transform
        self.pcd_transform = pcd_transform

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_data_aug_conf(self, data_aug_conf):
        """为 lls 而设置的参数"""
        self.data_aug_conf = data_aug_conf

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

    @staticmethod
    def get_ext_int(cav_info: Dict, camera_id: int):
        """
        TODO: 这段代码是直接从 HEAL 那边搬过来的, 需要搞明白这部分代码到底在干什么 (ps: HEAL 的代码很复杂, 改起来很困难)
        """
        camera_coords = np.array(cav_info[f"camera{camera_id}"]["cords"]).astype(np.float32)
        # TODO: 这里使用的是 `lidar_pose` 而并非 `lidar_pose_clean`
        camera_to_lidar = x1_to_x2(camera_coords, cav_info["lidar_pose"]).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
        )  # UE4 coord to opencv coord
        camera_intrinsic = np.array(cav_info[f"camera{camera_id}"]["intrinsic"]).astype(np.float32)
        return camera_to_lidar, camera_intrinsic

    def get_lift_splat_shoot_inputs(self, data_aug_conf, cav_data: CAVData, return_tuple=True):
        imgs, rots, trans, intrins, post_rots, post_trans = [], [], [], [], [], []  # lift splat shoot 需要的参数
        extrinsics = []  # 不知道哪里需要的参数
        for i, img in enumerate(cav_data.camera_data):
            camera_to_lidar, camera_intrinsic = self.get_ext_int(cav_data.cav_info, i)

            intrin = torch.from_numpy(camera_intrinsic)
            rot = torch.from_numpy(camera_to_lidar[:3, :3])  # R_wc, we consider world-coord is the lidar-coord
            tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            img_src = [img]

            # TODO: 增加对深度图的处理
            # if self.load_depth_file:
            #     depth_img = selected_cav_base["depth_data"][idx]
            #     img_src.append(depth_img)
            # else:
            #     depth_img = None
            # 因为 `lift splat shoot` 使用了预训练模型, 所以不需要数据增强
            resize, resize_dims, crop, flip, rotate = sample_augmentation(data_aug_conf, False)
            img_src, post_rot2, post_tran2 = img_transform(
                img_src, post_rot, post_tran, resize, resize_dims, crop, flip, rotate
            )
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # decouple RGB and Depth

            img_src[0] = normalize_img(img_src[0])
            # if self.load_depth_file:
            #     img_src[1] = img_to_tensor(img_src[1]) * 255

            imgs.append(torch.cat(img_src, dim=0))
            intrins.append(intrin)
            extrinsics.append(torch.from_numpy(camera_to_lidar))
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        if return_tuple:
            return (
                torch.stack(imgs),
                torch.stack(rots),
                torch.stack(trans),
                torch.stack(intrins),
                torch.stack(post_rots),
                torch.stack(post_trans),
            )
        else:
            return LiftSplatShootParams(
                imgs=torch.stack(imgs),
                rots=torch.stack(rots),
                trans=torch.stack(trans),
                intrins=torch.stack(intrins),
                post_rots=torch.stack(post_rots),
                post_trans=torch.stack(post_trans),
            )
