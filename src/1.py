import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import PIL
import torch
from opencood.hypes_yaml.yaml_utils import LoadYAML, LoadYAMLFromStr
from opencood.utils.camera_utils import LoadCameraData
from opencood.utils.logger import get_logger
from opencood.utils.pcd_utils import pcd_to_np
from PIL import Image
from pydantic import BaseModel, ConfigDict, SkipValidation
from torch import Tensor
from torch.utils.data import Dataset

logger = get_logger()


class CAVData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_data: SkipValidation[List[Image.Image]]
    lidar_np: np.ndarray[np.float64]


class PFTimestampData(BaseModel):
    lidar: str
    cameras: List[Path]


def _GetTimestampDataPath(cavPath: Path, timestamp: str):
    """
    获取某一时间戳下数据的路径

    :param cavPath 汽车所在路径
    :param timestamp 时间戳
    """
    yaml_file, lidar_file = cavPath / f"{timestamp}.yaml", os.path.join(cavPath, f"{timestamp}.pcd")
    camera_files = [cavPath / f"{timestamp}_camera{i}.png" for i in range(4)]
    cavPath = Path(str(cavPath).replace("OPV2V", "OPV2V_Hetero"))
    depth_files = [cavPath / f"{timestamp}_depth{i}.png" for i in range(4)]

    return yaml_file, lidar_file, camera_files, depth_files


def _LoadParams(yamlFile):
    """Load params from YAML (同时将嵌套字典中的列表数据递归转换为 np.ndarray)"""
    if isinstance(yamlFile, Path):
        params = LoadYAML(yamlFile)
    elif isinstance(yamlFile, str):
        params = LoadYAMLFromStr(yamlFile)

    def _ConvertToArray(data):
        match data:
            case dict():
                return {k: _ConvertToArray(v) for k, v in data.items()}
            case list():
                return np.array(data)
            case _:
                return data

    return _ConvertToArray(params)  # 对 params 进行递归处理


def _ExtractValues(d: Dict):
    result = []
    for value in d.values():
        if isinstance(value, dict):
            result.extend(_ExtractValues(value))  # 如果值是字典，则递归处理
        else:
            result.append(value)  # 否则直接添加值
    return result


class StableDiffusionDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        logger.important(f"从 {root_dir} 中加载数据")
        self.scenario_folders: List[Path] = sorted(folder for folder in Path(root_dir).iterdir() if folder.is_dir())

        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database: List[Dict[str, Dict[str, PFTimestampData]]] = []
        self.flattened_database = []

    def SetTransform(self, transform):
        self.transform = transform

    def _Load4DataPaths(self, cav_path: Path):
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
            yaml_file, lidar_file, camera_files, depth_files = _GetTimestampDataPath(cav_path, timestamp)
            pfTimestampData = PFTimestampData(lidar=lidar_file, cameras=camera_files)
            outputs[timestamp] = pfTimestampData
        return outputs, len(timestamps)

    def reinitialize(self):
        # 每次初始化的时候记得清空之前存储的东西 (如果是第一次初始化可能不需要, 但是为了统一写法就不做判断了)
        self.scenario_database.clear()
        # 定义一个新变量用于存储加载数据的方法, 这样写能缩短代码的长度, 其实也

        # loop over all scenarios
        for i, scenario_folder in enumerate(self.scenario_folders):
            self.scenario_database.append({})

            # at least 1 cav should show up
            # 用三元运算符来简化判断 (使用 sample 函数代替原先的 shuffle 函数, 因为sample函数有返回值写起来比较统一, 不知道应不影响性能)
            cav_list: List[str] = [cav.name for cav in scenario_folder.iterdir() if cav.is_dir()]

            # loop over all CAV data
            for j, cav_id in enumerate(cav_list):
                # save all yaml files to the dictionary
                cav_path = scenario_folder / cav_id
                outputs, timestampsLen = self._Load4DataPaths(cav_path)
                self.scenario_database[i][cav_id] = outputs
        for scenario in self.scenario_database:
            self.flattened_database.extend(_ExtractValues(scenario))

        logger.important(f"数据总长度: {len(self.flattened_database)}")

    def __getitem__(self, idx):
        """
        Given the index, return the corresponding data.

        :param idx: Index given by dataloader
        :return: The dictionary contains loaded yaml params and lidar data for each cav.
        """
        pathes = self.flattened_database[idx]
        return CAVData(camera_data=LoadCameraData(pathes.cameras), lidar_np=pcd_to_np(pathes.lidar, need_color=False))

    def __len__(self):
        return len(self.flattened_database)

    def collate_fn(self, batches: List[CAVData]) -> Dict[str, Tensor]:
        camera_data, lidar_np = [], []  # type: List[Tensor], List[np.ndarray]
        for batch in batches:
            camera_data.append(torch.stack(self.transform(batch.camera_data)))
            lidar_np.append(batch.lidar_np)
        return {
            # camera shape: (batch, 4, 3, W, H), 这个 4 是每个车有四个相机
            "camera": torch.stack(camera_data),
            "lidar": lidar_np,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms

    resolution = 512

    augmentations = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform_images(examples: Union[List[PIL.Image.Image], PIL.Image.Image]) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(examples, list):
            images = [augmentations(image.convert("RGB")) for image in examples]
        elif isinstance(examples, PIL.Image.Image):
            images = augmentations(examples.convert("RGB"))
        return images

    dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    dataset.reinitialize()
    dataset.SetTransform(transform_images)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataLoader:
        print(f"batch 的类型为: {type(batch)}")

        break


def pcd_to_np(pcd_file: str, need_color=True):
    """
    Read  pcd and return numpy array.

    Returns
    -------
    pcd : o3d.PointCloud
    PointCloud object, used for visualization
    pcd_np : np.ndarray
    The lidar data in numpy format, shape:(n, 4)

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    if not need_color:
        return xyz
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float64)  # 这里改成 np.float64 为了方便后续计算的统一


import argparse
import os
import sys
from typing import List, Union

import numpy as np
import PIL
import torch
from opencood.diffusion.controlnet.diffusion_feature.capture import Capture
from opencood.diffusion.controlnet.diffusion_feature.dpt_processor import DPTProcessor
from opencood.diffusion.controlnet.diffusion_feature.img_processor import ImageProcessor
from opencood.diffusion.Converter import Converter
from opencood.diffusion.StableDiffusionDataset import StableDiffusionDataset
from opencood.utils.logger import get_logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--root_dir", type=str, default="/datasets/OPV2V/train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=30)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    augmentations = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform_images(examples: Union[List[PIL.Image.Image], PIL.Image.Image]) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(examples, list):
            images = [augmentations(image.convert("RGB")) for image in examples]
        elif isinstance(examples, PIL.Image.Image):
            images = augmentations(examples.convert("RGB"))
        return images

    dataset = StableDiffusionDataset(args.root_dir)
    dataset.reinitialize()
    dataset.SetTransform(transform_images)
    # TODO: 真运行的时候记得修改这个 num_workers
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)
    original_stdout = sys.stdout
    # Image 部分
    logger.important("加载 Img 部分模型")
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        img_capture = Capture(device=torch.device("cuda:0"))
        img_processor = ImageProcessor(img_capture)
        img_optimizer = torch.optim.AdamW(
            img_processor.capturer.model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-08
        )
        img_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(img_optimizer, args.epoch, eta_min=1e-6)
    sys.stdout = original_stdout

    # lidar 部分
    logger.important("加载 Dpt 部分模型")
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        dpt_capture = Capture(device=torch.device("cuda:1"))
        projector = Converter()
        dpt_processor = DPTProcessor(dpt_capture)
        dpt_optimizer = torch.optim.AdamW(
            dpt_processor.capturer.model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-08
        )
        dpt_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dpt_optimizer, args.epoch, eta_min=1e-6)
    sys.stdout = original_stdout
    logger.important("开始训练")
    for i in range(args.epoch):
        progress_bar = tqdm(total=len(data_loader))
        progress_bar.set_description(f"Epoch {i}")
        for batch in data_loader:
            # 输入到模型的图片总数是: batch_size * 4
            camera = batch["camera"]
            batch_size, num_cameras, channels, height, width = camera.shape  # num_cameras 恒定为 4, channels 恒定为 3
            print(f"camera 的 shape: {camera.shape}")
            img = camera.view(-1, channels, height, width)
            img_noise, img_pred_noise = img_processor(img.to(img_capture.device))

            # lidar 部分
            lidar = batch["lidar"]
            assert isinstance(lidar, list) and len(lidar) == 1
            lidar = lidar[0]
            # TODO: 咱们的点云是 4 维的, 参考项目的是三维的, 没法直接用啊
            # TODO: 投影方式需要改变一下
            dpt = projector.proj_pc2dpt(lidar, extrinsic=np.eye(4), intrinsic=np.eye(3), h=height, w=width)
            print(type(dpt))
            _, dpt = dpt_processor.process_given_dpt(dpt)
            dpt = (dpt * 1000.0).astype(
                np.uint16
            )  # 别问, 问就是拿过来的. (最开始他是扩大 1000 倍之后存入磁盘中, 然后需要的时候再读取出来)
            dpt = dpt_processor.control_input(dpt)
            # with SummaryWriter (comment="dpt_processor") as w:
            #     w.add_graph(dpt_processor, (dpt.to(dpt_processor.device),))
            dpt_noise, dpt_pred_noise = dpt_processor(dpt)

            img_loss = torch.nn.functional.mse_loss(img_noise, img_pred_noise)
            dpt_loss = torch.nn.functional.mse_loss(dpt_noise, dpt_pred_noise)

            img_optimizer.zero_grad()
            img_loss.backward()
            img_optimizer.step()
            img_lr_scheduler.step()

            dpt_optimizer.zero_grad()
            dpt_loss.backward()
            dpt_optimizer.step()
            dpt_lr_scheduler.step()

            progress_bar.update(1)
            logs = {
                "img_loss": img_loss.detach().item(),
                "img_lr": img_lr_scheduler.get_last_lr()[0],
                "dpt_loss": dpt_loss.detach().item(),
                "dpt_lr": dpt_lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
        progress_bar.close()


class Converter:
    def __init__(self):
        pass

    def to_harmonic(self, input: np.ndarray):
        """转换成齐次坐标形式"""
        M = input.shape[0]
        input = np.concatenate([input, np.ones([M, 1])], axis=1)
        return input

    def proj_3to2(self, xyz: np.ndarray, intrinsic, extrinsic):
        """
        不懂原理, 直接照抄
        :param xyz: shape: [M, 3]
        :param extrinsic: shape: [3, 3]
        :param intrinsic: shape: [4, 4]
        """
        xyz = self.to_harmonic(xyz)
        xyz = np.linalg.inv(extrinsic) @ xyz.T
        uvd = intrinsic @ xyz[0:3]
        uvd = uvd.T
        uv, d = uvd[:, 0:2] / (uvd[:, -1:] + 1e-5), uvd[:, -1]
        return uv, d

    def proj_pc2dpt(self, point_cloud: np.ndarray, extrinsic, intrinsic, h, w):
        uv, dpt = self.proj_3to2(point_cloud, intrinsic, extrinsic)
        mask_w = (uv[:, 0] < w) & (uv[:, 0] >= 0)
        mask_h = (uv[:, 1] < h) & (uv[:, 1] >= 0)
        # mask mask off the back-project points
        mask_d = dpt > 0.05
        mask = mask_h & mask_w & mask_d
        uv = uv[mask].astype(np.int32)
        dpt = dpt[mask]
        result = np.ones([h, w]) * 10000
        for i in range(uv.shape[0]):
            u, v = uv[i]
            d = dpt[i]
            result[v, u] = min(result[v, u], d)
        result[result > 9999] = 0.0
        return result

    def control_input(self, dpt):
        dpt = torch.from_numpy(dpt.copy()).float() / 255.0
        dpt = torch.stack([dpt for _ in range(1)], dim=0)  # 有意思, 为了增加一个维度直接用一个 None 不就行了吗
        dpt = einops.rearrange(dpt, "b h w c -> b c h w").clone()
        dpt = dpt.to(self.capturer.device)
        return dpt

    def process_given_dpt(self, dpt_backup):
        dpt = deepcopy(dpt_backup)
        # depth normalization -> 0-255 uint8
        dpt = depth_normalize(dpt)
        # dpt as network input
        # TODO: 这里也许可以不用转换成 HWC 格式的形式, 感觉 HWC 这种形式不是很常见啊
        dpt = HWC3(dpt)  # 这里将dpt转换为 HWC 的形式, 因为下面那个函数需要输入的参数是HWC的
        # dpt = cv2.resize(dpt, self.capturer.img_resolution, interpolation=cv2.INTER_LINEAR)
        dpt = resize_image(dpt, self.capturer.img_resolution)
        dpt = np.array(dpt)
        self.H, self.W = dpt.shape[0:2]
        # for visualization
        dpt_backup = dpt_backup[:, :, None].repeat(3, axis=-1).astype(np.float32)
        return dpt_backup, dpt


def depth_normalize(depth):
    depth = depth.astype(np.float64)
    # 这两个操作用于计算深度图的一个有效范围，以避免极端值（如异常噪声）对归一化处理的影响，从而将大部分有效的深度信息用于后续处理。
    # 取出深度图中的第 2 和第 85 百分位的值
    vmin, vmax = np.percentile(depth, 2), np.percentile(depth, 85)
    depth -= vmin  # 首先将所有的深度值减去 vmin，使得最小值变为零。
    depth /= vmax - vmin  # 接着将所有的深度值除以 (vmax - vmin)，使得归一化后的深度值范围在 [0, 1] 之间。
    # 这一操作将深度值进行反转，原本较小的深度值变为较大的值，较大的深度值变为较小的值。这样做的目的是为了使得“近距离”在视觉上更加突出（通常在深度图中，深度越小表示越近）。
    depth = 1.0 - depth
    # 将归一化后的深度值（范围 [0, 1]）映射到 8-bit 图像的值域 [0, 255]。
    # clip(0, 255) 确保输出的像素值不会超出这个范围，最后将其转换为 np.uint8 类型，适合用于图像显示。
    depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
    return depth_image


import os
import random

import cv2
import numpy as np

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), "ckpts")


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    # my adding for kitti process
    if W > 2.5 * H:
        W = int(2.5 * H)
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F : F + H, F : F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def img2mask(img, H, W, low=10, high=90):
    assert img.ndim == 3 or img.ndim == 2
    assert img.dtype == np.uint8

    if img.ndim == 3:
        y = img[:, :, random.randrange(0, img.shape[2])]
    else:
        y = img

    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)

    if random.uniform(0, 1) < 0.5:
        y = 255 - y

    return y < np.percentile(y, random.randrange(low, high))
