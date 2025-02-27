import os

from omegaconf import OmegaConf

from diffusion_utils import init_dataloader
from loguru import logger

from data_related.transform_funs import _project_points_to_bev_depth_map

import einops
import numpy as np
import matplotlib.pylab as plt


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        logger.warning(f"{path} 不存在, 将创建文件夹")
        os.makedirs(path)


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    # 首先检查主文件夹
    if not os.path.exists(args.output_dir):
        logger.warning(f"{args.output_dir} 不存在, 将创建文件夹及其下属的子文件夹")
        os.makedirs(args.output_dir)
    # 创建子文件夹
    create_dir_if_not_exists(os.path.join(args.output_dir, "visualize"))
    logger.success("文件夹路径准备完毕")
    vis_save_path_root = os.path.join(args.output_dir, "visualize")
    OmegaConf.save(args, os.path.join(args.output_dir, "config.yaml"))
    logger.success(f"将配置文件保存到 {args.output_dir} 目录中的 config.yaml 中.")

    train_dataloader, train_dataset = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf, need_dataset=True)

    for i, batch in enumerate(train_dataloader):
        bev = batch["pcd"][0].numpy().astype(np.float32)
        lidar = batch["origin_lidar_list"][0].numpy()
        dep = _project_points_to_bev_depth_map(args.cav_lidar_range, lidar, 0.4)

        bev = einops.rearrange(bev, "c h w -> h w c")
        plt.axis("on")
        plt.imshow(bev)
        plt.savefig(os.path.join(vis_save_path_root, f"{i}bev.png"))

        plt.cla()
        plt.imshow(dep)
        plt.savefig(os.path.join(vis_save_path_root, f"{i}dep.png"))

        logger.info(f"{i}th batch, pcd shape: {lidar.shape}, dep shape: {dep.shape}")
        break


if __name__ == "__main__":
    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    args.output_dir = "~/Desktop/logs/pcd_detection_2025_02_27"
    args.batch_size = 1
    main(args)
