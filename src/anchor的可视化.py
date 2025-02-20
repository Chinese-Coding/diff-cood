import os
from pathlib import Path

from loguru import logger

from opencood.visualization.simple_vis import visualize
from src.diffusion_utils import init_dataloader
from opencood.utils import box_utils
import torch
import numpy as np


def main(args):
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    train_dataloader, train_dataset = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf, need_dataset=True)

    cavData = train_dataset.getitem_by_yaml_path(Path("/datasets/OPV2V/train/2021_09_09_23_21_21/6862/000931.yaml"))
    batch = train_dataset.collate_fn([cavData])

    # logger.info(f"对 {step} 数据进行可视化, 对应的路径为 {batch['file_path_list'][0]}")
    save_path = os.path.join(args.output_dir, f"{931}.png")
    anchor_boxes = train_dataset.anchor_boxes
    pos_equal_one = batch["pos_equal_one"][0]
    indices = torch.nonzero(pos_equal_one)
    anchor_boxes = anchor_boxes[indices[:, 0], indices[:, 1], indices[:, 2]]
    anchor_boxes = anchor_boxes.reshape(-1, 7)
    anchor_boxes = box_utils.boxes_to_corners_3d(anchor_boxes, order="hwl")
    infer_result = {"gt_box_tensor": batch["gt_bbx_list"][0], "pred_box_tensor": torch.tensor(anchor_boxes)}
    visualize(infer_result, batch["origin_lidar_list"][0], args.cav_lidar_range, save_path, method="bev")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    args.output_dir = "~/Desktop/logs/vis_2025_02_20"
    args.batch_size = 1
    args.ratio = 0.1  # 此时生成的 bev_map 的尺寸为 (1024, 1024)
    args.postprocess_args.ratio = 0.1
    args.postprocess_args.anchor_args.feature_stride = 4

    main(args)
