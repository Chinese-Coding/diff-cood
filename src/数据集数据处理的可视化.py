import os

from loguru import logger

from opencood.visualization.simple_vis import visualize
from src.diffusion_utils import init_dataloader


def main(args):
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

    for step, batch in enumerate(train_dataloader):
        logger.info(f"对 {step} 数据进行可视化")
        save_path = os.path.join(args.output_dir, f"{step}.png")
        infer_result = {"gt_box_tensor": batch["gt_bbx_list"][0]}
        visualize(infer_result, batch["origin_lidar_list"][0], args.cav_lidar_range, save_path, method="bev")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    args.output_dir = "~/Desktop/logs/vis_2025_02_12"
    args.batch_size = 1
    main(args)
