"""
这个文件用于展示, 将点云投影为 BEV 图后, 将 BEV 结果进行可视化
"""

from diffusion_utils import init_dataloader
import numpy as np
import matplotlib.pylab as plt
import os
import einops


def main(args):
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

    """
    2025-02-08 17:31 看上去投影效果挺好的啊
    """
    for step, batch in enumerate(train_dataloader):
        assert batch["pcd"].shape[0] == 1, "只支持 batch_size 为 1 的情况"
        bev = batch["pcd"][0].numpy().astype(np.float32)
        bev = einops.rearrange(bev, "c h w -> h w c")
        # bev = bev * 255
        save_path = os.path.join(args.output_dir, str(step))
        plt.axis("off")
        plt.imshow(bev)
        plt.savefig(save_path)
        print(f"完成对 {step} 的可视化, 对应的路径为 {batch['file_path_list'][0]}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_diffusion.yaml"))
    args.batch_size = 1
    args.output_dir = os.path.expanduser("~/Desktop/logs/pcd_diffusion_2025_02_17/老师任务")
    main(args)
