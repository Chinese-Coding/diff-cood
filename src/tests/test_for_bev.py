from data_related.stable_diffusion_dataset import StableDiffusionDataset
from opencood.models.lift_splat_shoot import LiftSplatShoot
from loguru import logger
import torch
from opencood.utils.camera_utils import denormalize_img


def _lift_splat_shoot_test(args):
    dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    dataset.reinitialize()
    device = torch.device("cuda")
    lift_splat_shoot = LiftSplatShoot(args.lift_splat_shoot, device)
    imgs, rots, trans, intrins, post_rots, post_trans = dataset.get_lift_splat_shoot_inputs(
        args.lift_splat_shoot.data_aug_conf, dataset[0]
    )
    imgs, rots, trans, intrins, post_rots, post_trans = (
        imgs.unsqueeze(0).to(device),
        rots.unsqueeze(0).to(device),
        trans.unsqueeze(0).to(device),
        intrins.unsqueeze(0).to(device),
        post_rots.unsqueeze(0).to(device),
        post_trans.unsqueeze(0).to(device),
    )

    bev = lift_splat_shoot(imgs, rots, trans, intrins, post_rots, post_trans)
    bev.sigmoid().cpu()
    bev_img = denormalize_img(bev[0].cpu())
    bev_img.save("bev_img.png")
    logger.success(bev.shape)  # torch.Size([1, 128, 256, 256])


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/config.yaml"))

    _lift_splat_shoot_test(args)
