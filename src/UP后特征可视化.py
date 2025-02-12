"""
本文件旨在将 diffusion 过程中交换的中间层的特征可视化出来,
用来验证 diffusion 是否真的学到了可靠的东西 (即使 loss 不下降)
"""

import os

import einops
import matplotlib.pylab as plt
import numpy as np
import torch
from loguru import logger
from torch import Tensor

from diffusion_utils import init_dataloader
from modules.layering_unet_2dc_model import LayeringUNet2DCModel
from modules.layering_unet_2dc_model import LayeringUNet2DCParams
from modules.prepare_processpr import PrepareProcessor


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

    # 加载官方库提供的预训练权重
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    pcd_unet: LayeringUNet2DCModel = LayeringUNet2DCModel.from_pretrained(
        args.pretrained_model, subfolder="unet", revision=args.revision
    )

    """加载自己训练的权重 (注意这里只加载 UNet 的权重, 并且一定要设置 resume_file)"""
    resume_file = os.path.expanduser(args.resume_file)
    checkpoint = torch.load(resume_file, weights_only=False)
    pcd_unet.load_state_dict(checkpoint["unet"])

    logger.success(f"从 {args.resume_file} 中加载 pcd 模型")

    """显存优化部分"""
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    weight_dtype = torch.float32
    # fmt: off
    match args.mixed_precision:
        case "fp16": weight_dtype = torch.float16
        case "bf16": weight_dtype = torch.bfloat16
        case "fp32": pass
        case _: logger.error(f"使用了不受支持的 {args.mixed_precision}, 现在默认默认的 dtype: {weight_dtype}")
    # fmt: on
    prepare_processor.set_weight_dtype(weight_dtype)
    if args.enable_xformers_memory_efficient_attention:
        pcd_unet.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        pcd_unet.enable_gradient_checkpointing()

    device = torch.device("cuda:0")
    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)

    """
    开始进行可视化, 可视化只进行一次, 也就是只走一边 epoch
    可视化还有一个问题就是, 是否要对加入的噪声的时刻进行指定?
    """
    for step, batch in enumerate(train_dataloader):
        assert batch["pcd"].shape[0] == 1, "只支持 batch_size 为 1 的情况"
        logger.success(f"开始对第 {step} 张照片进行可视化")
        latents = prepare_processor.get_latents(batch["pcd"].to(device, dtype=weight_dtype))
        noise = torch.randn_like(latents)
        timestep = prepare_processor.generate_timestep().item()
        pcd_noise, pcd_params = prepare_processor.prepare(
            batch["pcd"].to(device, weight_dtype), batch["pcd_inputs_ids"].to(device), False, timestep, noise=noise
        )  # type: Tensor, LayeringUNet2DCParams
        vaed_feature = pcd_params.sample.detach().to(device="cpu", dtype=torch.float32)
        logger.debug(f"vae 后的 feature 的 shape: {vaed_feature.shape}")
        pcd_params = pcd_unet.forward_control(pcd_unet.forward_down(pcd_unet.forward_pre(pcd_params)))

        pcd_params.preserved_up_indices = args.preserved_up_indices  # 相比 diffusion 多了这一步
        pcd_params.to(device)
        pcd_params = pcd_unet.forward_up(pcd_unet.forward_middle(pcd_params))
        pcd_feature = pcd_params.preserved_up_feature

        feature = pcd_feature[3].to(device="cpu", dtype=torch.float32)
        logger.debug(f"获得的 UP 层的最后一层的 Tensor 的 shape: pcd: {feature.shape}")

        # feature = pcd_params.sample.detach().to(device="cpu", dtype=torch.float32)
        # logger.debug(f"获得的中间层 Tensor 的 shape: pcd: {feature.shape}")

        """获取到交换层层特征后就不走后面的层了, 直接进行可视化"""
        bev = batch["pcd"][0].numpy().astype(np.float32)
        bev = einops.rearrange(bev, "c h w -> h w c")
        plt.axis("off")
        plt.imshow(bev)
        plt.savefig(os.path.join(args.output_dir, f"{step}_bev"))
        plt.close()

        feature = feature.squeeze(0)  # shape: (320, 64, 64)
        for i in range(feature.shape[0]):
            logger.info(f"可视化第 {i} 层特征")
            channel = feature[i]
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            plt.axis("off")
            plt.title(f"Channel {i}")
            plt.imshow(channel_norm, cmap="viridis")
            plt.savefig(os.path.join(args.output_dir, f"{step}_feature_channel{i}"))
            plt.close()

        vaed_feature = vaed_feature.squeeze(0)
        for i in range(vaed_feature.shape[0]):
            logger.info(f"可视化第 {i} 层特征")
            channel = vaed_feature[i]
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            plt.axis("off")
            plt.title(f"Channel {i}")
            plt.imshow(channel_norm, cmap="viridis")
            plt.savefig(os.path.join(args.output_dir, f"{step}_vaed_feature_channel{i}"))
            plt.close()

        break


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_diffusion.yaml"))
    args.output_dir = "~/Desktop/logs/pcd_diffusion_2025_02_11"
    args.resume_file = "~/Desktop/logs/pcd_diffusion_2025_01_16/checkpoint-27-pcd.pth"
    args.batch_size = 1
    args.preserved_up_indices = [0, 1, 2, 3]
    main(args)
