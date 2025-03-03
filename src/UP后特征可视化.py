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

from diffusion_utils import init_dataloader
from modules.prepare_processpr import PrepareProcessor
from opencood.visualization.simple_vis import visualize
from src.modules.detection_unet_2d_condition import DetectionUNet2DConditionModel
import copy


def feature_visualize(feature: torch.Tensor, save_dir: str):
    assert feature.dim() == 4 and feature.shape[0] == 1
    feature = feature.squeeze(0)  # shape: (320, 64, 64)
    for i in range(feature.shape[0]):
        logger.info(f"可视化第 {i} 层特征")
        channel = feature[i]
        channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        plt.axis("on")
        plt.title(f"Channel {i}")
        plt.imshow(channel_norm, cmap="viridis")
        plt.savefig(os.path.join(save_dir, f"feature_channel{i}"))
        plt.close()


def feature_visualize_sum_method(feature: torch.Tensor, save_path: str):
    assert feature.dim() == 4 and feature.shape[0] == 1
    feature = feature.squeeze(0)
    feature = feature.sum(0)
    feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
    plt.axis("on")
    plt.imshow(feature, cmap="viridis")
    plt.savefig(save_path)
    plt.show()


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    args.pcd_unet_file = os.path.expanduser(args.pcd_unet_file)

    save_dir = os.path.join(args.output_dir, f"our_weight")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir2 = os.path.join(args.output_dir, f"pretrained")
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

    device = torch.device("cuda:0")

    # 加载官方库提供的预训练权重
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    # pcd_unet_our_weight = DetectionUNet2DConditionModel(cross_attention_dim=1024)

    # pcd_unet_checkpoint = torch.load(args.pcd_unet_file, map_location=device, weights_only=False)
    # pcd_unet_our_weight.load_state_dict(pcd_unet_checkpoint["unet"])

    pcd_unet_pretrained = DetectionUNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", revision=args.revision
    )

    # logger.success(f"从 {args.pcd_unet_file} 中加载 pcd 模型")

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
        # pcd_unet_our_weight.enable_xformers_memory_efficient_attention()
        pcd_unet_pretrained.enable_xformers_memory_efficient_attention()
    if args.get("gradient_checkpointing", False):  # 训练 diffusion 的时候会用到
        # pcd_unet_our_weight.enable_gradient_checkpointing()
        pcd_unet_pretrained.enable_gradient_checkpointing()

    prepare_processor.to(device, weight_dtype)
    # pcd_unet_our_weight.to(device, dtype=weight_dtype)
    pcd_unet_pretrained.to(device, dtype=weight_dtype)
    """
    开始进行可视化, 可视化只进行一次, 也就是只走一边 epoch
    可视化还有一个问题就是, 是否要对加入的噪声的时刻进行指定?
    """
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            """diffusion 部分"""
            latents = prepare_processor.get_latents(batch["pcd"].to(device, dtype=weight_dtype))
            noise = torch.zeros_like(latents)  # 训练 `prepare_processor.num_train_timesteps` 前, 计算出 noise 的形状
            bsz = latents.shape[0]
            # TODO: 这里训练 detection 的时候依然随机选择一个噪声是否依旧合理
            if args.get("t", None) is not None:
                timesteps = torch.full((bsz,), args.t, device=device).long()
            else:
                timesteps = prepare_processor.generate_timestep(bsz, device).long()

            encoder_hidden_states = prepare_processor.text_encoder(batch["pcd_inputs_ids"].to(device), return_dict=False)[0]
            noisy_latents = prepare_processor.add_noise(latents, noise, timesteps)
            # internal_sample = {}  # TODO: 如果显存不够用的话需要从 cuda 转移到 cpu 上, 在 forward 里面修改

            encoder_hidden_states2 = copy.deepcopy(encoder_hidden_states)
            noisy_latents2 = copy.deepcopy(noisy_latents)
            internal_sample2 = {}

            # pcd_unet_our_weight(noisy_latents, timesteps, encoder_hidden_states, internal_sample=internal_sample)
            pcd_unet_pretrained(noisy_latents2, timesteps, encoder_hidden_states2, internal_sample=internal_sample2)
        # pcd_feature = internal_sample[args.internal_sample_lay_name]  # our_weight
        pcd_feature2 = internal_sample2[args.internal_sample_lay_name]  # pretrained
        # feature = pcd_feature.to(device="cpu", dtype=torch.float32)
        feature2 = pcd_feature2.to(device="cpu", dtype=torch.float32)

        """获取到交换层层特征后就不走后面的层了, 直接进行可视化"""
        bev = batch["pcd"][0].numpy().astype(np.float32)
        bev = einops.rearrange(bev, "c h w -> h w c")
        plt.axis("on")
        plt.imshow(bev)
        plt.savefig(os.path.join(args.output_dir, f"{step}_bev"))
        plt.close()

        save_path = os.path.join(args.output_dir, f"{step}_gt.png")
        infer_result = {"gt_box_tensor": batch["gt_bbx_list"][0]}
        visualize(infer_result, batch["origin_lidar_list"][0], args.cav_lidar_range, save_path, method="bev")

        # feature_visualize(feature, os.path.join(save_dir, f"{step}_feature.png"))
        feature_visualize_sum_method(feature2, os.path.join(save_dir2, f"{step}_feature.png"))
        if step >= args.max_step:
            break


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    args.output_dir = "~/Desktop/logs/pcd_diffusion_2025_02_25/特征可视化"
    args.pcd_unet_file = "~/Desktop/logs/pcd_diffusion_2025_02_12/checkpoint-10-pcd.pth"
    args.batch_size = 1
    args.internal_sample_lay_name = "after_upsample_block_3"
    args.t = 261
    args.max_step = 100
    main(args)
