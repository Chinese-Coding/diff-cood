"""
anchor 在特征层上的可视化, 旨在看看 anchor 在特征层上的真实位置
"""

import os

import torch
from loguru import logger
from matplotlib import pyplot as plt

from diffusion_utils import init_logging, init_dataloader
from detection_utils import init_detection_modules, load_detection_modules
from modules.detection_unet_2d_condition import DetectionUNet2DConditionModel
from modules.prepare_processpr import PrepareProcessor
from opencood.visualization import simple_vis
import einops
import numpy as np

from opencood.visualization.simple_vis import visualize
from opencood.utils import box_utils


def feature_visualize(feature: torch.Tensor, save_dir: str):
    assert feature.dim() == 4 and feature.shape[0] == 1
    feature = feature.squeeze(0)  # shape: (320, 64, 64)
    for i in range(feature.shape[0]):
        logger.info(f"可视化第 {i} 层特征")
        channel = feature[i]
        channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        plt.axis("off")
        plt.title(f"Channel {i}")
        plt.imshow(channel_norm, cmap="viridis")
        plt.savefig(os.path.join(save_dir, f"feature_channel{i}"), transparent=False, dpi=500)
        plt.close()


def feature_visualize_with_anchor(feature: torch.Tensor, anchors: torch.Tensor, save_dir: str):
    assert feature.dim() == 4 and feature.shape[0] == 1
    feature = feature.squeeze(0)
    for i in range(feature.shape[0]):
        logger.info(f"可视化第 {i} 层特征")
        channel = feature[i]
        channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        plt.axis("off")
        plt.title(f"Channel {i}")
        plt.imshow(channel_norm, cmap="viridis")
        logger.info(f"可视化 anchor 的数目: {anchors.shape[0]}")
        for j in range(anchors.shape[0]):
            x1, x2 = anchors[j, 0], anchors[j, 1]
            y1, y2 = anchors[j, 2], anchors[j, 3]
            plt.plot([x1, x2], [y1, y2], color="red", linewidth=1)
        plt.savefig(os.path.join(save_dir, f"feature_channel{i}"), transparent=False, dpi=500)
        plt.clf()
        break
    plt.close()


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    writer = init_logging(args)
    train_dataloader, train_dataset = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf, need_dataset=True)

    """特征提取网络"""
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    pcd_unet = DetectionUNet2DConditionModel.from_pretrained(args.pretrained_model, revision=args.revision, subfolder="unet")

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
    if args.get("gradient_checkpointing", False):
        pcd_unet.enable_gradient_checkpointing()

    first_epoch = 0
    upsample_layer = torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    """设备选择, 模型转移以及 train 不 train"""
    device = torch.device("cuda:0")
    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)

    prepare_processor.set_requires_grad_(False)
    pcd_unet.requires_grad_(False)

    for step, batch in enumerate(train_dataloader):
        """diffusion 部分"""
        latents = prepare_processor.get_latents(batch["pcd"].to(device, dtype=weight_dtype))
        noise = torch.randn_like(latents)  # 训练 `prepare_processor.num_train_timesteps` 前, 计算出 noise 的形状
        bsz = latents.shape[0]

        if args.get("t", None) is not None:
            timesteps = torch.full((bsz,), args.t, device=device).long()
        else:
            timesteps = prepare_processor.generate_timestep(bsz, device).long()

        encoder_hidden_states = prepare_processor.text_encoder(batch["pcd_inputs_ids"].to(device), return_dict=False)[0]
        noisy_latents = prepare_processor.add_noise(latents, noise, timesteps)
        logger.debug(f"{noisy_latents.shape=}, {encoder_hidden_states.shape=}")
        internal_sample = {}  # TODO: 如果显存不够用的话需要从 cuda 转移到 cpu 上, 在 forward 里面修改
        model_pred = pcd_unet(noisy_latents, timesteps, encoder_hidden_states, internal_sample=internal_sample)[0]

        vis_save_path_root = os.path.join(args.output_dir, "visualize")

        bev = batch["pcd"][0].numpy().astype(np.float32)
        logger.info(f"{bev.shape=}")
        bev = einops.rearrange(bev, "c h w -> h w c")
        plt.axis("off")
        plt.imshow(bev)
        plt.savefig(os.path.join(vis_save_path_root, f"{step}_bev"), transparent=False, dpi=500)
        plt.close()

        pcd_feature = internal_sample[args.internal_sample_lay_name]
        feature = upsample_layer(pcd_feature.to(device, dtype=torch.float32))
        feature = feature.detach().cpu()

        # feature_visualize(feature, save_dir=vis_save_path_root)

        infer_result = {"gt_box_tensor": batch["gt_bbx_list"][0]}
        bev = batch["pcd"][0].numpy().astype(np.float32)
        pcd = bev[0]  # 取其中一个维度
        logger.info(f"{pcd.shape=}")
        pcd = bev.reshape(-1, 3)
        ones_array = np.ones((pcd.shape[0], 1))
        pcd = np.hstack((pcd, ones_array))
        save_path1 = os.path.join(vis_save_path_root, f"投影后的点云.png")
        save_path2 = os.path.join(vis_save_path_root, f"未投影前的点云.png")
        visualize(infer_result, torch.from_numpy(pcd), args.cav_lidar_range, save_path1, method="bev")
        visualize(infer_result, batch["origin_lidar_list"][0], args.cav_lidar_range, save_path2, method="bev")

        anchor_boxes = train_dataset.anchor_boxes
        pos_equal_one = batch["pos_equal_one"][0]
        indices = torch.nonzero(pos_equal_one)
        anchor_boxes = anchor_boxes[indices[:, 0], indices[:, 1], indices[:, 2]]
        anchor_boxes = anchor_boxes.reshape(-1, 7)
        anchor_boxes = box_utils.boxes_to_corners_3d(anchor_boxes, order="hwl")
        anchor_boxes = box_utils.box3d_to_2d(anchor_boxes)
        # feature_visualize_with_anchor(feature, anchor_boxes, save_dir=vis_save_path_root)
        break
    writer.close()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    # 从特征可视化来看, 在数据集上进行过预训练的, 检测效果并不是很好, 不如直接使用 diffusers 提供的预训练权重
    # args.pcd_unet_file = "~/Desktop/logs/pcd_diffusion_2025_02_12/checkpoint-10-pcd.pth"
    args.output_dir = "~/Desktop/logs/pcd_detection_2025_02_21"
    args.t = 261
    args.save_freq = 2
    args.batch_size = 1
    args.save_vis_interval = 100
    # 这四个参数有匹配关系, 修改其中一个记得修改另一个
    # internal_sample_lay_name 决定送入检测头的输入维度和每张特征图的大小, 这样就间接影响了生成 anchor 的数量
    # 而 anchor 的数量还受, 投影粒度以及 anchor 的尺寸决定的, 还有 采样步长 `feature_stride`
    args.internal_sample_lay_name = "after_upsample_block_3"
    args.in_channels = 320
    args.ratio = 0.1  # 此时生成的 bev_map 的尺寸为 (1024, 1024)
    args.postprocess_args.ratio = 0.1
    args.postprocess_args.anchor_args.feature_stride = 2

    # args.resume_file_det = os.path.expanduser("~/Desktop/logs/pcd_detection_2025_02_17_afternoon/checkpoint-8-det.pth")
    args.train_epoches = 30
    main(args)
