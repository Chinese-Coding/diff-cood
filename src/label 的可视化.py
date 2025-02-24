import os
from pathlib import Path

import matplotlib.pylab as plt
import torch
import numpy as np

from opencood.utils import box_utils
from opencood.visualization.simple_vis import visualize
from src.diffusion_utils import init_dataloader
import einops
from diffusion_utils import init_logging, init_dataloader
from detection_utils import init_detection_modules, load_detection_modules
from modules.detection_unet_2d_condition import DetectionUNet2DConditionModel
from modules.prepare_processpr import PrepareProcessor
from opencood.visualization import simple_vis
import matplotlib.pylab as plt
from loguru import logger


def feature_visualize(feature: torch.Tensor, save_dir: str):
    feature = feature.squeeze(0)
    for i in range(feature.shape[0]):
        logger.info(f"可视化第 {i} 层特征")
        channel = feature[i]
        channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        plt.axis("on")
        plt.title(f"Channel {i}")
        plt.imshow(channel_norm, cmap="viridis")
        plt.savefig(os.path.join(save_dir, f"feature_channel{i}"))
        plt.close()


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    # args.pcd_unet_file = os.path.expanduser(args.pcd_unet_file)

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

    """目标检测部分"""
    first_epoch = 0
    detection_head, before_detection_head, loss_fn, optimizer, lr_scheduler = init_detection_modules(args)
    if "resume_file_det" in args:
        checkpoint = load_detection_modules(
            args.resume_file_det, before_detection_head, detection_head, optimizer, lr_scheduler
        )
        first_epoch = checkpoint["epoch"] + 1

    """设备选择, 模型转移以及 train 不 train"""
    device = torch.device("cuda:0")
    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)
    before_detection_head.to(device)
    detection_head.to(device)
    # 需要显式地将优化器的状态迁移到目标设备 (没想到这么复杂原本以为只要模型移动到目标设备就能正常用了)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    detection_head.train()
    prepare_processor.set_requires_grad_(False)
    pcd_unet.requires_grad_(False)

    cavData = train_dataset.getitem_by_yaml_path(Path("/datasets/OPV2V/train/2021_09_09_23_21_21/6862/000931.yaml"))
    batch = train_dataset.collate_fn([cavData])

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

    """目标检测部分"""
    pcd_feature = internal_sample[args.diffusion_args.internal_sample_lay_name]
    # feature_visualize(pcd_feature.cpu()[0], os.path.join(args.output_dir, "feature_visualize_before"))
    feature = before_detection_head(pcd_feature.to(device, dtype=torch.float32))
    # feature_visualize(feature.cpu().detach()[0], os.path.join(args.output_dir, "feature_visualize_after"))
    cls_pred, reg_pred, dir_pred = detection_head(feature)
    # fmt: off
    total_loss = loss_fn(
        cls_pred, reg_pred, dir_pred,
        batch["pos_equal_one"].to(device), batch["neg_equal_one"].to(device), batch["targets"].to(device)
    )

    bev = batch["pcd"][0].numpy().astype(np.float32)
    bev = einops.rearrange(bev, "c h w -> h w c")

    plt.axis("on")
    plt.imshow(bev, cmap="viridis")
    plt.savefig(os.path.join(args.output_dir, "bev.png"))
    plt.close()

    # logger.info(f"对 {step} 数据进行可视化, 对应的路径为 {batch['file_path_list'][0]}")

    anchor_boxes = train_dataset.anchor_boxes
    pos_equal_one = batch["pos_equal_one"][0]
    result = torch.any(pos_equal_one, dim=2).int()
    plt.axis("on")
    plt.imshow(result, cmap="viridis")
    # 反转纵坐标轴
    # plt.gca().invert_yaxis()  # 这将会使得 y 坐标轴反转


    # 设置DPI提高分辨率
    dpi = 1600  # 可以根据需要调整
    fig = plt.gcf()
    fig.set_dpi(dpi)
    count = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i, j] == 1:  # 如果是 True
                # 在图像上标记坐标
                # plt.text(i, j, f"({count})", color="white", fontsize=1, ha="center", va="center")
                plt.text(j, i, f"({count})", color="black", fontsize=1, ha="center", va="center")
                print(f"{count}: {i} {j}")
                count+=1
                if count >= 10:
                    break
        if count >= 10:
            break

    plt.savefig(os.path.join(args.output_dir, "pos_equal_one.png"))
    plt.close()

    indices = torch.nonzero(pos_equal_one)
    anchor_boxes = anchor_boxes[indices[:, 0], indices[:, 1], indices[:, 2]]

    anchor_boxes = anchor_boxes[0][np.newaxis, :]
    anchor_boxes = anchor_boxes.reshape(-1, 7)
    anchor_boxes = box_utils.boxes_to_corners_3d(anchor_boxes, order="hwl")
    infer_result = {"gt_box_tensor": batch["gt_bbx_list"][0], "pred_box_tensor": torch.tensor(anchor_boxes)}
    # infer_result = {"gt_box_tensor": batch["gt_bbx_list"][0]}
    save_path = os.path.join(args.output_dir, "gt.png")
    visualize(infer_result, batch["origin_lidar_list"][0], args.cav_lidar_range, save_path, method="bev")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    args.output_dir = "~/Desktop/logs/vis_2025_02_24"
    args.batch_size = 1
    # args.ratio = 0.1  # 此时生成的 bev_map 的尺寸为 (1024, 1024)
    # args.postprocess_args.ratio = 0.1
    # args.postprocess_args.anchor_args.feature_stride = 4

    main(args)
