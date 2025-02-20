import os

import torch
from loguru import logger

from diffusion_utils import init_logging, init_dataloader
from detection_utils import init_detection_modules, load_detection_modules
from modules.detection_unet_2d_condition import DetectionUNet2DConditionModel
from modules.prepare_processpr import PrepareProcessor
from opencood.visualization import simple_vis


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
    # pcd_unet = DetectionUNet2DConditionModel(cross_attention_dim=1024)
    # pcd_unet_checkpoint = torch.load(args.pcd_unet_file, weights_only=False)
    # pcd_unet.load_state_dict(pcd_unet_checkpoint["unet"])

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
    detection_head, upsample_layer, loss_fn, optimizer, lr_scheduler = init_detection_modules(args)
    if "resume_file_det" in args:
        checkpoint = load_detection_modules(args.resume_file_det, detection_head, optimizer, lr_scheduler)
        first_epoch = checkpoint["epoch"] + 1

    """设备选择, 模型转移以及 train 不 train"""
    device = torch.device("cuda:0")
    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)
    detection_head.to(device)
    # 需要显式地将优化器的状态迁移到目标设备 (没想到这么复杂原本以为只要模型移动到目标设备就能正常用了)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    detection_head.train()
    prepare_processor.set_requires_grad_(False)
    pcd_unet.requires_grad_(False)

    for epoch in range(first_epoch, args.train_epoches):
        logger.success(f"第 {epoch} 个 epoch 开始训练")
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

            """目标检测部分"""
            pcd_feature = internal_sample[args.internal_sample_lay_name]
            # feature = upsample_layer(pcd_feature.to(device, dtype=torch.float32))
            # 如果不把这个 feature 扩大呢? (对应的 anchor 生成的数目也要减少)
            feature = pcd_feature.to(device, dtype=torch.float32)
            cls_pred, reg_pred, dir_pred = detection_head(feature)
            # fmt: off
            total_loss = loss_fn(
                cls_pred, reg_pred, dir_pred,
                batch["pos_equal_one"].to(device), batch["neg_equal_one"].to(device), batch["targets"].to(device)
            )
            # fmt: on
            loss_fn.logging(epoch, step, len(train_dataloader), writer)
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            """在训练过车中对结果进行可视化处理 (看看为什么检测头的 loss 在下降, 而 检测效果却不好)"""
            if step % args.save_vis_interval == 0:
                gt_box_tensor = batch["gt_bbx_list"][0]
                pred_box_tensor, pred_score = train_dataset.postprocessor.postprocess(
                    train_dataset.anchor_boxes_tensor.to(device),
                    cls_pred.detach().to(device),
                    reg_pred.detach().to(device),
                    dir_pred.detach().to(device),
                )
                logger.info(f"对 {step} 的结果进行可视化")
                vis_save_path_root = os.path.join(args.output_dir, "visualize")

                vis_save_path = os.path.join(vis_save_path_root, f"step_{step:05d}.png")
                infer_result = {
                    "pred_box_tensor": pred_box_tensor,
                    "gt_box_tensor": gt_box_tensor,
                    "score_tensor": pred_score,
                }
                simple_vis.visualize(
                    infer_result, batch["origin_lidar_list"][0], args.cav_lidar_range, vis_save_path, method="bev"
                )

        if args.save_freq != -1 and epoch % args.save_freq == 0:
            save_dict = {
                "epoch": epoch,
                "detection_head": detection_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            save_path = f"{args.output_dir}/checkpoint-{epoch}-det.pth"
            torch.save(save_dict, save_path)
            logger.success(f"将模型保存在 {save_path}")
    writer.close()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    # 从特征可视化来看, 在数据集上进行过预训练的, 检测效果并不是很好, 不如直接使用 diffusers 提供的预训练权重
    # args.pcd_unet_file = "~/Desktop/logs/pcd_diffusion_2025_02_12/checkpoint-10-pcd.pth"
    args.output_dir = "~/Desktop/logs/pcd_detection_2025_02_19"
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
    args.postprocess_args.anchor_args.feature_stride = 8

    # args.resume_file_det = os.path.expanduser("~/Desktop/logs/pcd_detection_2025_02_17_afternoon/checkpoint-8-det.pth")
    args.train_epoches = 30
    main(args)
