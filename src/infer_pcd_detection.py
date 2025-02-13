"""
推理 (inference: 简写为 infer 只为和 train 长度一样)目标检测模型
专门针对 pcd 写的
"""

import os

from diffusion_utils import init_dataloader
from modules.detection_unet_2d_condition import DetectionUNet2DConditionModel
from modules.prepare_processpr import PrepareProcessor
import torch
from loguru import logger

from detection_utils import load_detection_modules, init_detection_modules, caluclate_tp_fp, eval_final_results
from opencood.visualization import simple_vis


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    args.pcd_unet_file = os.path.expanduser(args.pcd_unet_file)

    infer_dataloader, infer_dataset = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf, True)

    """特征提取网络"""
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    pcd_unet = DetectionUNet2DConditionModel(cross_attention_dim=1024)
    pcd_unet_checkpoint = torch.load(args.pcd_unet_file, weights_only=False)
    pcd_unet.load_state_dict(pcd_unet_checkpoint["unet"])

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
    detection_head, upsample_layer, _, _, _ = init_detection_modules(args)
    # 推理的时候一定要有模型权重
    load_detection_modules(args.resume_file_det, detection_head)

    """设备选择, 模型转移以及 train 不 train"""
    device = torch.device("cuda:0")
    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)
    detection_head.train()
    detection_head.to(device)

    # Create the dictionary for evaluation
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    for step, batch in enumerate(infer_dataloader):
        with torch.no_grad():
            """diffusion 部分"""
            latents = prepare_processor.get_latents(batch["pcd"].to(device, dtype=weight_dtype))
            noise = torch.randn_like(latents)  # 训练 `prepare_processor.num_train_timesteps` 前, 计算出 noise 的形状
            bsz = latents.shape[0]
            # TODO: 这里训练 detection 的时候依然随机选择一个噪声是否依旧合理
            timesteps = prepare_processor.generate_timestep(bsz, device).long()
            encoder_hidden_states = prepare_processor.text_encoder(batch["pcd_inputs_ids"].to(device), return_dict=False)[0]
            noisy_latents = prepare_processor.add_noise(latents, noise, timesteps)
            internal_sample = {}  # TODO: 如果显存不够用的话需要从 cuda 转移到 cpu 上, 在 forward 里面修改
            model_pred = pcd_unet(noisy_latents, timesteps, encoder_hidden_states, internal_sample=internal_sample)[0]

            """目标检测部分"""
            pcd_feature = internal_sample["after_upsample_block_3"]
            feature = upsample_layer(pcd_feature.to(device, dtype=torch.float32))
            cls_pred, reg_pred, dir_pred = detection_head(feature)

            """推理"""
            # fmt: off
            pred_box_tensor, pred_score = infer_dataset.postprocessor.postprocess(
                infer_dataset.anchor_boxes_tensor.to(device),
                cls_pred.to(device), reg_pred.to(device), dir_pred.to(device)
            )
            # fmt: on
            gt_box_tensor = batch["gt_bbx_list"][0]
            caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor.to(device), result_stat, 0.3)
            caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor.to(device), result_stat, 0.5)
            caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor.to(device), result_stat, 0.7)

            if (step % args.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                logger.info(f"对 {step} 的结果进行可视化")
                vis_save_path_root = os.path.join(args.output_dir, "visualize")

                vis_save_path = os.path.join(vis_save_path_root, f"step_{step:05d}.png")
                infer_result = {
                    "pred_box_tensor": pred_box_tensor,
                    "gt_box_tensor": batch["gt_bbx_list"][0],
                    "score_tensor": pred_score,
                }
                simple_vis.visualize(
                    infer_result, batch["origin_lidar_list"][0], args.cav_lidar_range, vis_save_path, method="bev"
                )
    eval_final_results(result_stat, args.output_dir)
    logger.success(f"推理结果保存在 {args.output_dir}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    args.pcd_unet_file = "~/Desktop/logs/pcd_diffusion_2025_02_12/checkpoint-10-pcd.pth"
    args.resume_file_det = "~/Desktop/logs/pcd_detection_2025_02_12/checkpoint-9-det.pth"
    args.output_dir = "~/Desktop/logs/pcd_detection_2025_02_12"
    args.save_vis_interval = 40  # 从 heal 里面抄过来的
    args.batch_size = 1
    main(args)
