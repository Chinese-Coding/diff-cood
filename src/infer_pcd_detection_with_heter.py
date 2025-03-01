"""
推理 (inference: 简写为 infer 只为和 train 长度一样)目标检测模型
专门针对 pcd 写的
"""

import os

from torch.utils.data import DataLoader

from diffusion_utils import init_dataloader
from modules.detection_unet_2d_condition import DetectionUNet2DConditionModel
from modules.prepare_processpr import PrepareProcessor
import torch
from loguru import logger

from detection_utils import load_detection_modules, init_detection_modules, caluclate_tp_fp, eval_final_results
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import simple_vis


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        logger.warning(f"{path} 不存在, 将创建文件夹")
        os.makedirs(path)


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    # 首先检查主文件夹
    if not os.path.exists(args.output_dir):
        logger.warning(f"{args.output_dir} 不存在, 将创建文件夹及其下属的子文件夹")
        os.makedirs(args.output_dir)
    # 创建子文件夹
    create_dir_if_not_exists(os.path.join(args.output_dir, "visualize"))
    logger.success("文件夹路径准备完毕")
    vis_save_path_root = os.path.join(args.output_dir, "visualize")
    OmegaConf.save(args, os.path.join(args.output_dir, "config.yaml"))
    logger.success(f"将配置文件保存到 {args.output_dir} 目录中的 config.yaml 中.")

    infer_dataset = build_dataset(args, visualize=False, train=False)
    infer_dataloader = DataLoader(
        infer_dataset,
        args.batch_size,
        args.shuffle,
        num_workers=args.num_workers,
        collate_fn=infer_dataset.collate_batch_test,
        pin_memory=True,
        drop_last=True,
    )

    """特征提取网络"""
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    pcd_unet = DetectionUNet2DConditionModel.from_pretrained(args.pretrained_model, revision=args.revision, subfolder="unet")
    """显存优化部分"""
    device = torch.device("cuda:1")
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
    with torch.cuda.device(device):
        if args.enable_xformers_memory_efficient_attention:
            pcd_unet.enable_xformers_memory_efficient_attention()
    if args.get("gradient_checkpointing", False):
        pcd_unet.enable_gradient_checkpointing()

    """目标检测部分"""
    detection_head, before_detection_head, _, _, _ = init_detection_modules(args)
    # 推理的时候一定要有模型权重
    load_detection_modules(args.resume_file_det, before_detection_head, detection_head)

    """模型转移以及 train 不 train"""
    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)
    before_detection_head.to(device, dtype=weight_dtype)
    detection_head.to(device, dtype=weight_dtype)

    detection_head.eval()
    before_detection_head.eval()
    prepare_processor.set_requires_grad_(False)
    pcd_unet.requires_grad_(False)

    # Create the dictionary for evaluation
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    for step, batch in enumerate(infer_dataloader):
        logger.info(f"开始对 {step} 进行推理")
        with torch.no_grad():
            ego = batch["ego"]
            input = ego["inputs_m1"]

            """diffusion 部分"""
            latents = prepare_processor.get_latents(input["bev_maps"].to(device, dtype=weight_dtype))
            noise = torch.zeros_like(latents)  # 训练 `prepare_processor.num_train_timesteps` 前, 计算出 noise 的形状
            bsz = latents.shape[0]
            # TODO: 这里训练 detection 的时候依然随机选择一个噪声是否依旧合理
            timesteps = prepare_processor.generate_timestep(bsz, device).long()
            encoder_hidden_states = prepare_processor.text_encoder(input["pcd_inputs_ids"].to(device), return_dict=False)[0]
            noisy_latents = prepare_processor.add_noise(latents, noise, timesteps)
            internal_sample = {}  # TODO: 如果显存不够用的话需要从 cuda 转移到 cpu 上, 在 forward 里面修改
            model_pred = pcd_unet(noisy_latents, timesteps, encoder_hidden_states, internal_sample=internal_sample)[0]

            """目标检测部分"""
            pcd_feature = internal_sample[args.diffusion_args.internal_sample_lay_name]
            feature = before_detection_head(pcd_feature.to(device, dtype=weight_dtype))
            cls_pred, reg_pred, dir_pred = detection_head(feature)

            """推理"""
            output_dict = {
                "ego": {
                    "cls_preds": cls_pred.detach()[:1].cpu(),
                    "reg_preds": reg_pred.detach()[:1].cpu(),
                    "dir_preds": dir_pred.detach()[:1].cpu(),
                }
            }
            pred_box_tensor, pred_score, gt_box_tensor = infer_dataset.post_process(batch, output_dict)

            caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor.to(device), result_stat, 0.3)
            caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor.to(device), result_stat, 0.5)
            caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor.to(device), result_stat, 0.7)

            if (step % args.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                logger.info(f"对 {step} 的结果进行可视化")
                vis_save_path = os.path.join(vis_save_path_root, f"step_{step:05d}.png")
                infer_result = {
                    "pred_box_tensor": pred_box_tensor,
                    "gt_box_tensor": gt_box_tensor,
                    "score_tensor": pred_score,
                }
                simple_vis.visualize(
                    infer_result, input["origin_lidar_list"][0], args.cav_lidar_range, vis_save_path, method="bev"
                )
    eval_final_results(result_stat, args.output_dir)
    logger.success(f"推理结果保存在 {args.output_dir}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection_with_heter.yaml"))
    args.resume_file_det = "/home/zfq/Desktop/logs/pcd_detection_2025_03_01/checkpoint-18-det.pth"
    args.output_dir = "~/Desktop/logs/pcd_detection_2025_03_01_infer"
    args.validate_dir = "/datasets/OPV2V/validate"
    args.save_vis_interval = 50  # 从 heal 里面抄过来的
    args.batch_size = 1
    main(args)
