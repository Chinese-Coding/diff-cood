"""推理 (inference: 简写为 infer 只为和 train 长度一样)目标检测模型"""

import torch
from loguru import logger
from torch import nn

from data_related.entity import LiftSplatShootParams
from detection_utils import caluclate_tp_fp, eval_final_results, init_detection_modules, load_detection_modules
from diffusion_utils import (
    enable_xformers_memory_efficient_attention,
    get_change_fun,
    init_dataloader,
    init_load_lss_model,
    init_logging,
    load_diffusion_processor,
)
from modules.img_processor import ImgProcessor
from modules.layering_unet_2dc_model import LayeringUNet2DCModel
from modules.pcd_processor import PcdProcessor
from opencood.visualization import simple_vis


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    args.control_model = os.path.expanduser(args.control_model)

    # 没有看错, 这里先加载 train 数据集, 因为针对目标检测任务还是在 train 数据集上进行训练
    _change = get_change_fun(args.change_args)
    infer_dataloader, infer_dataset = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf, True)

    """加载模型 (要是能写在一行就好了, 这几行代码有很明显的并列关系)"""
    img_processor = ImgProcessor(args.pretrained_model, args.revision, args.control_model, args.layering)
    pcd_processor = PcdProcessor(args.pretrained_model, args.revision, args.control_model, args.layering)
    img_processor.set_eval()
    pcd_processor.set_eval()

    """加载权重"""
    if "resume_file" in args:
        resume_file = os.path.expanduser(args.resume_file)
        img_checkpoint = load_diffusion_processor(resume_file, img_processor, pcd_processor)
        logger.success(f"从 {args.resume_file} 中加载 img 和 pcd 模型")
    else:
        logger.warning("没有指定 resume_file, 将使用最原始的预训练的 diffusion 模型")

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
    img_processor.set_weight_dtype(weight_dtype)
    pcd_processor.set_weight_dtype(weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers_memory_efficient_attention(img_processor, pcd_processor)

    if args.get("gradient_checkpointing", False):  # 训练 diffusion 的时候会用到
        img_processor.enable_gradient_checkpointing()
        pcd_processor.enable_gradient_checkpointing()

    img_device, pcd_device = torch.device("cuda:0"), torch.device("cuda:1")
    img_processor.to(img_device, weight_dtype, True)
    pcd_processor.to(pcd_device, weight_dtype, True)

    """LiftSplatShoot 模型和对应的降低通道数的卷积层"""
    lss_model = init_load_lss_model(args, img_device)

    bottleneck_layer = nn.Conv2d(128, 3, kernel_size=1)
    # 这段代码是后面添加的, 为了不改变原来函数的调用接口, 这里再单独做一个判断,
    # 可能不够简洁高效, 但是开发周期短, 先这么将就一下
    if "resume_file" in args:
        bottleneck_layer.load_state_dict(img_checkpoint["bottleneck_layer"])
        bottleneck_layer.eval()
        logger.success(f"从 {args.resume_file} 中加载 bottleneck_layer 模型")

    """目标检测部分"""
    detection_head, unsample_layer, _, _, _ = init_detection_modules(args)

    if "resume_file_det" in args:
        # 这个就是直接指定文件了, 而不是像 diffusion 那样需要指定文件名前缀
        checkpoint = load_detection_modules(args.resume_file_det, detection_head)
        """
        先用 diffusion 在 OPV2V 数据集进行预训练, 再将其用于目标检测任务, loss 为 nan (推理过程中 sample 变为 nan)
        单纯训练 diffusion 的时候 loss 也会变成 nan, 是预训练的太好了吗?
        于是我想要直接改成直接加载 diffusion 预训练的权重, 而不经过 OPV2V 数据集的预训练, 所以 `bottleneck_layer` 也变成目标检测任务的权重
        """
        if "bottleneck_layer" in checkpoint:
            bottleneck_layer.load_state_dict(checkpoint["bottleneck_layer"])
            logger.success(f"从 {args.resume_file_det} 中加载 detection_head, optimizer, lr_scheduler 以及 bottleneck_layer")
        else:
            logger.warning("没有从 `resume_file` 或 `resume_file_det`中加载 `bottleneck_layer`, 使用随机初始化的权重")
            logger.success("从 {args.resume_file_det} 中加载 detection_head, optimizer, lr_scheduler")
    else:
        raise ValueError("没有指定 `resume_file_det`")
    detection_head.eval()
    bottleneck_layer.eval()
    detection_head.to(img_device)  # 和图片是用一张显卡, 因为图片那部分占用的现存比较小
    unsample_layer.to(img_device)
    bottleneck_layer.to(img_device)

    # Create the dictionary for evaluation
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    """开始推理"""
    for step, batch in enumerate(infer_dataloader):
        with torch.no_grad():
            """预处理图像, 把图像处理为 BEV 图"""
            lss_params: LiftSplatShootParams = batch["lss_params"]
            lss_params.to(img_device)
            img = lss_model(
                lss_params.imgs, lss_params.rots, lss_params.trans, lss_params.intrins, lss_params.post_rots, lss_params.post_trans # fmt: skip
            )
            img = bottleneck_layer(img)  # 降低通道数 (128 -> 3)
            logger.debug(f"获得的输入 diffusion 的 shape: img({img.shape}) pcd({batch['pcd'].shape})")

            """图像推理"""
            _, img_params = img_processor.prepare(img.to(img_device), batch["img_inputs_ids"].to(img_device), False, args.t)
            img_params.preserved_up_indices = args.preserved_up_indices  # 相比 diffusion 多了这一步
            img_params.to(img_device)
            img_unet: LayeringUNet2DCModel = img_processor.unet
            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))

            """点云推理"""
            _, pcd_params = pcd_processor.prepare(
                batch["pcd"].to(pcd_device), batch["pcd_inputs_ids"].to(pcd_device), False, args.t
            )
            pcd_params.preserved_up_indices = args.preserved_up_indices  # 相比 diffusion 多了这一步
            pcd_params.to(pcd_device)
            pcd_unet: LayeringUNet2DCModel = pcd_processor.unet
            pcd_params = pcd_unet.forward_control(pcd_unet.forward_down(pcd_unet.forward_pre(pcd_params)))

            """交换空间 (这些东西以后写成超参数)"""
            img_sample, pcd_sample = img_params.sample.to("cpu"), pcd_params.sample.to("cpu")
            logger.debug(f"获得的中间层 Tensor 的 shape: img: {img_sample.shape}, pcd: {pcd_sample.shape}")
            img_params.sample, pcd_params.sample = _change(img_sample, pcd_sample)
            img_params.to(img_device), pcd_params.to(pcd_device)

            img_params, pcd_params = img_unet.forward_up(img_unet.forward_middle(img_params)), pcd_unet.forward_up(
                pcd_unet.forward_middle(pcd_params)
            )

            pcd_feature = pcd_params.preserved_up_feature
            # for i in args.preserved_up_indices:
            #     logger.debug(f"第 {i} 层的 feature shape. img({img_feature[i].shape}), pcd: ({pcd_feature[i].shape})")
            """取点云, 上采样层中的第 4 层作为目标检测使用的特征"""
            feature = unsample_layer(pcd_feature[3].to(device=img_device, dtype=torch.float32))
            cls_pred, reg_pred, dir_pred = detection_head(feature)
            # fmt: off

            pred_box_tensor, pred_score = infer_dataset.postprocessor.postprocess(
                infer_dataset.anchor_boxes_tensor.to(img_device), cls_pred.to(img_device), reg_pred.to(img_device), dir_pred.to(img_device)
            )
            
            caluclate_tp_fp(pred_box_tensor, pred_score, batch["gt_bbx"].to(img_device), result_stat, 0.3)
            caluclate_tp_fp(pred_box_tensor, pred_score, batch["gt_bbx"].to(img_device), result_stat, 0.5)
            caluclate_tp_fp(pred_box_tensor, pred_score, batch["gt_bbx"].to(img_device), result_stat, 0.7)
            
            if (step % args.save_vis_interval == 0) and (pred_box_tensor is not None or batch["gt_bbx"] is not None):
                vis_save_path_root = os.path.join(args.output_dir, "visualize")
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)
                
                vis_save_path = os.path.join(vis_save_path_root, f"step_{step:%05}.png")
                infer_result = {
                    "pred_box_tensor": pred_box_tensor,
                    "gt_tensor": batch["gt_bbx"],
                    "pred_score": pred_score,
                }
                simple_vis.visualize(
                    infer_result,
                    batch["origin_lidar"],
                    args["postprocess_args"]["gt_range"],
                    vis_save_path,
                    method="bev",
                    left_hand=True,
                )

    eval_final_results(result_stat, args.output_dir)


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    main(args)
