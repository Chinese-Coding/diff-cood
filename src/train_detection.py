import torch

torch.autograd.set_detect_anomaly(True)
from loguru import logger
from torch import nn

from data_related.entity import LiftSplatShootParams
from detection_utils import init_detection_modules, load_detection_modules
from diffusion_utils import (
    enable_xformers_memory_efficient_attention,
    get_change_fun,
    init_dataloader,
    init_logging,
    load_diffusion_processor,
)
from modules.img_processor import ImgProcessor
from modules.layering_unet_2dc_model import LayeringUNet2DCModel
from modules.pcd_processor import PcdProcessor
from opencood.models.lift_splat_shoot import LiftSplatShoot


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    args.control_model = os.path.expanduser(args.control_model)
    writer = init_logging(args)

    # 没有看错, 这里先加载 train 数据集, 因为针对目标检测任务还是在 train 数据集上进行训练
    _change = get_change_fun(args.change_args)
    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

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

    img_device, pcd_device = torch.device("cuda:1"), torch.device("cuda:1")
    img_processor.to(img_device, weight_dtype, True)
    pcd_processor.to(pcd_device, weight_dtype, True)

    """LiftSplatShoot 模型和对应的降低通道数的卷积层"""
    lss_model = LiftSplatShoot(args.lift_splat_shoot_args, img_device)
    lss_model.load_state_dict(
        torch.load(os.path.expanduser(args.lift_splat_shoot_args.pretrained_model_path), weights_only=False), strict=False
    )
    lss_model.eval()
    lss_model.to(device=img_device)
    bottleneck_layer = nn.Conv2d(128, 3, kernel_size=1)
    # 这段代码是后面添加的, 为了不改变原来函数的调用接口, 这里再单独做一个判断,
    # 可能不够简洁高效, 但是开发周期短, 先这么将就一下
    if "resume_file" in args:
        bottleneck_layer.load_state_dict(img_checkpoint["bottleneck_layer"])
        bottleneck_layer.eval()
        logger.success(f"从 {args.resume_file} 中加载 bottleneck_layer 模型")

    """目标检测部分"""
    detection_head, unsample_layer, loss_fn, optimizer, lr_scheduler = init_detection_modules(args)
    first_epoch = 0
    if "resume_file_det" in args:
        checkpoint = load_detection_modules(args.resume_file_det, detection_head, optimizer, lr_scheduler)
        first_epoch = checkpoint["epoch"] + 1

        """
        先用 diffusion 在 OPV2V 数据集进行预训练, 再将其用于目标检测任务, loss 为 nan (推理过程中 sample 变为 nan)
        单纯训练 diffusion 的时候 loss 也会变成 nan, 是预训练的太好了吗?
        于是我想要直接改成直接加载 diffusion 预训练的权重, 而不经过 OPV2V 数据集的预训练, 所以 `bottleneck_layer` 也变成目标检测任务的权重
        """
        if "resume_file" not in args:
            bottleneck_layer.load_state_dict(checkpoint["bottleneck_layer"])
            logger.success(f"从 {args.resume_file_det} 中加载 detection_head, optimizer, lr_scheduler 以及 bottleneck_layer")
        else:
            logger.success(f"从 {args.resume_file_det} 中加载 detection_head, optimizer, lr_scheduler")

    detection_head.train()
    bottleneck_layer.train()
    detection_head.to(img_device)  # 和图片是用一张显卡, 因为图片那部分占用的现存比较小
    unsample_layer.to(img_device)
    bottleneck_layer.to(img_device)
    """开始推理"""
    for epoch in range(first_epoch, args.train_epoches):  # TODO: 这里的训练次数应该写成超参数
        logger.success(f"第 {epoch} 个 epoch 开始训练")
        for step, batch in enumerate(train_dataloader):
            """预处理图像, 把图像处理为 BEV 图"""
            lss_params: LiftSplatShootParams = batch["lss_params"]
            lss_params.to(img_device)
            img = lss_model(
                lss_params.imgs, lss_params.rots, lss_params.trans, lss_params.intrins, lss_params.post_rots, lss_params.post_trans # fmt: skip
            )
            img = bottleneck_layer(img)  # 降低通道数 (128 -> 3)
            logger.debug(f"获得的输入 diffusion 的 shape: img({img.shape}) pcd({batch['pcd'].shape})")

            """图像推理"""
            _, img_params = img_processor.prepare(
                img.to(img_device), batch["img_inputs_ids"].to(img_device), False, args.t
            )  # type: torch.Tensor, LayeringUNet2DCParams
            img_params.preserved_up_indices = args.preserved_up_indices  # 相比 diffusion 多了这一步
            img_params.to(img_device)
            img_unet: LayeringUNet2DCModel = img_processor.unet
            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))

            """点云推理"""
            _, pcd_params = pcd_processor.prepare(
                batch["pcd"].to(pcd_device), batch["pcd_inputs_ids"].to(pcd_device), False, args.t
            )  # type: torch.Tensor, LayeringUNet2DCParams
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
            total_loss = loss_fn(
                cls_pred, reg_pred, dir_pred,
                batch["pos_equal_one"].to(img_device), batch["neg_equal_one"].to(img_device), batch["targets"].to(img_device)
            )
            # fmt: on
            loss_fn.logging(epoch, step, len(train_dataloader), writer)

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        save_dict = {
            "epoch": epoch,
            "detection_head": detection_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        if "resume_file" not in args:
            save_dict["bottleneck_layer"] = bottleneck_layer.state_dict()
        torch.save(
            save_dict,
            f"{args.output_dir}/checkpoint-{epoch}-det.pth",
        )
    writer.close()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection.yaml"))
    main(args)
