from tokenize import Ignore

import torch
from loguru import logger

from main_utils import enable_xformers_memory_efficient_attention, get_change_fun, init_datasloader
from modules.detection_head import DetectionHead
from modules.img_processor import ImgProcessor
from modules.layering_unet_2dc_model import LayeringUNet2DCModel, LayeringUNet2DCParams
from modules.pcd_processor import PcdProcessor
from opencood.loss.point_pillar_loss import PointPillarLoss


def main(args):
    # 没有看错, 这里先加载 train 数据集, 因为只有 train 数据集里面的数据进行过点云拆分
    _change = get_change_fun(args.change_args)
    train_dataloader = init_datasloader(args)

    detection_head = DetectionHead(args.postprocess_args.anchor_args.num, args.postprocess_args.dir_args)
    loss_fn = PointPillarLoss(args.loss_args)
    # 优化器参数从 HEAL 中的某个配置文件抄过来的, 应该写成超参数的形式
    optimizer = torch.optim.Adam(detection_head.parameters(), lr=0.002, eps=1e-10, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    """加载模型 (要是能写在一行就好了, 这几行代码有很明显的并列关系)"""
    img_processor = ImgProcessor(args.pretrained_model, args.revision, args.control_model, args.layering)
    pcd_processor = PcdProcessor(args.pretrained_model, args.revision, args.control_model, args.layering)
    img_processor.set_eval()
    pcd_processor.set_eval()

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

    if args.gradient_checkpointing:
        img_processor.enable_gradient_checkpointing()
        pcd_processor.enable_gradient_checkpointing()

    img_device, pcd_device = torch.device("cuda:0"), torch.device("cuda:1")
    img_processor.to(img_device, weight_dtype, True)
    pcd_processor.to(pcd_device, weight_dtype, True)
    detection_head.to(img_device)  # 和图片是用一张显卡, 因为图片那部分占用的现存比较小
    """开始推理"""
    for epoch in range(10):
        for step, batch in enumerate(train_dataloader):
            """图像推理"""
            _, img_params = img_processor.prepare(
                batch["img"].to(img_device), batch["img_inputs_ids"].to(img_device), False, args.t
            )  # type: torch.Tensor, LayeringUNet2DCParams
            img_params.preserved_up_indices = args.preserved_up_indices
            img_params.to(img_device)
            img_unet: LayeringUNet2DCModel = img_processor.unet
            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))

            """点云推理"""
            _, pcd_params = pcd_processor.prepare(
                batch["pcd"].to(pcd_device), batch["pcd_inputs_ids"].to(pcd_device), False, args.t
            )  # type: torch.Tensor, LayeringUNet2DCParams
            pcd_params.preserved_up_indices = args.preserved_up_indices
            pcd_params.to(pcd_device)
            pcd_unet: LayeringUNet2DCModel = pcd_processor.unet
            pcd_params = pcd_unet.forward_control(pcd_unet.forward_down(pcd_unet.forward_pre(pcd_params)))

            """交换空间 (这些东西以后写成超参数)"""
            img_sample, pcd_sample = img_params.sample.to("cpu"), pcd_params.sample.to("cpu")
            # logger.debug(f"获得的中间层 Tensor 的 shape: img: {img_sample.shape}, pcd: {pcd_sample.shape}")
            img_params.sample, pcd_params.sample = _change(img_sample, pcd_sample)
            img_params.to(img_device), pcd_params.to(pcd_device)

            img_params, pcd_params = img_unet.forward_up(img_unet.forward_middle(img_params)), pcd_unet.forward_up(
                pcd_unet.forward_middle(pcd_params)
            )

            img_feature, pcd_feature = img_params.preserved_up_feature, pcd_params.preserved_up_feature
            # for i in args.preserved_up_indices:
            #     logger.success(f"第 {i} 层的 feature shape. img: {img_feature[i].shape}, pcd: {pcd_feature[i].shape}")
            """取点云, 上采样层中的第 4 层作为目标检测使用的特征"""
            """ TODO: 把 4 张特征图合成一个, 然后送入检测头"""
            cls_pred, reg_pred, dir_pred = detection_head(pcd_feature[3].to(device=img_device, dtype=torch.float32))
            # fmt: off
            total_loss = loss_fn(
                cls_pred, reg_pred, dir_pred,
                batch["pos_equal_one"].to(img_device), batch["neg_equal_one"].to(img_device), batch["targets"].to(img_device)
            )
            # fmt: on
            loss_fn.logging(epoch, step, len(train_dataloader))

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/config.yaml"))
    main(args)
