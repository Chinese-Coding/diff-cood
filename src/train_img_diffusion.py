"""不使用 accelerate, 同时进行某一层交换的 train diffusion 函数"""

import math

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm.auto import tqdm

from data_related.entity import LiftSplatShootParams
from diffusion_utils import (
    enable_xformers_memory_efficient_attention_with_one_processor,
    get_optimizer_class,
    init_dataloader,
    init_logging,
    init_modules,
    load_diffusion_modules,
    save_modules,
)
from modules.img_processor import ImgProcessor
from modules.layering_unet_2dc_model import LayeringUNet2DCModel
from opencood.models.lift_splat_shoot import LiftSplatShoot


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    writer = init_logging(args)

    optimizer_class = get_optimizer_class(args)
    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epoches * num_update_steps_per_epoch
        args.train_epoches = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    """
    加载模型 (标注提示信息, 方便 IDE 提示)
    简写说明: Img for Image (图像); Pcd for point cloud (点云, 缩写成三个字母, 而不是两个字母的 pc, 主要是为了和图像的缩写保持同样的长度, 这样看起来比较方便)
    """
    img_processor, img_optimizer, img_lr_scheduler = init_modules(
        args, ImgProcessor, optimizer_class
    )  # type: ImgProcessor, ignore, ignore
    img_processor.set_train()

    """加载权重 (上面那个是预训练权重, 下面这个是自己的权重)"""
    first_epoch, img_loss = 0, 0  # 为保存权重特地将变量声明到前面
    if "resume_file" in args:
        resume_file = os.path.expanduser(args.resume_file)
        # img_optimizer 不在这里加载权重了, 因为此时 img_optimizer 里面还没有 bottleneck_layer 的权重
        img_checkpoint = load_diffusion_modules(f"{resume_file}-img.pth", img_processor.unet, img_lr_scheduler)
        first_epoch = img_checkpoint["epoch"] + 1
        logger.success(f"从 {args.resume_file} 中加载 img 模型")

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

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers_memory_efficient_attention_with_one_processor(img_processor)

    if args.gradient_checkpointing:
        img_processor.enable_gradient_checkpointing()

    img_device = torch.device("cuda:1")
    img_processor.to(img_device, weight_dtype, True)

    """LiftSplatShoot 模型和对应的降低通道数的卷积层"""
    lss_model = LiftSplatShoot(args.lift_splat_shoot_args, img_device)
    lss_model.load_state_dict(
        torch.load(os.path.expanduser(args.lift_splat_shoot_args.pretrained_model_path), weights_only=False), strict=False
    )
    lss_model.eval()
    lss_model.to(device=img_device)
    bottleneck_layer = nn.Conv2d(128, 3, kernel_size=1)
    # 增加了一个新的卷积层, 别忘了把他添加到 optimizer 里面
    img_optimizer.add_param_group({"params": bottleneck_layer.parameters()})

    # 这段代码是后面添加的, 为了不改变原来函数的调用接口, 这里再单独做一个判断,
    # 可能不够简洁高效, 但是开发周期短, 先这么将就一下
    if "resume_file" in args:
        bottleneck_layer.load_state_dict(img_checkpoint["bottleneck_layer"])
        img_optimizer.load_state_dict(img_checkpoint["optimizer"])
    bottleneck_layer.train()
    bottleneck_layer.to(device=img_device)

    global_step = (first_epoch - 1) * len(train_dataloader) / args.batch_size if first_epoch > 0 else 0
    logger.success(f"从 {first_epoch} 开始训练, 共训练 {args.train_epoches} 个 epoch")

    for epoch in range(first_epoch, args.train_epoches):
        logger.success(f"第 {epoch} 个 epoch 开始训练")
        progress_bar = tqdm(range(0, int(len(train_dataloader))), initial=0, desc=f"Epoch: {epoch}/{args.train_epoches}")
        for step, batch in enumerate(train_dataloader):
            """预处理图像, 把图像处理为 BEV 图"""
            lss_params: LiftSplatShootParams = batch["lss_params"]
            lss_params.to(img_device)
            img = lss_model(
                lss_params.imgs, lss_params.rots, lss_params.trans, lss_params.intrins, lss_params.post_rots, lss_params.post_trans # fmt: skip
            )
            img = bottleneck_layer(img)  # 降低通道数 (128 -> 3)

            """处理图像"""
            img_noise, img_params = img_processor.prepare(
                img.to(img_device), batch["img_inputs_ids"].to(img_device), False
            )  # type: torch.Tensor, LayeringUNet2DCParams
            img_unet: LayeringUNet2DCModel = img_processor.unet
            img_params.to(img_device)
            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))

            """交换完之后的步骤, 开始走没走完的层"""
            img_params.to(img_device)
            img_noise_pred = img_unet.forward_up(img_unet.forward_middle(img_params)).sample

            """计算损失, 开始反向传播"""
            img_loss = F.mse_loss(img_noise_pred.float(), img_noise.float(), reduction="mean")
            img_loss.backward()

            torch.nn.utils.clip_grad_norm_(img_unet.parameters(), args.max_grad_norm)

            img_optimizer.step()
            img_lr_scheduler.step()
            img_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            img_logs = {
                "img_loss": img_loss.detach().item(),
                "img_lr": img_lr_scheduler.get_last_lr()[0],
            }

            # 记录损失到 TensorBoard
            writer.add_scalar("Loss/img_loss", img_logs["img_loss"], global_step)

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**img_logs)

        # fmt: off
        save_modules(
            args.output_dir, epoch, img_processor.unet, img_optimizer, img_lr_scheduler,
            "img", bottleneck_layer=bottleneck_layer
        )
    writer.close()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_diffusion.yaml"))
    args.output_dir = os.path.expanduser("~/Desktop/logs/img_diffusion_2025_01_22")
    # args.resume_file = os.path.expanduser("~/Desktop/logs/img_diffusion_2025_01_16/checkpoint-12")
    args.batch_size = 2
    args.num_workers = 8
    args.shuffle = True
    args.train_epoches = 30
    main(args)
