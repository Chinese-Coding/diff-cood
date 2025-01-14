"""不使用 accelerate, 同时进行某一层交换的 train diffusion 函数"""

import math

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm

from diffusion_utils import (
    enable_xformers_memory_efficient_attention_with_one_processor,
    get_optimizer_class,
    init_datasloader,
    init_logging,
    init_modules,
    load_modules,
    save_modules,
)
from modules.layering_unet_2dc_model import LayeringUNet2DCModel
from modules.pcd_processor import PcdProcessor


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)
    args.control_model = os.path.expanduser(args.control_model)

    writer = init_logging(args)

    optimizer_class = get_optimizer_class(args)
    train_dataloader = init_datasloader(args, args.lift_splat_shoot_args.data_aug_conf)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epoches * num_update_steps_per_epoch
        args.train_epoches = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    """
    加载模型 (标注提示信息, 方便 IDE 提示)
    简写说明: Img for Image (图像); Pcd for point cloud (点云, 缩写成三个字母, 而不是两个字母的 pc, 主要是为了和图像的缩写保持同样的长度, 这样看起来比较方便)
    """
    pcd_processor, pcd_optimizer, pcd_lr_scheduler = init_modules(
        args, PcdProcessor, optimizer_class
    )  # type: PcdProcessor, ignore, ignore
    pcd_processor.set_train()

    """加载权重 (上面那个是预训练权重, 下面这个是自己的权重)"""
    first_epoch, pcd_loss = 0, 0  # 为保存权重特地将变量声明到前面
    if "resume_file" in args:
        resume_file = os.path.expanduser(args.resume_file)
        pcd_epoch = load_modules(f"{resume_file}-pcd.pth", pcd_processor.unet, pcd_optimizer, pcd_lr_scheduler)
        first_epoch = pcd_epoch + 1
        logger.success(f"从 {args.resume_file} 中加载 pcd 模型")

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
    pcd_processor.set_weight_dtype(weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers_memory_efficient_attention_with_one_processor(pcd_processor)

    if args.gradient_checkpointing:
        pcd_processor.enable_gradient_checkpointing()

    pcd_device = torch.device("cuda:1")
    pcd_processor.to(pcd_device, weight_dtype, True)

    global_step = 0
    progress_bar = tqdm(range(0, int(args.max_train_steps)), initial=0, desc="Steps")
    logger.success(f"从 {first_epoch} 开始训练, 共训练 {args.train_epoches} 个 epoch")

    for epoch in range(first_epoch, args.train_epoches):
        logger.success(f"第 {epoch} 个 epoch 开始训练")
        for step, batch in enumerate(train_dataloader):
            """处理点云"""
            pcd_noise, pcd_params = pcd_processor.prepare(
                batch["pcd"].to(pcd_device), batch["pcd_inputs_ids"].to(pcd_device), False
            )  # type: torch.Tensor, LayeringUNet2DCParams

            pcd_unet: LayeringUNet2DCModel = pcd_processor.unet
            pcd_params.to(pcd_device)
            pcd_params = pcd_unet.forward_control(pcd_unet.forward_down(pcd_unet.forward_pre(pcd_params)))

            """交换空间 (这些东西以后写成超参数)"""
            pcd_params.to(pcd_device)
            pcd_noise_pred = pcd_unet.forward_up(pcd_unet.forward_middle(pcd_params)).sample

            """计算损失, 开始反向传播"""
            pcd_loss = F.mse_loss(pcd_noise_pred.float(), pcd_noise.float(), reduction="mean")

            pcd_loss.backward()

            pcd_optimizer.step()
            pcd_lr_scheduler.step()
            pcd_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            pcd_logs = {
                "pcd_loss": pcd_loss.detach().item(),
                "pcd_lr": pcd_lr_scheduler.get_last_lr()[0],
            }

            writer.add_scalar("Loss/pcd_loss", pcd_loss.detach().item(), global_step)

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**pcd_logs)

        save_modules(args.output_dir, epoch, pcd_processor.unet, pcd_optimizer, pcd_lr_scheduler, "pcd")

    writer.close()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_diffusion.yaml"))
    args.output_dir = os.path.expanduser("~/Desktop/logs/pcd_diffusion")
    args.batch_size = 8
    args.num_workers = 16
    main(args)
