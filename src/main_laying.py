"""不使用 accelerate, 同时进行某一层交换的 main 函数"""

import math

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from main_utils import (
    enable_xformers_memory_efficient_attention,
    get_change_fun,
    get_optimizer_class,
    init_datasloader,
    init_modules,
    load_modules,
    save_modules,
)
from modules.img_processor import ImgProcessor
from modules.layering_unet_2dc_model import LayeringUNet2DCModel, LayeringUNet2DCParams
from modules.pcd_processor import PcdProcessor


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    logfile_path = os.path.join(logging_dir, "{time:YYYY-MM-DD}.log")
    writer = SummaryWriter(log_dir=logging_dir)
    logger.add(logfile_path, rotation="1 day")

    _change = get_change_fun(args.change_args)
    optimizer_class = get_optimizer_class(args)
    train_dataloader = init_datasloader(args)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epochs * num_update_steps_per_epoch
        args.train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    """
    加载模型 (标注提示信息, 方便 IDE 提示)
    简写说明: Img for Image (图像); Pcd for point cloud (点云, 缩写成三个字母, 而不是两个字母的 pc, 主要是为了和图像的缩写保持同样的长度, 这样看起来比较方便) 
    """
    img_processor, img_optimizer, img_lr_scheduler = init_modules(
        args, ImgProcessor, optimizer_class
    )  # type: ImgProcessor, ignore, ignore
    pcd_processor, pcd_optimizer, pcd_lr_scheduler = init_modules(
        args, PcdProcessor, optimizer_class
    )  # type: PcdProcessor, ignore, ignore
    img_processor.set_train()
    pcd_processor.set_train()

    """加载权重 (上面那个是预训练权重, 下面这个是自己的权重)"""
    first_epoch, img_loss, pcd_loss = 0, 0, 0  # 为保存权重特地将变量声明到前面
    if "resume_file" in args:
        img_epoch = load_modules(f"{args.resume_file}-img.pth", img_processor.unet, img_optimizer, img_lr_scheduler)
        pcd_epoch = load_modules(f"{args.resume_file}-pcd.pth", pcd_processor.unet, pcd_optimizer, pcd_lr_scheduler)
        assert (
            img_epoch == pcd_epoch
        )  # 检查一下两个 epoch 相等 (虽然 assert 可以选择关闭, 但是一行检查代码写起来简单, 而且一般也不会关闭)
        first_epoch = img_epoch + 1
        logger.success(f"从 {args.resume_file} 中加载 img 和 pcd 模型")

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

    global_step = 0
    progress_bar = tqdm(range(0, int(args.max_train_steps)), initial=0, desc="Steps")
    logger.success(f"从 {first_epoch} 开始训练, 共训练 {args.train_epochs} 个 epoch")
    for epoch in range(first_epoch, args.train_epochs):
        logger.success(f"第 {epoch} 个 epoch 开始训练")
        for step, batch in enumerate(train_dataloader):
            """处理图像"""
            img_noise, img_params = img_processor.prepare(
                batch["img"].to(img_device), batch["img_inputs_ids"].to(img_device), False
            )  # type: torch.Tensor, LayeringUNet2DCParams
            img_unet: LayeringUNet2DCModel = img_processor.unet
            img_params.to(img_device)
            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))

            """处理点云"""
            pcd_noise, pcd_params = pcd_processor.prepare(
                batch["pcd"].to(pcd_device), batch["pcd_inputs_ids"].to(pcd_device), False
            )  # type: torch.Tensor, LayeringUNet2DCParams

            pcd_unet: LayeringUNet2DCModel = pcd_processor.unet
            pcd_params.to(pcd_device)
            pcd_params = pcd_unet.forward_control(pcd_unet.forward_down(pcd_unet.forward_pre(pcd_params)))

            """交换空间 (这些东西以后写成超参数)"""
            img_sample, pcd_sample = img_params.sample.to("cpu"), pcd_params.sample.to("cpu")
            # logger.debug(f"获得的中间层 Tensor 的 shape: img: {img_sample.shape}, pcd: {pcd_sample.shape}")
            img_params.sample, pcd_params.sample = _change(img_sample, pcd_sample, args.change_config)

            """交换完之后的步骤, 开始走没走完的层"""
            img_params.to(img_device), pcd_params.to(pcd_device)
            img_noise_pred, pcd_noise_pred = (
                img_unet.forward_up(img_unet.forward_middle(img_params)).sample,
                pcd_unet.forward_up(pcd_unet.forward_middle(pcd_params)).sample,
            )  # 犹豫再三还是卸载了一行里面 (虽然会被 black 格式化成 4 行)

            """计算损失, 开始反向传播"""
            img_loss, pcd_loss = (
                F.mse_loss(img_noise_pred.float(), img_noise.float(), reduction="mean"),
                F.mse_loss(pcd_noise_pred.float(), pcd_noise.float(), reduction="mean"),
            )  # 犹豫再三还是卸载了一行里面 (虽然会被 black 格式化成 4 行)
            img_loss.backward(retain_graph=True)
            pcd_loss.backward()

            img_optimizer.step()
            img_lr_scheduler.step()
            img_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            pcd_optimizer.step()
            pcd_lr_scheduler.step()
            pcd_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            img_logs = {
                "img_loss": img_loss.detach().item(),
                "img_lr": img_lr_scheduler.get_last_lr()[0],
            }
            pcd_logs = {
                "pcd_loss": pcd_loss.detach().item(),
                "pcd_lr": pcd_lr_scheduler.get_last_lr()[0],
            }

            # 记录损失到 TensorBoard
            writer.add_scalar("Loss/img_loss", img_loss.detach().item(), global_step)
            writer.add_scalar("Loss/pcd_loss", pcd_loss.detach().item(), global_step)

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**{**img_logs, **pcd_logs})

        save_modules(args.output_dir, epoch, img_processor.unet, img_optimizer, img_lr_scheduler, "img")
        save_modules(args.output_dir, epoch, pcd_processor.unet, pcd_optimizer, pcd_lr_scheduler, "pcd")

    writer.close()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/config.yaml"))
    main(args)
