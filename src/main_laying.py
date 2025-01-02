"""不使用 accelerate, 同时进行某一层交换的 main 函数"""

import math

import torch
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm

from main_utils import enable_xformers_memory_efficient_attention, get_optimizer_class, init_datasloader, init_modules
from modules.dpt_processor import DptProcessor
from modules.img_processor import ImgProcessor
from modules.layering_unet_2d_condition import LayeringUNet2DCModel, LayeringUNet2DCParams


def main(args):
    optimizer_class = get_optimizer_class(args)
    train_dataloader = init_datasloader(args)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epochs * num_update_steps_per_epoch
        args.train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    """加载模型 (标注提示信息, 方便 IDE 提示)"""
    img_processor, img_optimizer, img_lr_scheduler = init_modules(
        args, ImgProcessor, optimizer_class
    )  # type: ImgProcessor, ignore, ignore
    dpt_processor, dpt_optimizer, dpt_lr_scheduler = init_modules(
        args, DptProcessor, optimizer_class
    )  # type: DptProcessor, ignore, ignore

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
    dpt_processor.set_weight_dtype(weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        enable_xformers_memory_efficient_attention(img_processor, dpt_processor)

    if args.gradient_checkpointing:
        img_processor.enable_gradient_checkpointing()
        dpt_processor.enable_gradient_checkpointing()
    img_device, dpt_device = torch.device("cuda:0"), torch.device("cuda:1")
    img_processor.to(img_device, weight_dtype, True)
    dpt_processor.to(dpt_device, weight_dtype, True)

    initial_global_step = 0
    global_step = 0
    first_epoch = 0
    progress_bar = tqdm(range(0, int(args.max_train_steps)), initial=initial_global_step, desc="Steps")
    for epoch in range(first_epoch, args.train_epochs):
        for step, batch in enumerate(train_dataloader):
            """处理图像"""
            img_noise, img_params = img_processor.prepare(
                batch["img"].to(img_device), batch["img_inputs_ids"].to(img_device), False
            )  # type: torch.Tensor, LayeringUNet2DCParams
            img_unet: LayeringUNet2DCModel = img_processor.unet
            img_params.to(img_device)
            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))

            """处理点云"""
            dpt_noise, dpt_params = dpt_processor.prepare(
                batch["dpt"].to(dpt_device), batch["dpt_inputs_ids"].to(dpt_device), False
            )  # type: torch.Tensor, LayeringUNet2DCParams

            dpt_unet: LayeringUNet2DCModel = dpt_processor.unet
            dpt_params.to(dpt_device)
            dpt_params = dpt_unet.forward_control(dpt_unet.forward_down(dpt_unet.forward_pre(dpt_params)))

            """
            交换空间 (这些东西以后写成超参数)
            交换的代码有问题, 这样写会导致:
            Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).
            Saved intermediate values of the graph are freed when you call .backward() or autograd.grad().
            Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
            还有如何进行交换的问题:
            获得的中间层 Tensor 的 shape: img: torch.Size([4, 1280, 8, 8]), dpt: torch.Size([1, 1280, 8, 8])
            得到的中间特征, 通道多, 而每个通道上的特征图少
            """
            img_sample, dpt_sample = img_params.sample.to("cpu"), dpt_params.sample.to("cpu")
            # logger.success(f"获得的中间层 Tensor 的 shape: img: {img_sample.shape}, dpt: {dpt_sample.shape}")
            channel_index = 2
            i1, j1, i2, j2 = 16, 16, 16, 16
            img_block, dpt_block = img_sample[0, channel_index, i1:i2, j1:j2], dpt_sample[0, channel_index, i1:i2, j1:j2]
            img_sample[0, channel_index, i1:i2, j1:j2], dpt_sample[0, channel_index, i1:i2, j1:j2] = dpt_block, img_block
            # logger.success(f"交换之后的中间层 Tensor 的 shape: img: {img_sample.shape}, dpt: {dpt_sample.shape}")
            img_params.sample, dpt_params.sample = img_sample, dpt_sample

            """交换完之后的步骤, 开始走没走完的层"""
            img_params.to(img_device), dpt_params.to(dpt_device)
            img_noise_pred, dpt_noise_pred = (
                img_unet.forward_up(img_unet.forward_middle(img_params)).sample,
                dpt_unet.forward_up(dpt_unet.forward_middle(dpt_params)).sample,
            )  # 犹豫再三还是卸载了一行里面 (虽然会被 black 格式化成 4 行)

            """计算损失, 开始反向传播"""
            img_loss, dpt_loss = (
                F.mse_loss(img_noise_pred.float(), img_noise.float(), reduction="mean"),
                F.mse_loss(dpt_noise_pred.float(), dpt_noise.float(), reduction="mean"),
            )  # 犹豫再三还是卸载了一行里面 (虽然会被 black 格式化成 4 行)
            img_loss.backward(retain_graph=True)
            dpt_loss.backward()

            img_optimizer.step()
            img_lr_scheduler.step()
            img_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            dpt_optimizer.step()
            dpt_lr_scheduler.step()
            dpt_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            img_logs = {
                "img_loss": img_loss.detach().item(),
                "img_lr": img_lr_scheduler.get_last_lr()[0],
            }
            dpt_logs = {
                "dpt_loss": dpt_loss.detach().item(),
                "dpt_lr": dpt_lr_scheduler.get_last_lr()[0],
            }
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**{**img_logs, **dpt_logs})


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/config.yaml"))
    main(args)
