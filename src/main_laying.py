"""不使用 accelerate, 同时进行某一层交换的 main 函数"""

import math

import torch
from loguru import logger
from tqdm.auto import tqdm

from main_utils import enable_xformers_memory_efficient_attention, get_optimizer_class, init_datasloader, init_modules
from modules.dpt_processor import DptProcessor
from modules.img_processor import ImgProcessor
from modules.layering_unet_2d_condition import LayeringUNet2dConditionModel, LayeringUNetParams


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
    progress_bar = tqdm(
        range(0, int(args.max_train_steps)),
        initial=initial_global_step,
        desc="Steps",
    )
    for epoch in range(first_epoch, args.train_epochs):
        for step, batch in enumerate(train_dataloader):
            """处理图像"""
            img_noise, noisy_latents, timestep, encoder_hidden_states = img_processor.prepare(
                batch["img"].to(img_device), batch["img_inputs_ids"].to(img_device)
            )
            img_params = LayeringUNetParams(
                sample=noisy_latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )
            img_params.to(img_device)
            img_unet: LayeringUNet2dConditionModel = img_processor.unet
            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))

            """处理点云"""
            dpt_noise, noisy_latents, timestep, encoder_hidden_states, down_block_res_samples, mid_block_res_sample = (
                dpt_processor.prepare(batch["dpt"].to(dpt_device), batch["dpt_inputs_ids"].to(dpt_device))
            )
            dpt_params = LayeringUNetParams(
                sample=noisy_latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_res_samples=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
            dpt_params.to(dpt_device)
            dpt_unet: LayeringUNet2dConditionModel = dpt_processor.unet
            dpt_params = dpt_unet.forward_control(dpt_unet.forward_down(dpt_unet.forward_pre(dpt_params)))

            logger.success(f"img_sample: {img_params.sample.shape}, dpt_sample: {dpt_params.sample.shape}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load("config.yaml")
    main(args)
