import einops
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, AutoTokenizer
from pathlib import Path
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
import torch
from diffusers.optimization import get_scheduler
from data_related.stable_diffusion_dataset import StableDiffusionDataset
from data_related.transform_funs import dpt_transform, img_transform
from accelerate.logging import get_logger
import torch.nn.functional as F
import math

logger = get_logger(__name__)


def import_model_class_from_pretrained_model(pretrained_model: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    """加载模型 (TODO: 一些不需要训练的模型能共用一个吗?)"""
    text_encoder_cls = import_model_class_from_pretrained_model(args.pretrained_model, args.revision)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    """模型数据集部分"""
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    train_dataset = StableDiffusionDataset(args.root_dir)
    train_dataset.reinitialize()
    train_dataset.set_transform(img_transform(args), dpt_transform(args))
    train_dataset.set_tokenizer(tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epochs * num_update_steps_per_epoch
        args.train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    """
    TODO: 模型定义部分还可以改进, 可以使用 python 向程序中直接添加变量的方式,
    这样只需要定义一个函数然后再制定一个前缀就可以了
    """
    # 图像部分模型
    img_accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    img_vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    img_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    img_text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    img_controlnet = ControlNetModel.from_pretrained(args.pretrained_model)
    img_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    img_vae.requires_grad_(False)
    img_unet.train()
    img_text_encoder.requires_grad_(False)
    img_controlnet.requires_grad_(False)

    img_optimizer = optimizer_class(
        img_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    img_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=img_optimizer,
        num_warmup_steps=args.lr_warmup_steps * img_accelerator.num_processes,
        num_training_steps=args.max_train_steps * img_accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    img_unet, img_optimizer, img_lr_scheduler = img_accelerator.prepare(img_unet, img_optimizer, img_lr_scheduler)

    # 点云处理部分 (目前先按照参考项目的来设置)
    dpt_accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    dpt_vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    dpt_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    dpt_text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    dpt_controlnet = ControlNetModel.from_pretrained(args.pretrained_model)
    dpt_noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    dpt_vae.requires_grad_(False)
    dpt_unet.train()
    dpt_text_encoder.requires_grad_(False)
    dpt_controlnet.requires_grad_(False)

    dpt_optimizer = optimizer_class(
        dpt_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    dpt_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=dpt_optimizer,
        num_warmup_steps=args.lr_warmup_steps * dpt_accelerator.num_processes,
        num_training_steps=args.max_train_steps * dpt_accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    dpt_unet, dpt_optimizer, dpt_lr_scheduler = dpt_accelerator.prepare(dpt_unet, dpt_optimizer, dpt_lr_scheduler)

    weight_dtype = torch.float32
    if img_accelerator.mixed_precision == "fp16" and dpt_accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif img_accelerator.mixed_precision == "bf16" and dpt_accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    """优化部分"""
    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please"
                    " update xFormers to at least 0.0.17. See"
                    " https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            img_unet.enable_xformers_memory_efficient_attention()
            img_controlnet.enable_xformers_memory_efficient_attention()
            dpt_unet.enable_xformers_memory_efficient_attention()
            img_controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    """移动模型到指定设备上"""
    img_vae.to(img_accelerator.device, dtype=weight_dtype)
    img_controlnet.to(img_accelerator.device, dtype=weight_dtype)
    img_text_encoder.to(img_accelerator.device, dtype=weight_dtype)

    dpt_vae.to(dpt_accelerator.device, dtype=weight_dtype)
    dpt_controlnet.to(dpt_accelerator, dtype=weight_dtype)
    dpt_text_encoder.to(dpt_accelerator, dtype=weight_dtype)

    initial_global_step = 0
    # progress_bar = tqdm(
    #     range(0, args.max_train_steps),
    #     initial=initial_global_step,
    #     desc="Steps",
    # )
    first_epoch = 0
    for epoch in range(first_epoch, args.train_epochs):
        for step, batch in enumerate(train_dataloader):
            with img_accelerator.accumulate(img_unet), dpt_accelerator.accumulate(dpt_unet):
                """图像部分"""
                imgs = batch["img"]
                # TODO: 不确定这样是对还是错
                imgs = einops.rearrange(imgs, "b n c h w -> b * n c h w")
                img_latents = img_vae.encode(imgs.to(dtype=weight_dtype)).latent_dist.sample()
                img_latents = img_latents * img_vae.config.scaling_factor

                img_noise = torch.randn_like(img_latents)
                img_bsz = img_latents.shape[0]
                img_timestamps = torch.randint(
                    0, img_noise_scheduler.config.num_train_timesteps, (img_bsz,), device=img_latents.device
                ).long()
                img_noisy_latents = img_noise_scheduler.add_noise(img_latents.float(), img_noise.float(),
                                                                  img_timestamps).to(
                    dtype=weight_dtype
                )
                img_encoder_hidden_states = img_text_encoder(batch["inputs_ids"], return_dict=False)[0]
                img_noise_pred = img_unet(
                    img_noisy_latents, img_timestamps, encoder_hidden_states=img_encoder_hidden_states,
                    return_dict=False
                )

                """点云部分 (TODO: 即将深度信息作为输入, 又将其作为控制条件)"""
                dpt = batch["dpt"]
                dpt_latents = dpt_vae.encode(dpt.to(dtype=weight_dtype)).latent_dist.sample()
                dpt_latents = dpt_latents * dpt_vae.config.scaling_factor
                dpt_noise = torch.randn_like(dpt_latents)
                dpt_bsz = dpt_latents.shape[0]
                dpt_timestamps = torch.randint(
                    0, dpt_noise_scheduler.config.num_train_timesteps, (dpt_bsz,), device=dpt_latents.device
                ).long()
                dpt_noisy_latents = dpt_noise_scheduler.add_noise(dpt_latents.float(), dpt_noise.float(),
                                                                  dpt_timestamps).to(
                    dtype=weight_dtype
                )
                down_block_res_samples, mid_block_res_sample = dpt_controlnet(
                    dpt_noisy_latents,
                    dpt_latents,
                    controlnet_cond=dpt.to(dtype=weight_dtype),
                    return_dict=False,
                )
                dpt_noise_pred = dpt_unet(
                    dpt_noisy_latents,
                    dpt_timestamps,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in
                                                     down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )

                # 计算损失与优化部分
                img_loss = F.mse_loss(img_noise_pred.float(), img_noise.float(), reduction="mean")
                dpt_loss = F.mse_loss(dpt_noise_pred.float(), dpt_noise.float(), reduction="mean")
                img_accelerator.backward(img_loss)
                dpt_accelerator.backward(dpt_loss)

                img_optimizer.step()
                img_lr_scheduler.step()
                img_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                dpt_optimizer.step()
                dpt_lr_scheduler.step()
                dpt_optimizer.zero_grad(set_to_none=args.set_grads_to_none)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import sys

    args = OmegaConf.load("config.yaml")
    main(args)
