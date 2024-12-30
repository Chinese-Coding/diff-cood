import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from data_related.stable_diffusion_dataset import StableDiffusionDataset
from data_related.transform_funs import dpt_transform, img_transform
from modules.dpt_processor import DptProcessor
from modules.img_processor import ImgProcessor

logger = get_logger(__name__)


def _get_optimizer_class(args):
    """判断是否启用 8bit 的 adam, 并判断是否可用"""
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    return optimizer_class


def _init_datasloader(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    train_dataset = StableDiffusionDataset(args.root_dir)
    train_dataset.reinitialize()
    train_dataset.set_transform(img_transform(args.resolution), dpt_transform(args.resolution))
    train_dataset.set_tokenizer(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, pin_memory=True
    )
    return train_dataloader


def _init_modules(args, accelerator_project_config, processor_class, optimizer_class):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    processor = processor_class(args.pretrained_model, args.revision, args.control_model)
    optimizer = optimizer_class(
        processor.unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, lr_scheduler = accelerator.prepare(processor.unet, optimizer, lr_scheduler)
    processor.unt = unet  # 记得更新一下
    return accelerator, processor, optimizer, lr_scheduler


def _enable_xformers_memory_efficient_attention(img_processor, dpt_processor):
    """判断 xformers 是否可以启用, 如果可以则启用, 否则则抛出异常"""
    from diffusers.utils.import_utils import is_xformers_available

    if is_xformers_available():
        import xformers
        from packaging import version

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warning(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please"
                " update xFormers to at least 0.0.17. See"
                " https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
            img_processor.enable_xformers_memory_efficient_attention()
            dpt_processor.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")


def _resume_from_checkpoint(checkpoint, output_dir, img_processor, dpt_processor):
    if checkpoint != "latest":
        path = os.path.basename(checkpoint)
    else:
        dirs = os.listdir(output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    initial_global_step = 0
    if path is None:
        print(f"Checkpoint {checkpoint} does not exist. Starting a new training run.")
        return
    else:
        print(f"Resuming from checkpoint {path}")
        # img_processor.load_state(os.path.join())
        # TODO:


def _remove_checkpoints(output_dir, checkpoints_total_limit):
    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)


def _save_checkpoint(output_dir, accelerator, global_step, postfix):
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}-{postfix}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    optimizer_class = _get_optimizer_class(args)

    """模型数据集部分"""
    train_dataloader = _init_datasloader(args)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epochs * num_update_steps_per_epoch
        args.train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    """加载模型"""
    img_accelerator, img_processor, img_optimizer, img_lr_scheduler = _init_modules(
        args, accelerator_project_config, ImgProcessor, optimizer_class
    )
    dpt_accelerator, dpt_processor, dpt_optimizer, dpt_lr_scheduler = _init_modules(
        args, accelerator_project_config, DptProcessor, optimizer_class
    )

    """配置 accelerator """
    if torch.backends.mps.is_available():
        img_accelerator.native_amp, dpt_accelerator.native_amp = False, False
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    weight_dtype = torch.float32
    if img_accelerator.mixed_precision == "fp16" and dpt_accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif img_accelerator.mixed_precision == "bf16" and dpt_accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    img_processor.set_weight_type(weight_dtype)
    dpt_processor.set_weight_type(weight_dtype)

    """显存优化部分"""
    if args.enable_xformers_memory_efficient_attention:
        _enable_xformers_memory_efficient_attention(img_processor, dpt_processor)

    if args.gradient_checkpointing:
        img_processor.enable_gradient_checkpointing()
        dpt_processor.enable_gradient_checkpointing()

    """移动模型到指定设备上"""
    img_processor.to(img_accelerator.device, dtype=weight_dtype)
    dpt_processor.to(dpt_accelerator.device, dtype=weight_dtype)

    initial_global_step = 0
    global_step = 0
    # 从 OmegaConf 读取的配置文件中如果直接修改里面的值会变成 str
    progress_bar = tqdm(
        range(0, int(args.max_train_steps)),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not (img_accelerator.is_local_main_process and dpt_accelerator.is_main_process),
    )
    first_epoch = 0
    for epoch in range(first_epoch, args.train_epochs):
        for step, batch in enumerate(train_dataloader):
            with img_accelerator.accumulate(img_processor.unet), dpt_accelerator.accumulate(dpt_processor.unet):
                img_noise, img_noise_pred = img_processor(
                    batch["img"].to(img_accelerator.device), batch["img_inputs_ids"].to(img_accelerator.device)
                )
                dpt_noise, dpt_noise_pred = dpt_processor(
                    batch["dpt"].to(dpt_accelerator.device), batch["dpt_inputs_ids"].to(dpt_accelerator.device)
                )

                # 计算损失与优化部分
                img_loss = F.mse_loss(img_noise_pred.float(), img_noise.float(), reduction="mean")
                dpt_loss = F.mse_loss(dpt_noise_pred.float(), dpt_noise.float(), reduction="mean")

                img_accelerator.backward(img_loss)
                dpt_accelerator.backward(dpt_loss)

                # 多机器训练部分
                if img_accelerator.sync_gradients and dpt_accelerator.sync_gradients:
                    img_accelerator.clip_grad_norm_(img_processor.unet.parameters(), args.max_grad_norm)
                    dpt_accelerator.clip_grad_norm_(dpt_processor.unet.parameters(), args.max_grad_norm)

                img_optimizer.step()
                img_lr_scheduler.step()
                img_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                dpt_optimizer.step()
                dpt_lr_scheduler.step()
                dpt_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if img_accelerator.sync_gradients and dpt_accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if img_accelerator.is_main_process and dpt_accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            _remove_checkpoints(args.output_dir, args.checkpoints_total_limit)
                        _save_checkpoint(args.output_dir, img_accelerator, global_step, "img")
                        _save_checkpoint(args.output_dir, dpt_accelerator, global_step, "dpt")

            img_logs = {
                "img_loss": img_loss.detach().item(),
                "img_lr": img_lr_scheduler.get_last_lr()[0],
            }
            dpt_logs = {
                "dpt_loss": dpt_loss.detach().item(),
                "dpt_lr": dpt_lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**{**img_logs, **dpt_logs})
            img_accelerator.log(img_logs, step=global_step)
            dpt_accelerator.log(dpt_logs, step=global_step)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load("config.yaml")
    main(args)
