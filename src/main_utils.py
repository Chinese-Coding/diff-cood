import os
import shutil

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_related.stable_diffusion_dataset import StableDiffusionDataset
from data_related.transform_funs import img_transform, pcd_transform

logger = get_logger(__name__)


def get_optimizer_class(args):
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


def init_datasloader(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    train_dataset = StableDiffusionDataset(args.root_dir)
    train_dataset.reinitialize()
    train_dataset.set_transform(img_transform(args.resolution), pcd_transform(args.resolution))
    train_dataset.set_tokenizer(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, pin_memory=True
    )
    return train_dataloader


def init_modules(args, processor_class, optimizer_class, accelerator_project_config=None):
    processor = processor_class(args.pretrained_model, args.revision, args.control_model, args.layering)
    optimizer = optimizer_class(
        processor.unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if accelerator_project_config is not None:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
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
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
        return processor, optimizer, lr_scheduler


def enable_xformers_memory_efficient_attention(img_processor, pcd_processor):
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
            pcd_processor.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")


def resume_from_checkpoint(checkpoint, output_dir, img_processor, pcd_processor):
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


def remove_checkpoints(output_dir, checkpoints_total_limit):
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


def save_checkpoint(output_dir, accelerator, global_step, postfix):
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}-{postfix}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")
