import math
import os.path
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm

from diffusion_utils import (
    enable_xformers_memory_efficient_attention,
    get_optimizer_class,
    init_datasloader,
    init_modules,
    remove_checkpoints,
    save_checkpoint,
)
from modules.img_processor import ImgProcessor
from modules.Pcd_processor import PcdProcessor

logger = get_logger(__name__)


def main(args):
    args.output_dir = os.path.expanduser(args.output_dir)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    optimizer_class = get_optimizer_class(args)

    """模型数据集部分"""
    train_dataloader = init_datasloader(args)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.train_epoches * num_update_steps_per_epoch
        args.train_epoches = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    """加载模型"""
    img_accelerator, img_processor, img_optimizer, img_lr_scheduler = init_modules(
        args, ImgProcessor, optimizer_class, accelerator_project_config
    )
    pcd_accelerator, pcd_processor, pcd_optimizer, pcd_lr_scheduler = init_modules(
        args, PcdProcessor, optimizer_class, accelerator_project_config
    )

    """配置 accelerator """
    if torch.backends.mps.is_available():
        img_accelerator.native_amp, pcd_accelerator.native_amp = False, False
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    # 这个 `mixed_precision` 在 `init_modules` 里面设置的
    weight_dtype = torch.float32
    if img_accelerator.mixed_precision == "fp16" and pcd_accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif img_accelerator.mixed_precision == "bf16" and pcd_accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f"将 weight_dtype 设置为 {weight_dtype}")
    img_processor.set_weight_dtype(weight_dtype)
    pcd_processor.set_weight_dtype(weight_dtype)

    """显存优化部分"""
    if args.enable_xformers_memory_efficient_attention:
        enable_xformers_memory_efficient_attention(img_processor, pcd_processor)

    if args.gradient_checkpointing:
        img_processor.enable_gradient_checkpointing()
        pcd_processor.enable_gradient_checkpointing()

    """移动模型到指定设备上"""
    img_processor.to(img_accelerator.device, dtype=weight_dtype)
    pcd_processor.to(pcd_accelerator.device, dtype=weight_dtype)

    initial_global_step = 0
    global_step = 0
    # 从 OmegaConf 读取的配置文件中如果直接修改里面的值会变成 str
    progress_bar = tqdm(
        range(0, int(args.max_train_steps)),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not (img_accelerator.is_local_main_process and pcd_accelerator.is_main_process),
    )
    first_epoch = 0
    for epoch in range(first_epoch, args.train_epoches):
        for step, batch in enumerate(train_dataloader):
            with img_accelerator.accumulate(img_processor.unet), pcd_accelerator.accumulate(pcd_processor.unet):
                img_noise, img_noise_pred = img_processor(
                    batch["img"].to(img_accelerator.device), batch["img_inputs_ids"].to(img_accelerator.device)
                )
                pcd_noise, pcd_noise_pred = pcd_processor(
                    batch["pcd"].to(pcd_accelerator.device), batch["pcd_inputs_ids"].to(pcd_accelerator.device)
                )

                # 计算损失与优化部分
                img_loss = F.mse_loss(img_noise_pred.float(), img_noise.float(), reduction="mean")
                pcd_loss = F.mse_loss(pcd_noise_pred.float(), pcd_noise.float(), reduction="mean")

                img_accelerator.backward(img_loss)
                pcd_accelerator.backward(pcd_loss)

                # 多机器训练部分
                if img_accelerator.sync_gradients and pcd_accelerator.sync_gradients:
                    img_accelerator.clip_grad_norm_(img_processor.unet.parameters(), args.max_grad_norm)
                    pcd_accelerator.clip_grad_norm_(pcd_processor.unet.parameters(), args.max_grad_norm)

                img_optimizer.step()
                img_lr_scheduler.step()
                img_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                pcd_optimizer.step()
                pcd_lr_scheduler.step()
                pcd_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if img_accelerator.sync_gradients and pcd_accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if img_accelerator.is_main_process and pcd_accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            remove_checkpoints(args.output_dir, args.checkpoints_total_limit)
                        save_checkpoint(args.output_dir, img_accelerator, global_step, "img")
                        save_checkpoint(args.output_dir, pcd_accelerator, global_step, "pcd")

            img_logs = {
                "img_loss": img_loss.detach().item(),
                "img_lr": img_lr_scheduler.get_last_lr()[0],
            }
            pcd_logs = {
                "pcd_loss": pcd_loss.detach().item(),
                "pcd_lr": pcd_lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**{**img_logs, **pcd_logs})
            img_accelerator.log(img_logs, step=global_step)
            pcd_accelerator.log(pcd_logs, step=global_step)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/config.yaml"))
    main(args)
