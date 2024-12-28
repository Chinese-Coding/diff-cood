from torch.utils.data import DataLoader
from transformers import AutoTokenizer
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

from modules.dpt_processor import DptProcessor
from modules.img_processor import ImgProcessor

logger = get_logger(__name__)


def _init_modules(args, accelerator_project_config, processor_class, optimizer_class):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    processor = processor_class(args.pretrained_model, args.revision)
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


def main(args):
    """加载模型 (TODO: 一些不需要训练的模型能共用一个吗?)"""
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
    # 加载模型
    img_accelerator, img_processor, img_optimizer, img_lr_scheduler = _init_modules(
        args, accelerator_project_config, ImgProcessor, optimizer_class
    )
    dpt_accelerator, dpt_processor, dpt_optimizer, dpt_lr_scheduler = _init_modules(
        args, accelerator_project_config, DptProcessor, optimizer_class
    )

    weight_dtype = torch.float32
    if img_accelerator.mixed_precision == "fp16" and dpt_accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif img_accelerator.mixed_precision == "bf16" and dpt_accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    img_processor.set_weight_type(weight_dtype)
    dpt_processor.set_weight_type(weight_dtype)

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
                img_processor.enable_xformers_memory_efficient_attention()
                dpt_processor.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    """移动模型到指定设备上"""
    img_processor.to(img_accelerator.device, dtype=weight_dtype)
    dpt_processor.to(dpt_accelerator.device, dtype=weight_dtype)

    initial_global_step = 0
    # progress_bar = tqdm(
    #     range(0, args.max_train_steps),
    #     initial=initial_global_step,
    #     desc="Steps",
    # )
    first_epoch = 0
    for epoch in range(first_epoch, args.train_epochs):
        for step, batch in enumerate(train_dataloader):
            with img_accelerator.accumulate(img_processor.unet), dpt_accelerator.accumulate(dpt_processor.unet):
                img_noise, img_noise_pred = img_processor(batch["img"], batch["inputs_ids"])
                dpt_noise, dpt_noise_pred = dpt_processor(batch["dpt"])

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
