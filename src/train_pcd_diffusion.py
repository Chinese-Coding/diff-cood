"""不使用 accelerate, 同时进行某一层交换的 train diffusion 函数"""

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from loguru import logger
from tqdm.auto import tqdm

from diffusion_utils import get_optimizer_class, init_dataloader, init_logging, load_diffusion_modules2, save_modules
from modules.layering_unet_2dc_model import LayeringUNet2DCModel
from modules.prepare_processpr import PrepareProcessor


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    writer = init_logging(args)

    optimizer_class = get_optimizer_class(args)
    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

    """
    加载模型 (标注提示信息, 方便 IDE 提示)
    简写说明: Img for Image (图像); Pcd for point cloud (点云, 缩写成三个字母, 而不是两个字母的 pc, 主要是为了和图像的缩写保持同样的长度, 这样看起来比较方便)
    """
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    pcd_unet: LayeringUNet2DCModel = LayeringUNet2DCModel.from_pretrained(
        args.pretrained_model, subfolder="unet", revision=args.revision
    )
    optimizer = optimizer_class(
        pcd_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    pcd_unet.train()

    """加载权重 (上面那个是预训练权重, 下面这个是自己的权重)"""
    first_epoch, pcd_loss = 0, 0  # 为保存权重特地将变量声明到前面
    if "resume_file" in args:
        resume_file = os.path.expanduser(args.resume_file)
        checkpoint = load_diffusion_modules2(f"{resume_file}-pcd.pth", pcd_unet, optimizer, lr_scheduler)
        first_epoch = checkpoint["epoch"] + 1
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
    prepare_processor.set_weight_dtype(weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        pcd_unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        pcd_unet.enable_gradient_checkpointing()

    device = torch.device("cuda:0")
    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)

    global_step = (first_epoch - 1) * len(train_dataloader) / args.batch_size if first_epoch > 0 else 0
    logger.success(f"从 {first_epoch} 开始训练, 共训练 {args.train_epoches} 个 epoch")

    for epoch in range(first_epoch, args.train_epoches):
        logger.success(f"第 {epoch} 个 epoch 开始训练")
        progress_bar = tqdm(range(0, int(len(train_dataloader) * prepare_processor.num_train_timesteps)), initial=0, desc=f"Epoch: {epoch}/{args.train_epoches}")
        for step, batch in enumerate(train_dataloader):
            logger.success(f"第 {step} 个图片开始训练")
            latents = prepare_processor.get_latents(batch["pcd"].to(device, dtype=weight_dtype))
            noise = torch.randn_like(latents)  # 训练 `prepare_processor.num_train_timesteps` 前, 计算出 noise 的形状
            """对每个点云都训练 `num_train_timesteps` 次"""
            for t in range(prepare_processor.num_train_timesteps):
                pcd_noise, pcd_params = prepare_processor.prepare(
                    batch["pcd"].to(device, dtype=weight_dtype), batch["pcd_inputs_ids"].to(device), False, t, noise
                )
                pcd_params.to(device)
                pcd_params = pcd_unet.forward_control(pcd_unet.forward_down(pcd_unet.forward_pre(pcd_params)))
                """交换 (省略)"""
                pcd_noise_pred = pcd_unet.forward_up(pcd_unet.forward_middle(pcd_params)).sample

                """计算损失, 开始反向传播"""
                pcd_loss = F.mse_loss(pcd_noise_pred.float(), pcd_noise.float(), reduction="mean")

                pcd_loss.backward()

                torch.nn.utils.clip_grad_norm_(pcd_unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                pcd_logs = {
                    "pcd_loss": pcd_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }

                writer.add_scalar("Loss/pcd_loss", pcd_loss.detach().item(), global_step)

                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(**pcd_logs)

        save_modules(args.output_dir, epoch, pcd_unet, optimizer, lr_scheduler, "pcd")

    writer.close()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_diffusion.yaml"))
    args.output_dir = os.path.expanduser("~/Desktop/logs/pcd_diffusion_2025_01_25")
    args.batch_size = 16
    args.num_workers = 16
    args.train_epoches = 30
    main(args)
