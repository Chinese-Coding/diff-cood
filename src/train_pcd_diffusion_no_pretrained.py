"""
不使用预训练权重, 只是使用 diffusers 提供的模型进行训练
"""

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from loguru import logger
from tqdm.auto import tqdm

from diffusion_utils import get_optimizer_class, init_dataloader, init_logging, load_diffusion_modules2, save_modules
from modules.prepare_processpr import PrepareProcessor
from src.modules.detection_unet_2d_condition import DetectionUNet2DConditionModel


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        logger.warning(f"{path} 不存在, 将创建文件夹")
        os.makedirs(path)


def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    if not os.path.exists(args.output_dir):
        logger.warning(f"{args.output_dir} 不存在, 将创建文件夹及其下属的子文件夹")
        os.makedirs(args.output_dir)
    OmegaConf.save(args, os.path.join(args.output_dir, "config.yaml"))
    logger.success(f"将配置文件保存到 {args.output_dir} 目录中的 config.yaml 中.")

    writer = init_logging(args)
    optimizer_class = get_optimizer_class(args)
    train_dataloader = init_dataloader(args, args.lift_splat_shoot_args.data_aug_conf)

    """
    加载模型 (标注提示信息, 方便 IDE 提示)
    简写说明: Img for Image (图像); Pcd for point cloud (点云, 缩写成三个字母, 而不是两个字母的 pc, 主要是为了和图像的缩写保持同样的长度, 这样看起来比较方便)
    """
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    # resnet_out_scale_factor=0.5 conv_in_kernel=1, conv_out_kernel=1
    pcd_unet = DetectionUNet2DConditionModel(
        in_channels=3,
        out_channels=3,
        cross_attention_dim=1024,
    )  # 要和 encoder_hidden_states 的大小保持一致
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

    prepare_processor.set_requires_grad_(False)
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
    device = torch.device("cuda:1")
    with torch.cuda.device(device):
        if args.enable_xformers_memory_efficient_attention:
            pcd_unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        pcd_unet.enable_gradient_checkpointing()

    prepare_processor.to(device, weight_dtype)
    pcd_unet.to(device, dtype=weight_dtype)
    # 需要显式地将优化器的状态迁移到目标设备 (没想到这么复杂原本以为只要模型移动到目标设备就能正常用了)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    prepare_processor.requires_grad_(False)
    pcd_unet.train()

    global_step = (first_epoch - 1) * len(train_dataloader) / args.batch_size if first_epoch > 0 else 0
    logger.success(f"从 {first_epoch} 开始训练, 共训练 {args.train_epoches} 个 epoch")

    for epoch in range(first_epoch, args.train_epoches):
        logger.success(f"第 {epoch} 个 epoch 开始训练")
        progress_bar = tqdm(range(0, len(train_dataloader)), initial=0, desc=f"Epoch: {epoch}/{args.train_epoches}")
        for step, batch in enumerate(train_dataloader):
            latents = batch["pcd"].to(device, dtype=weight_dtype)
            noise = torch.randn_like(latents)  # 训练 `prepare_processor.num_train_timesteps` 前, 计算出 noise 的形状
            bsz = latents.shape[0]
            timesteps = prepare_processor.generate_timestep(bsz, device).long()
            encoder_hidden_states = prepare_processor.text_encoder(batch["pcd_inputs_ids"].to(device), return_dict=False)[0]
            noisy_latents = prepare_processor.add_noise(latents, noise, timesteps)
            logger.debug(f"{noisy_latents.shape=}, {encoder_hidden_states.shape=}")
            internal_sample = {}
            model_pred = pcd_unet(noisy_latents, timesteps, encoder_hidden_states, internal_sample=internal_sample)[0]

            """从原版那边又抄过来的代码, 每次都从 `prepare_processor` 里面拿东西, 看着比较奇怪, 先这样写着"""
            if prepare_processor.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif prepare_processor.noise_scheduler.config.prediction_type == "v_prediction":
                target = prepare_processor.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {prepare_processor.noise_scheduler.config.prediction_type}")

            pcd_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            pcd_loss.backward()
            torch.nn.utils.clip_grad_norm_(pcd_unet.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            pcd_logs = {"pcd_loss": pcd_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            writer.add_scalar("Loss/pcd_loss", pcd_loss.detach().item(), global_step)

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**pcd_logs)
        if args.save_freq != -1 and epoch % args.save_freq == 0:
            save_modules(args.output_dir, epoch, pcd_unet, optimizer, lr_scheduler, "pcd")

    writer.close()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_diffusion.yaml"))
    args.output_dir = os.path.expanduser("~/Desktop/logs/pcd_diffusion_2025_03_01")
    args.batch_size = 1
    args.num_workers = 4
    args.train_epoches = 30
    main(args)
