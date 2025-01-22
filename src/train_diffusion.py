"""不使用 accelerate, 同时进行某一层交换的 train diffusion 函数"""

import torch

from modules.prepare_processpr import PrepareProcessor


torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm.auto import tqdm

from data_related.entity import LiftSplatShootParams
from diffusion_utils import (
    get_change_fun,
    get_optimizer_class,
    init_datasloader,
    init_logging,
    load_diffusion_modules2,
    save_modules2,
)
from modules.layering_unet_2dc_model import LayeringUNet2DCModel
from opencood.models.lift_splat_shoot import LiftSplatShoot
from diffusers.optimization import get_scheduler

def main(args):
    # 路径展开
    args.output_dir = os.path.expanduser(args.output_dir)
    args.pretrained_model = os.path.expanduser(args.pretrained_model)

    writer = init_logging(args)

    _change = get_change_fun(args.change_args)
    optimizer_class = get_optimizer_class(args)
    train_dataloader = init_datasloader(args, args.lift_splat_shoot_args.data_aug_conf)
    
    prepare_processor = PrepareProcessor(args.pretrained_model, args.revision)
    img_unet: LayeringUNet2DCModel = LayeringUNet2DCModel.from_pretrained(args.pretrained_model, subfolder="unet", revision=args.revision)
    pcd_unet: LayeringUNet2DCModel = LayeringUNet2DCModel.from_pretrained(args.pretrained_model, subfolder="unet", revision=args.revision)
    optimizer  = optimizer_class(
        list(img_unet.parameters()) +  list(pcd_unet.parameters()),
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

    img_unet.train()
    pcd_unet.train()
    

    """加载权重 (上面那个是预训练权重, 下面这个是自己的权重)"""
    first_epoch, img_loss, pcd_loss = 0, 0, 0  # 为保存权重特地将变量声明到前面
    if "resume_file" in args:
        resume_file = os.path.expanduser(args.resume_file)
        checkpoint = load_diffusion_modules2(resume_file, img_unet, pcd_unet, lr_scheduler)
        first_epoch = checkpoint["epoch"] + 1
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
    prepare_processor.set_weight_dtype(weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        img_unet.enable_xformers_memory_efficient_attention()
        pcd_unet.enable_xformers_memory_efficient_attention()
    
    if args.gradient_checkpointing:
        img_unet.enable_gradient_checkpointing()
        pcd_unet.enable_gradient_checkpointing()

    device = torch.device("cuda:1")

    prepare_processor.to(device, weight_dtype)
    img_unet.to(device=device, dtype=weight_dtype)
    pcd_unet.to(device=device, dtype=weight_dtype)
    """LiftSplatShoot 模型和对应的降低通道数的卷积层"""
    lss_model = LiftSplatShoot(args.lift_splat_shoot_args, device)
    lss_model.load_state_dict(
        torch.load(os.path.expanduser(args.lift_splat_shoot_args.pretrained_model_path), weights_only=False), strict=False
    )
    lss_model.eval()
    lss_model.to(device=device)
    bottleneck_layer = nn.Conv2d(128, 3, kernel_size=1)
    # 增加了一个新的卷积层, 别忘了把他添加到 optimizer 里面
    optimizer.add_param_group({"params": bottleneck_layer.parameters()})

    # 这段代码是后面添加的, 为了不改变原来函数的调用接口, 这里再单独做一个判断,
    # 可能不够简洁高效, 但是开发周期短, 先这么将就一下
    if "resume_file" in args:
        bottleneck_layer.load_state_dict(checkpoint["bottleneck_layer"])
        # 这里需要把 optimizer 的参数也加载进来, 因为此时的 img_optimizer 里面已经有了 bottleneck_layer 的权重
        optimizer.load_state_dict(checkpoint["optimizer"])

    bottleneck_layer.train()
    bottleneck_layer.to(device=device)

    # 这里假定每次训练的 batch size 相同, 这样就可以计算出前面已经训练了多少步
    global_step = (first_epoch - 1) * len(train_dataloader) / args.batch_size if first_epoch > 1 else 0
    logger.success(f"从 {first_epoch} 开始训练, 共训练 {args.train_epoches} 个 epoch")

    for epoch in range(first_epoch, args.train_epoches):
        logger.success(f"第 {epoch} 个 epoch 开始训练")
        progress_bar = tqdm(range(0, int(len(train_dataloader))), initial=0, desc=f"Epoch: {epoch}/{args.train_epoches}")
        for step, batch in enumerate(train_dataloader):
            """预处理图像, 把图像处理为 BEV 图"""
            lss_params: LiftSplatShootParams = batch["lss_params"]
            lss_params.to(device)
            img = lss_model(
                lss_params.imgs, lss_params.rots, lss_params.trans, lss_params.intrins, lss_params.post_rots, lss_params.post_trans # fmt: skip
            )
            img = bottleneck_layer(img)  # 降低通道数 (128 -> 3)
            logger.debug(f"获得的输入 diffusion 的 shape: img({img.shape}) pcd({batch['pcd'].shape})")

            """
            现在 img 和 pcd 的 shape 都是: (B, 3, 512, 512)
            只需要修改雷达监测范围以及生成体素的粒度就能改变 BEV 图的分辨率吗?
            """
            timestep = prepare_processor.generate_timestep().item()
            
            """处理图像"""
            img_noise, img_params = prepare_processor.prepare(
                img.to(device, dtype=weight_dtype), batch["img_inputs_ids"].to(device), False, timestep
            )
            pcd_noise, pcd_params = prepare_processor.prepare(
                batch["pcd"].to(device, dtype=weight_dtype), batch["pcd_inputs_ids"].to(device), False, timestep, noise=img_noise
            )
            assert torch.allclose(img_noise, pcd_noise, atol=1e-6) # 如果给出的时间片相同的话, 两者生成的噪声应该是一致的
            noise = img_noise
            img_params.to(device)
            pcd_params.to(device)

            img_params = img_unet.forward_control(img_unet.forward_down(img_unet.forward_pre(img_params)))
            pcd_params = pcd_unet.forward_control(pcd_unet.forward_down(pcd_unet.forward_pre(pcd_params)))

            """交换空间 (这些东西以后写成超参数)"""
            img_sample, pcd_sample = img_params.sample.to("cpu"), pcd_params.sample.to("cpu")
            logger.debug(f"获得的中间层 Tensor 的 shape: img: {img_sample.shape}, pcd: {pcd_sample.shape}")
            img_params.sample, pcd_params.sample = _change(img_sample, pcd_sample)

            """交换完之后的步骤, 开始走没走完的层"""
            img_params.to(device), pcd_params.to(device)
            img_noise_pred, pcd_noise_pred = (
                img_unet.forward_up(img_unet.forward_middle(img_params)).sample,
                pcd_unet.forward_up(pcd_unet.forward_middle(pcd_params)).sample,
            )  # 犹豫再三还是卸载了一行里面 (虽然会被 black 格式化成 4 行)

            """计算损失, 开始反向传播"""
            img_loss, pcd_loss = (
                F.mse_loss(img_noise_pred.float(), noise.float(), reduction="mean"),
                F.mse_loss(pcd_noise_pred.float(), noise.float(), reduction="mean"),
            )  # 犹豫再三还是卸载了一行里面 (虽然会被 black 格式化成 4 行)

            # 检查一下这两个损失是否为 NaN
            total_loss = img_loss + pcd_loss
            total_loss.backward()

            # 梯度裁剪, 防止梯度保障, 参考的训练 diffusion 的脚本里有, 之前忘记加上了
            torch.nn.utils.clip_grad_norm_(img_unet.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(pcd_unet.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            logs = {
                "img_loss": img_loss.detach().item(),
                "pcd_loss": pcd_loss.detach().item(),
                "total_loss": total_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            # 记录损失到 TensorBoard
            writer.add_scalar("Loss/img_loss", logs["img_loss"], global_step)
            writer.add_scalar("Loss/pcd_loss", logs["pcd_loss"], global_step)
            writer.add_scalar("Loss/total_loss", logs["total_loss"], global_step)

            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**logs)

        # fmt: off
        save_modules2(
            args.output_dir, epoch, img_unet,pcd_unet, optimizer, lr_scheduler,
            bottleneck_layer=bottleneck_layer
        )
        # fmt: on

    writer.close()


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_diffusion.yaml"))
    args.output_dir = os.path.expanduser("~/Desktop/logs/diffusion_2025_01_18")
    args.batch_size = 2
    main(args)
