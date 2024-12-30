from typing import Optional

import torch
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from torch import nn
from transformers import PretrainedConfig


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


class BaseProcessor(nn.Module):
    def __init__(self, pretrained_model: str, revision: str, controlnet_model: Optional[str] = None):
        super().__init__()
        text_encoder_cls = import_model_class_from_pretrained_model(pretrained_model, revision)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet", revision=revision)
        self.text_encoder = text_encoder_cls.from_pretrained(pretrained_model, subfolder="text_encoder")

        self.controlnet = (
            ControlNetModel.from_pretrained(controlnet_model, revision=revision)
            if controlnet_model
            else ControlNetModel.from_unet(self.unet)
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler", revision=revision)

        self.vae.requires_grad_(False)
        self.unet.train()
        self.text_encoder.requires_grad_(False)
        self.controlnet.requires_grad_(False)

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()
        self.controlnet.enable_xformers_memory_efficient_attention()

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    def to(self, device, dtype):
        """为什么 `unet` 不迁移呢? 因为 unet 使用 accelerator 进行管理 (参考代码至少是这样的)"""
        self.vae.to(device, dtype)
        self.controlnet.to(device, dtype)
        self.text_encoder.to(device, dtype)

    def set_weight_type(self, weight_type: torch.dtype):
        self.weight_type = weight_type
