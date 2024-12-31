from typing import Optional

import torch
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from torch import nn
from transformers import PretrainedConfig

from modules.layering_unet_2d_condition import LayeringUNet2dConditionModel


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
    def __init__(self, pretrained_model: str, revision: str, controlnet_model: Optional[str] = None, layering=False):
        super().__init__()
        text_encoder_cls = import_model_class_from_pretrained_model(pretrained_model, revision)
        unet_class = LayeringUNet2dConditionModel if layering else UNet2DConditionModel

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision)
        self.unet = unet_class.from_pretrained(pretrained_model, subfolder="unet", revision=revision)
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

    def to(self, device, dtype, need_unet: bool = False):
        """
        为什么 `unet` 不迁移呢? 因为 unet 使用 accelerator 进行管理 (参考代码至少是这样的)
        现在想要通过 torch 进行改写, 所以加一个标志位用于全部移动
        """
        self.vae.to(device, dtype)
        self.controlnet.to(device, dtype)
        self.text_encoder.to(device, dtype)
        if need_unet:
            self.unet.to(device, dtype)

    def set_weight_dtype(self, weight_dtype: torch.dtype):
        self.weight_dtype = weight_dtype

    @torch.no_grad()
    def prepare(self, x: torch.Tensor, inputs_ids: torch.Tensor):
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timestep).to(dtype=self.weight_dtype)
        encoder_hidden_states = self.text_encoder(inputs_ids, return_dict=False)[0]
        return noise, noisy_latents, timestep, encoder_hidden_states
