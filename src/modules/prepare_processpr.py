from typing import Optional

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from modules.layering_unet_2dc_model import LayeringUNet2DCParams
from torch import nn
from transformers import CLIPTextModel


class PrepareProcessor(nn.Module):
    def __init__(self, pretrained_model: str, revision: str):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae", revision=revision)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler", revision=revision)
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def to(self, device, dtype):
        self.vae.to(device, dtype)
        self.text_encoder.to(device, dtype)

    def set_weight_dtype(self, weight_dtype: torch.dtype):
        self.weight_dtype = weight_dtype

    def generate_timestep(self, bsz, device):
        return torch.randint(0, self.num_train_timesteps, (bsz,), device=device)

    @torch.no_grad()
    def prepare(
        self,
        x: torch.Tensor,
        inputs_ids: torch.Tensor,
        all_return_tuple: bool = True,
        t: int = -1,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        :param all_return_tuple: 是否将全部返回值以 Tuple 的形式返回
        :param t: 用于推理时指定时刻
        """
        latents = self.get_latents(x)
        noise = torch.randn_like(latents) if noise is None else noise
        batch_size = latents.shape[0]
        timestep = torch.randint(0, self.num_train_timesteps, (batch_size,)) if t == -1 else torch.full((batch_size,), t)
        timestep = timestep.to(device=latents.device, dtype=torch.long)
        noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timestep).to(dtype=self.weight_dtype)
        encoder_hidden_states = self.text_encoder(inputs_ids, return_dict=False)[0]
        if all_return_tuple:
            return noise, noisy_latents, timestep, encoder_hidden_states
        else:
            return noise, LayeringUNet2DCParams(
                sample=noisy_latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )

    @torch.no_grad()
    def get_latents(self, x: torch.Tensor):
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
        return self.noise_scheduler.add_noise(latents, noise, timestep).to(dtype=self.weight_dtype)

    def set_requires_grad_(self, requires_grad=False):
        self.vae.requires_grad_(requires_grad)
        self.text_encoder.requires_grad_(requires_grad)
