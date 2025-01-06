import torch

from modules.base_processor import BaseProcessor
from modules.layering_unet_2dc_model import LayeringUNet2DCParams

import einops


class PcdProcessor(BaseProcessor):

    @torch.no_grad()
    def prepare(self, pcds: torch.Tensor, inputs_ids: torch.Tensor, all_return_tuple: bool = True, t: int = -1):
        pcds = einops.rearrange(pcds, "b n c h w -> (b n) c h w").to(dtype=self.weight_dtype)
        noise, noisy_latents, timestep, encoder_hidden_states = super().prepare(
            pcds.to(dtype=self.weight_dtype), inputs_ids, t=t
        )
        if all_return_tuple:
            return noise, noisy_latents, timestep, encoder_hidden_states
        else:
            return noise, LayeringUNet2DCParams(
                sample=noisy_latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )

    def forward(self, pcd: torch.Tensor, inputs_ids: torch.Tensor):
        noise, noisy_latents, timestep, encoder_hidden_states = self.prepare(pcd, inputs_ids)
        noise_pred = self.unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        return noise, noise_pred
