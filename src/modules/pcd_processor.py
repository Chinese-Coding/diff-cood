import torch

from modules.base_processor import BaseProcessor
from modules.layering_unet_2d_condition import LayeringUNet2DCParams


class PcdProcessor(BaseProcessor):

    @torch.no_grad()
    def prepare(self, pcd: torch.Tensor, inputs_ids: torch.Tensor, all_return_tuple: bool = True):
        noise, noisy_latents, timestep, encoder_hidden_states = super().prepare(pcd.to(dtype=self.weight_dtype), inputs_ids)
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
