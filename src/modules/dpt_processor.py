import torch

from modules.base_processor import BaseProcessor
from modules.layering_unet_2d_condition import LayeringUNet2DCParams


class DptProcessor(BaseProcessor):

    @torch.no_grad()
    def prepare(self, dpt: torch.Tensor, inputs_ids: torch.Tensor, all_return_tuple: bool = True):
        noise, noisy_latents, timestep, encoder_hidden_states = super().prepare(dpt.to(dtype=self.weight_dtype), inputs_ids)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=dpt.to(dtype=self.weight_dtype),
            return_dict=False,
        )
        down_block_additional_residuals = [sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample.to(dtype=self.weight_dtype)
        if all_return_tuple:
            return noise, noisy_latents, timestep, encoder_hidden_states, down_block_additional_residuals, mid_block_res_sample
        else:
            return noise, LayeringUNet2DCParams(
                sample=noisy_latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_res_sample,
            )

    def forward(self, dpt: torch.Tensor, inputs_ids: torch.Tensor):
        noise, noisy_latents, timestep, encoder_hidden_states, down_block_res_samples, mid_block_res_sample = self.prepare(
            dpt, inputs_ids
        )
        noise_pred = self.unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
            return_dict=False,
        )[0]
        return noise, noise_pred
