import einops
import torch
from torch import Tensor

from modules.base_processor import BaseProcessor


class ImgProcessor(BaseProcessor):
    @torch.no_grad()
    def prepare(self, imgs: Tensor, inputs_ids: Tensor):
        imgs = einops.rearrange(imgs, "b n c h w -> (b n) c h w").to(dtype=self.weight_dtype)
        return super().prepare(imgs, inputs_ids)

    def forward(self, imgs: Tensor, inputs_ids: Tensor):
        noise, noisy_latents, timestamps, encoder_hidden_states = self.prepare(imgs, inputs_ids)
        noise_pred = self.unet(noisy_latents, timestamps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        return noise, noise_pred
