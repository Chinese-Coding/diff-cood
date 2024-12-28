from torch import Tensor
import einops
from modules.base_processor import BaseProcessor
import torch


class ImgProcessor(BaseProcessor):
    def forward(self, imgs: Tensor, inputs_ids: Tensor):
        imgs = einops.rearrange(imgs, "b n c h w -> b * n c h w").to(dtype=self.weight_dtype)
        latents = self.vae.encode(imgs).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timestamps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                   device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timestamps).to(
            dtype=self.weight_dtype)
        encoder_hidden_stats = self.text_encoder(inputs_ids, return_dict=False)[0]
        noise_pred = \
        self.unet(noisy_latents, timestamps, encoder_hidden_states=encoder_hidden_stats, return_dict=False)[0]
        return noise, noise_pred
