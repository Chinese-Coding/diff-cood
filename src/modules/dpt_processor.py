from modules.base_processor import BaseProcessor
import torch


class DptProcessor(BaseProcessor):
    def forward(self, dpt):
        latents = self.vae.encode(dpt).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timestamps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                   device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timestamps).to(
            dtype=self.weight_dtype)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            latents,
            controlnet_cond=dpt.to(dtype=self.weight_dtype),
            return_dict=False,
        )
        noise_pred = self.unet(
            noisy_latents,
            timestamps,
            down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
            return_dict=False,
        )[0]
        return noise, noise_pred
