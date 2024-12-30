import torch

from modules.base_processor import BaseProcessor


class DptProcessor(BaseProcessor):
    def forward(self, dpt, inputs_ids):
        dpt = dpt.to(dtype=self.weight_type)
        latents = self.vae.encode(dpt).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timestamps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timestamps).to(dtype=self.weight_type)

        encoder_hidden_states = self.text_encoder(inputs_ids, return_dict=False)[0]
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timestamps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=dpt.to(dtype=self.weight_type),
            return_dict=False,
        )
        noise_pred = self.unet(
            noisy_latents,
            timestamps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[sample.to(dtype=self.weight_type) for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_type),
            return_dict=False,
        )[0]
        return noise, noise_pred
