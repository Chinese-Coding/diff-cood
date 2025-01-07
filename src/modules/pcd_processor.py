import einops
import torch

from modules.base_processor import BaseProcessor
from modules.layering_unet_2dc_model import LayeringUNet2DCParams


class PcdProcessor(BaseProcessor):

    @torch.no_grad()
    def prepare(self, pcds: torch.Tensor, inputs_ids: torch.Tensor, all_return_tuple: bool = True, t: int = -1):
        # 如果 pcds 的维度是五维, 也就是多了一个分割点云的维度 (下面的 n), 就把 n 合并到 batch size 那一维度
        # 增加这一判断旨在避免反复修改输入而造成该处代码反复修改
        if pcds.dim() == 5:
            pcds = einops.rearrange(pcds, "b n c h w -> (b n) c h w")
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
