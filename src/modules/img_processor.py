import einops
import torch
from torch import Tensor

from modules.base_processor import BaseProcessor


class ImgProcessor(BaseProcessor):
    @torch.no_grad()
    def prepare(self, imgs: Tensor, inputs_ids: Tensor, all_return_tuple: bool = True, t: int = -1):
        # 如果 imgs 的维度是五维, 也就是多了一个多少个相机的维度 (下面的 n), 就把 n 合并到 batch size 那一维度
        # 增加这一判断旨在避免反复修改输入而造成该处代码反复修改
        if imgs.dim() == 5:
            imgs = einops.rearrange(imgs, "b n c h w -> (b n) c h w")
        return super().prepare(imgs.to(dtype=self.weight_dtype), inputs_ids, all_return_tuple, t)

    def forward(self, imgs: Tensor, inputs_ids: Tensor):
        noise, noisy_latents, timestamps, encoder_hidden_states = self.prepare(imgs, inputs_ids)
        noise_pred = self.unet(noisy_latents, timestamps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        return noise, noise_pred
