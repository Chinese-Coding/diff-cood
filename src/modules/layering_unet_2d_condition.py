from typing import Any, Dict, List, Optional, Tuple, Union

import diffusers
import torch
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from pydantic import BaseModel, ConfigDict


class LayeringUNet2DCParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample: torch.Tensor
    timestep: Union[torch.Tensor, float, int]
    encoder_hidden_states: torch.Tensor
    class_labels: Optional[torch.Tensor] = None
    timestep_cond: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
    down_block_additional_residuals: Optional[List[torch.Tensor]] = None  # 这里从 `Tuple` 变成了 `List`
    mid_block_additional_residual: Optional[torch.Tensor] = None
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    retrun_dict: bool = True

    # 这些参数是需要在这个给不同的 forward 方法中进行传递的中间参数, 原本 forward 方法自带的参数放在了上面
    forward_upsample_size: bool = False
    emb: Optional = None
    is_controlnet: bool = False
    is_adapter: bool = False
    lora_scale: float = 1.0
    down_block_res_samples: Optional[list] = None

    def to(self, device: torch.device):
        for attr, value in self.__dict__.items():
            # 将所有 torch.Tensor 类型的属性搬运到指定设备
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
            # 对于可能包含多个 torch.Tensor 的类型（如 Tuple 或 Dict），递归搬运
            elif isinstance(value, (tuple, list)):
                setattr(self, attr, type(value)(v.to(device) if isinstance(v, torch.Tensor) else v for v in value))
            elif isinstance(value, dict):
                setattr(self, attr, {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()})


class LayeringUNet2DCModel(diffusers.UNet2DConditionModel):
    """该类名中的 `UNet2DC` 是 `UNet2DCondition` 的简写, 上面两个 `Params` 同理"""

    def forward_pre(self, params: LayeringUNet2DCParams):
        default_overall_up_factor = 2**self.num_upsamplers
        sample = params.sample
        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                params.forward_upsample_size = True
                break

        attention_mask = params.attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
            params.attention_mask = attention_mask

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        encoder_attention_mask = params.encoder_attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            params.encoder_attention_mask = encoder_attention_mask

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=params.timestep)
        emb = self.time_embedding(t_emb, params.timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=params.class_labels)
        if class_emb is not None:
            emb = torch.cat([emb, class_emb], dim=-1) if self.config.class_embeddings_concat else emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=params.encoder_hidden_states, added_cond_kwargs=params.added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        params.encoder_hidden_states = self.process_encoder_hidden_states(
            params.encoder_hidden_states, params.added_cond_kwargs
        )
        params.sample, params.emb = sample, emb

        # 2.5 GLIGEN position net
        cross_attention_kwargs, lora_scale = params.cross_attention_kwargs, 1.0
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = params.cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
            if cross_attention_kwargs.get("gligen", None) is not None:
                gligen_args = cross_attention_kwargs.pop("gligen")
                cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}
        params.cross_attention_kwargs, params.lora_scale = cross_attention_kwargs, lora_scale
        """
        对 UNet 使用的控制条件进行判断 (改变了该代码原来的所在位置)
        `mid_block_additional_residual` 和 `down_block_additional_residuals` 均不为空说明使用 control net;
        `down_intrablock_additional_residuals` 不为空就说明使用 T2I
        还有一种废除的情况, 这里先把提示代码删除了, 选择直接抛出异常, 不然看着太长比较难受
        """
        params.is_controlnet = (
            params.mid_block_additional_residual is not None and params.down_block_additional_residuals is not None
        )
        params.is_adapter = params.down_intrablock_additional_residuals is not None
        if (
            not params.is_adapter
            and params.mid_block_additional_residual is None
            and params.down_block_additional_residuals is not None
        ):
            raise Exception("被废弃的用法")

        return params

    def forward_down(self, params: LayeringUNet2DCParams):
        sample = params.sample
        # 2. pre-process
        sample = self.conv_in(sample)

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, params.lora_scale)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if params.is_adapter and len(params.down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = params.down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=params.emb,
                    encoder_hidden_states=params.encoder_hidden_states,
                    attention_mask=params.attention_mask,
                    cross_attention_kwargs=params.cross_attention_kwargs,
                    encoder_attention_mask=params.encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=params.emb)
                if params.is_adapter and len(params.down_intrablock_additional_residuals) > 0:
                    sample += params.down_intrablock_additional_residuals.pop(0)
            down_block_res_samples += res_samples
        params.sample = sample
        params.down_block_res_samples = down_block_res_samples
        return params

    def forward_control(self, params: LayeringUNet2DCParams):
        """尽管说, IDE 给提示说这个方法, 可以变成函数, 但是为了统一写法这里就不提取出函数了"""
        if params.is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                params.down_block_res_samples, params.down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            params.down_block_res_samples = new_down_block_res_samples
        return params

    def forward_middle(self, params: LayeringUNet2DCParams):
        sample = params.sample
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    params.emb,
                    encoder_hidden_states=params.encoder_hidden_states,
                    attention_mask=params.attention_mask,
                    cross_attention_kwargs=params.cross_attention_kwargs,
                    encoder_attention_mask=params.encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, params.emb)
        # To support T2I-Adapter-XL
        if (
            params.is_adapter
            and len(params.down_intrablock_additional_residuals) > 0
            and sample.shape == params.down_intrablock_additional_residuals[0].shape
        ):
            sample += params.down_intrablock_additional_residuals.pop(0)

        if params.is_controlnet:
            sample = sample + params.mid_block_additional_residual
        params.sample = sample
        return params

    def forward_up(self, params: LayeringUNet2DCParams):
        sample = params.sample
        down_block_res_samples = params.down_block_res_samples  # 这里加一个 copy 应该会更合适一些的
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            upsample_size = (
                down_block_res_samples[-1].shape[2:] if not is_final_block and params.forward_upsample_size else None
            )
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=params.emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=params.encoder_hidden_states,
                    cross_attention_kwargs=params.cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=params.attention_mask,
                    encoder_attention_mask=params.encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=params.emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, params.lora_scale)
        params.sample, params.down_block_res_samples = sample, params.down_block_res_samples
        return params  # 为了和前面一系列的函数的返回值同一, 这里还是选择返回 `params` (虽然这是最后一层)
