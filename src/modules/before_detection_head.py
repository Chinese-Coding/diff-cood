import torch
from omegaconf import DictConfig
from torch import nn

from opencood.models.sub_modules.convnext import ConvNeXt
from opencood.models.sub_modules.downsample_conv import DownsampleConv


class BeforeDetectionHead(nn.Module):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=args.upsample_factor, mode="bilinear", align_corners=False)
        self.convnext = ConvNeXt(args.convnext.dim, args.convnext.num_of_blocks)
        self.shrink_header = DownsampleConv(args.shrink_header)

    def forward(self, x: torch.Tensor):
        x = self.upsample(x)
        x = self.convnext(x)
        x = self.shrink_header(x)
        return x
