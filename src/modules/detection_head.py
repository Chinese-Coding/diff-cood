import torch
from omegaconf import DictConfig
from torch import nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, anchor_num: int, dir_args: DictConfig):
        super().__init__()
        # TODO: 把这个输入通道数改成从配置文件中加载
        self.cls_head = nn.Conv2d(in_channels, anchor_num, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, 7 * anchor_num, kernel_size=1)
        self.dir_head = nn.Conv2d(in_channels, dir_args["num_bins"] * anchor_num, kernel_size=1)

    def forward(self, x: torch.Tensor):
        cls_pred = self.cls_head(x)
        reg_pred = self.reg_head(x)
        dir_pred = self.dir_head(x)
        return cls_pred, reg_pred, dir_pred
