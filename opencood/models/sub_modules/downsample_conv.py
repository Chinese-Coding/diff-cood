import torch.nn as nn
from omegaconf import DictConfig


class DoubleConv(nn.Module):
    """
    Double convolution
    Args:
        in_channels: input channel num
        out_channels: output channel num
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownsampleConv(nn.Module):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.layers = nn.Sequential(*[
            DoubleConv(input_dim, dim, kernel_size=ksize, stride=stride, padding=padding)
            for input_dim, ksize, dim, stride, padding in zip(
                [args.input_dim] + args.output_dim[:-1], args.kernal_size, args.output_dim, args.stride, args.padding
            )
        ])

    def forward(self, x):
        return self.layers(x)
