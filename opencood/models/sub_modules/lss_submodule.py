# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision.models.resnet import resnet18, resnet101

from opencood.utils.camera_utils import bin_depths


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(  # 两个3x3卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 对x1进行上采样
        x1 = torch.cat([x2, x1], dim=1)  # 将x1和x2 concat 在一起
        return self.conv(x1)


class CamEncode(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, D, C, downsample, ddiscr, mode, use_gt_depth=False, depth_supervision=True):
        super(CamEncode, self).__init__()
        self.D = D  # 42
        self.C = C  # 64
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision  # in the case of not use gt depth

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征

        self.up1 = Up(320 + 112, 512)  # 上采样模块，输入输出通道分别为320+112和512
        if downsample == 8:
            self.up2 = Up(512 + 40, 512)
        if not use_gt_depth:
            self.depth_head = nn.Conv2d(512, self.D, kernel_size=1, padding=0)  # 1x1卷积，变换维度

        self.image_head = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-5):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return F.softmax(x, dim=1)

    def get_gt_depth_dist(self, x):  # 对深度维进行onehot，得到每个像素不同深度的概率
        """
        Args:
            x: [B*N, H, W]
        Returns:
            x: [B*N, D, fH, fW]
        """
        target = self.training
        torch.clamp_max_(x, self.d_max)  # save memory
        # [B*N, H, W], indices (float), value: [0, num_bins)
        depth_indices, mask = bin_depths(x, self.mode, self.d_min, self.d_max, self.num_bins, target=target)
        depth_indices = depth_indices[:, self.downsample // 2 :: self.downsample, self.downsample // 2 :: self.downsample]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0, 3, 1, 2)  # [B*N, num_bins, fH, fW]

        if not target:
            mask = mask[:, self.downsample // 2 :: self.downsample, self.downsample // 2 :: self.downsample].unsqueeze(1)
            onehot_dist *= mask

        return onehot_dist, depth_indices

    def get_eff_features(self, x):  # 使用efficientnet提取特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231

        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x  # x: 24 x 320 x 4 x 11
        x = self.up1(
            endpoints["reduction_5"], endpoints["reduction_4"]
        )  # 先对endpoints[4]进行上采样，然后将 endpoints[5]和endpoints[4] concat 在一起
        if self.downsample == 8:
            x = self.up2(x, endpoints["reduction_3"])
        return x  # x: 24 x 512 x 8 x 22

    def forward(self, x):
        """
        Returns:
            log_depth : [B*N, D, fH, fW], or None if not used latter
            depth_gt_indices : [B*N, fH, fW], or None if not used latter
            new_x : [B*N, C, D, fH, fW]
        """
        x_img_ = x[:, :3:, :, :]
        features = self.get_eff_features(
            x_img_
        )  # depth: B*N x D x fH x fW(24 x 41 x 8 x 22)  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        x_img = self.image_head(features)

        if self.depth_supervision or self.use_gt_depth:  # depth data must exist
            x_depth = x[:, 3, :, :]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)

        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img.unsqueeze(2)  # new_x: 24 x 64 x 41 x 8 x 18
            return None, new_x
        else:
            depth_logit = self.depth_head(features)
            depth = self.get_depth_dist(depth_logit)
            new_x = depth.unsqueeze(1) * x_img.unsqueeze(2)  # new_x: 24 x 64 x 41 x 8 x 18
            if self.depth_supervision:
                return (depth_logit, depth_gt_indices), new_x
            else:
                return None, new_x


class CamEncode_Resnet101(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, D, C, downsample, ddiscr, mode, use_gt_depth=False, depth_supervision=True):
        super(CamEncode_Resnet101, self).__init__()
        self.D = D  # 42
        self.C = C  # 64
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision  # in the case of not use gt depth

        trunk = resnet101(pretrained=False, zero_init_residual=True)  # 使用 resnet101 提取特征
        # in fact we only use first two layers. Equal to resnet50!
        self.conv1 = trunk.conv1
        self.bn1 = trunk.bn1
        self.relu = nn.ReLU()
        self.maxpool = trunk.maxpool
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = nn.Identity()

        if not use_gt_depth:
            self.depth_head = nn.Conv2d(512, self.D, kernel_size=1, padding=0)  # 1x1卷积，变换维度

        self.image_head = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-5):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return F.softmax(x, dim=1)

    def get_gt_depth_dist(self, x):  # 对深度维进行onehot，得到每个像素不同深度的概率
        """
        Args:
            x: [B*N, H, W]
        Returns:
            x: [B*N, D, fH, fW]
        """
        target = self.training
        torch.clamp_max_(x, self.d_max)  # save memory
        # [B*N, H, W], indices (float), value: [0, num_bins)
        depth_indices, mask = bin_depths(x, self.mode, self.d_min, self.d_max, self.num_bins, target=target)
        depth_indices = depth_indices[:, self.downsample // 2 :: self.downsample, self.downsample // 2 :: self.downsample]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0, 3, 1, 2)  # [B*N, num_bins, fH, fW]

        if not target:
            mask = mask[:, self.downsample // 2 :: self.downsample, self.downsample // 2 :: self.downsample].unsqueeze(1)
            onehot_dist *= mask

        return onehot_dist, depth_indices

    def resnet101_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def get_resnet_features(self, x):  # 使用resnet101提取特征
        # x: 16 x 3 x 480 x 640
        return self.resnet101_forward(x)

    def forward(self, x):
        """
        Returns:
            log_depth : [B*N, D, fH, fW], or None if not used latter
            depth_gt_indices : [B*N, fH, fW], or None if not used latter
            new_x : [B*N, C, D, fH, fW]
        """
        # x: 16 x 3 x 480 x 640
        # print(x.shape)
        x_img = x[:, :3, :, :].clone()
        features = self.get_resnet_features(
            x_img
        )  # depth: B*N x D x fH x fW(24 x 41 x 8 x 22)  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        x_img_feature = self.image_head(features)

        if self.depth_supervision or self.use_gt_depth:  # depth data must exist
            x_depth = x[:, 3, :, :]
            depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)

        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img_feature.unsqueeze(2)  # new_x: 24 x 64 x 41 x 8 x 18
            return None, new_x
        else:
            depth_logit = self.depth_head(features)
            depth = self.get_depth_dist(depth_logit)
            new_x = depth.unsqueeze(1) * x_img_feature.unsqueeze(2)  # new_x: 24 x 64 x 41 x 8 x 18
            if self.depth_supervision:
                return (depth_logit, depth_gt_indices), new_x
            else:
                return None, new_x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):  # inC: 64  outC: not 1 for object detection
        super(BevEncode, self).__init__()

        # 使用resnet的前3个stage作为backbone
        trunk = resnet18(zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(  # 2倍上采样->3x3卷积->1x1卷积
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):  # x: 4 x 64 x 240 x 240
        x = self.conv1(x)  # x: 4 x 64 x 120 x 120
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: 4 x 64 x 120 x 120
        x = self.layer2(x1)  # x: 4 x 128 x 60 x 60
        x = self.layer3(x)  # x: 4 x 256 x 30 x 30

        x = self.up1(x, x1)  # 给x进行4倍上采样然后和x1 concat 在一起  x: 4 x 256 x 120 x 120
        x = self.up2(x)  # 2倍上采样->3x3卷积->1x1卷积  x: 4 x 1 x 240 x 240

        return x
