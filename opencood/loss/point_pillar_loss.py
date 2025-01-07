# -*- coding: utf-8 -*-
# Author: Yifan Lu
# Add direction classification loss
# The originally point_pillar_loss.py, can not determine if the box heading is opposite to the GT.

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from opencood.utils.common_utils import limit_period


def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)
    loss_ftor = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def _weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma**2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + (abs_diff - 0.5 / (sigma**2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        loss *= weights
    return loss


def _sigmoid_focal_loss(preds, targets, weights=None, **kwargs):
    assert "gamma" in kwargs and "alpha" in kwargs
    # sigmoid cross entropy with logits
    # more details: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    per_entry_cross_ent = torch.clamp(preds, min=0) - preds * targets.type_as(preds)
    per_entry_cross_ent += torch.log1p(torch.exp(-torch.abs(preds)))
    # focal loss
    prediction_probabilities = torch.sigmoid(preds)
    p_t = (targets * prediction_probabilities) + ((1 - targets) * (1 - prediction_probabilities))
    modulating_factor = torch.pow(1.0 - p_t, kwargs["gamma"])
    alpha_weight_factor = targets * kwargs["alpha"] + (1 - targets) * (1 - kwargs["alpha"])

    loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
    if weights is not None:
        loss *= weights
    return loss


def _one_hot_f(tensor, num_bins, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(*list(tensor.shape), num_bins, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


class PointPillarLoss(nn.Module):
    def __init__(self, loss_args: DictConfig):
        super(PointPillarLoss, self).__init__()
        self.pos_cls_weight = loss_args.pos_cls_weight

        self.cls_args = loss_args.cls_args
        self.reg_args = loss_args.reg_args
        self.dir_args = loss_args.get("dir_args", None)

        self.loss_dict = {}

    def forward(self, cls_pred, reg_pred, dir_pred, pos_equal_one, neg_equal_one, targets):
        batch_size = cls_pred.shape[0]

        cls_labels = pos_equal_one.view(batch_size, -1, 1)
        positives = cls_labels > 0
        negatives = neg_equal_one.view(batch_size, -1, 1) > 0
        pos_normalizer = positives.sum(1, keepdim=True).float()

        total_loss = 0

        """cls loss"""
        cls_preds = cls_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        cls_weights = positives * self.pos_cls_weight + negatives * 1.0
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss = _sigmoid_focal_loss(cls_preds, cls_labels, weights=cls_weights, **self.cls_args)
        cls_loss = cls_loss.sum() * self.cls_args.weight / batch_size

        """reg loss"""
        reg_weights = positives / torch.clamp(pos_normalizer, min=1.0)
        reg_preds = reg_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        reg_targets = targets.view(batch_size, -1, 7)
        reg_preds, reg_targets = self.add_sin_difference(reg_preds, reg_targets)
        reg_loss = _weighted_smooth_l1_loss(reg_preds, reg_targets, weights=reg_weights, sigma=self.reg_args.sigma)
        reg_loss = reg_loss.sum() * self.reg_args.weight / batch_size

        """dir loss"""
        if self.dir_args is not None:
            dir_targets = self.get_direction_target(targets.view(batch_size, -1, 7))
            dir_preds = dir_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            dir_loss = _softmax_cross_entropy_with_logits(dir_preds, dir_targets)
            dir_loss = dir_loss.flatten() * reg_weights.flatten()
            dir_loss = dir_loss.sum() * self.dir_args.weight / batch_size
            total_loss += dir_loss
            self.loss_dict.update({"dir_loss": dir_loss.item()})

        total_loss += reg_loss + cls_loss
        self.loss_dict.update({"total_loss": total_loss.item(), "reg_loss": reg_loss.item(), "cls_loss": cls_loss.item()})
        return total_loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(boxes2[..., dim : dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(boxes2[..., dim : dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1)
        return boxes1, boxes2

    def get_direction_target(self, reg_targets):
        """
        Args:
        reg_targets:  [N, H * W * #anchor_num, 7]
        The last term is (theta_gt - theta_a)

        Returns:
        dir_targets:
        theta_gt: [N, H * W * #anchor_num, NUM_BIN]
        NUM_BIN = 2
        """
        num_bins = self.dir_args["args"]["num_bins"]
        dir_offset = self.dir_args["args"]["dir_offset"]
        anchor_yaw = np.deg2rad(np.array(self.dir_args["args"]["anchor_yaw"]))  # for direction classification
        self.anchor_yaw_map = torch.from_numpy(anchor_yaw).view(1, -1, 1)  # [1,2,1]
        self.anchor_num = self.anchor_yaw_map.shape[1]

        H_times_W_times_anchor_num = reg_targets.shape[1]
        anchor_map = self.anchor_yaw_map.repeat(1, H_times_W_times_anchor_num // self.anchor_num, 1).to(
            reg_targets.device
        )  # [1, H * W * #anchor_num, 1]
        rot_gt = reg_targets[..., -1] + anchor_map[..., -1]  # [N, H*W*anchornum]
        offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()  # [N, H*W*anchornum]
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        # one_hot:
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        dir_cls_targets = _one_hot_f(dir_cls_targets, num_bins)
        return dir_cls_targets

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get("total_loss", 0)
        reg_loss = self.loss_dict.get("reg_loss", 0)
        cls_loss = self.loss_dict.get("cls_loss", 0)
        dir_loss = self.loss_dict.get("dir_loss", 0)

        print(
            "[epoch %d][%d/%d] || Loss: %.4f || Conf Loss: %.4f || Loc Loss: %.4f || Dir Loss: %.4f"
            % (epoch, batch_id + 1, batch_len, total_loss, cls_loss, reg_loss, dir_loss)
        )

        if not writer is None:
            writer.add_scalar("Regression_loss", reg_loss, epoch * batch_len + batch_id)
            writer.add_scalar("Confidence_loss", cls_loss, epoch * batch_len + batch_id)
            writer.add_scalar("Dir_loss", dir_loss, epoch * batch_len + batch_id)
