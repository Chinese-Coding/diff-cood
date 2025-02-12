# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Template for AnchorGenerator
"""

import math
from typing import Dict

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F

from data_related.entity import ObjectBbxData
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.utils.common_utils import limit_period


class DiffPostProcessor:
    """`diff`: diff-cood 项目专用"""

    def __init__(self, postprocess_args, train=True):
        self.order = postprocess_args.order
        self.max_num = postprocess_args.max_num
        self.cav_lidar_range = postprocess_args.cav_lidar_range
        self.gt_range = postprocess_args.gt_range
        self.anchor_args = postprocess_args.anchor_args
        self.target_args = postprocess_args.target_args
        self.dir_args = postprocess_args.dir_args

        self.nms_thresh = postprocess_args.nms_thresh
        self.anchor_num = self.anchor_args.num
        self.train = train

        # 数据处理
        range = self.cav_lidar_range  # 这个变量纯粹是为了下面代码能写短一点
        vh, vw = postprocess_args.ratio, postprocess_args.ratio
        self.anchor_args.vw, self.anchor_args.vh = vh, vw
        self.anchor_args.W, self.anchor_args.H = math.ceil((range[3] - range[0]) / vw), math.ceil((range[4] - range[1]) / vh)

    def generate_gt_bbx(self, object_bbx_data: ObjectBbxData, transformation_matrix=np.eye(4)):
        """
        TODO: 证明一下这里使用单位矩阵的正确性
        :param transformation_matrix: 两辆不同车之间的转换矩阵, 因为没有协同, 所以用单位矩阵就可以了吧, 对应的 shape: (4, 4)
        """
        object_bbx_center = object_bbx_data.object_bbx_center
        object_bbx_mask = object_bbx_data.object_bbx_mask
        object_ids = object_bbx_data.object_ids

        object_bbx_center = object_bbx_center[object_bbx_mask == 1]
        # convert center to corner
        object_bbx_corner: np.ndarray = box_utils.boxes_to_corners_3d(object_bbx_center, self.order)
        projected_object_bbx_corner = box_utils.project_box3d(object_bbx_corner, transformation_matrix)
        selected_indices = [object_ids.index(x) for x in set(object_ids)]
        gt_box = projected_object_bbx_corner[selected_indices]
        return box_utils.mask_boxes_outside_range_numpy(gt_box, self.gt_range, order=None)

    def generate_object_center_lidar(self, cav_data: Dict, ref_lidar_pose, enlarge_z=False):
        """使用 lidar 传感器时, 对应的 object center"""
        vehicles = cav_data.cav_info["vehicles"]
        return self._generate_object_center(vehicles, ref_lidar_pose, enlarge_z)

    def generate_object_center_camera(self, cav_data, ref_lidar_pose, enlarge_z=False):
        """获取使用 camera 传感器时, 对应的 object center"""
        vehicles = cav_data.cav_info["vehicles"]
        inf_filter_range = [-1e5, -1e5, -1e5, 1e5, 1e5, 1e5]
        visibility_map = np.asarray(cv2.cvtColor(cav_data.bev_img, cv2.COLOR_BGR2GRAY))
        # TODO: 注意这里使用的是 `lidar_pose` 而并非 `lidar_pose_clean` 也就是说这里暂时不考虑噪声的问题
        ego_lidar_pose = cav_data.cav_info.lidar_pose
        output_dict = box_utils.diff_project_world_objects(vehicles, ego_lidar_pose, inf_filter_range, self.order, enlarge_z)

        updated_vehicles = {
            k: v for k, v in vehicles.items() if k in output_dict and box_utils.box_is_visible(output_dict[k], visibility_map)
        }  # 选出那些在 BEV 中可见的物体 (object, vehicles)
        return self._generate_object_center(updated_vehicles, ref_lidar_pose, enlarge_z)

    def _generate_object_center(self, vehicles: Dict, ref_lidar_pose, enlarge_z=False):
        filter_range = self.cav_lidar_range if self.train else self.gt_range

        output_dict = box_utils.diff_project_world_objects(vehicles, ref_lidar_pose, filter_range, self.order, enlarge_z)

        object_np, mask = np.zeros((self.max_num, 7)), np.zeros(self.max_num)
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids

    def generate_anchor_boxes(self):
        """生成 anchor 框"""
        # 这两个参数配置文件里面没有需要自己计算 (写在 __init__ 里面了)
        W, H = self.anchor_args["W"], self.anchor_args["H"]
        vh, vw = self.anchor_args["vh"], self.anchor_args["vw"]  # voxel_size

        l, w, h, r = self.anchor_args.l, self.anchor_args.w, self.anchor_args.h, self.anchor_args.r

        assert self.anchor_num == len(r)
        r = [math.radians(ele) for ele in r]

        xrange = [self.cav_lidar_range[0], self.cav_lidar_range[3]]
        yrange = [self.cav_lidar_range[1], self.cav_lidar_range[4]]

        feature_stride = self.anchor_args.get("feature_stride", 2)

        # vw is not precise, vw * feature_stride / 2 should be better?
        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // feature_stride)

        cx, cy = np.meshgrid(x, y)
        cx = np.tile(cx[..., np.newaxis], self.anchor_num)  # center
        cy = np.tile(cy[..., np.newaxis], self.anchor_num)
        cz = np.ones_like(cx) * -1.0

        w, l, h, r_ = np.ones_like(cx) * w, np.ones_like(cx) * l, np.ones_like(cx) * h, np.ones_like(cx)
        for i in range(self.anchor_num):
            r_[..., i] = r[i]

        match self.order:
            case "hwl":
                anchors = np.stack([cx, cy, cz, h, w, l, r_], axis=-1)  # (50, 176, 2, 7)
            case "lhw":
                anchors = np.stack([cx, cy, cz, l, h, w, r_], axis=-1)
            case _:
                raise ValueError(f"Unknown bbx order: {self.order}")

        return anchors

    def generate_label(self, object_bbx_center, anchor_boxes, object_bbx_mask, return_dict=True):
        """
        生成标签

        :param anchor_boxes: shape: (H, W, anchor_num, 7)
        """
        assert self.order == "hwl", "Currently Voxel only supporthwl bbx order."

        feature_map_shape = anchor_boxes.shape[:2]  # (H, W)

        anchor_boxes = einops.rearrange(anchor_boxes, "H W anchor_num x -> (H W anchor_num) x", x=7)
        # normalization factor, (H * W * anchor_num)
        anchors_d = np.sqrt(anchor_boxes[:, 4] ** 2 + anchor_boxes[:, 5] ** 2)

        # (H, W, 2)
        pos_equal_one, neg_equal_one = np.zeros((*feature_map_shape, self.anchor_num)), np.zeros(
            (*feature_map_shape, self.anchor_num)
        )

        # (H, W, self.anchor_num * 7)
        targets = np.zeros((*feature_map_shape, self.anchor_num * 7))

        gt_box_center_valid = object_bbx_center[object_bbx_mask == 1]  # (n, 7)
        # shape: (n, 8, 3), (H * W * anchor_num, 8, 3)
        gt_box_corner_valid, anchors_corner = box_utils.boxes_to_corners_3d(
            gt_box_center_valid, self.order
        ), box_utils.boxes_to_corners_3d(anchor_boxes, order=self.order)
        anchors_standup_2d = box_utils.corner2d_to_standup_box(anchors_corner)  # (H * W * anchor_num, 4)
        gt_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid)  # (n, 4)

        # (H * W * anchor_n)
        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        """
        就像一个二维表格
            gt1  gt2  gt3
        an1 0.1  0.2  0.3
        an2 0.4  0.5  0.6
        an3 0.7  0.8  0.9
        """
        # the anchor boxes has the largest iou across
        # shape: (n)
        id_highest = np.argmax(iou.T, axis=1)  # 找出每个 gt 框对应的最大I oU 的 anchor
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = np.where(iou > self.target_args["pos_threshold"])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou < self.target_args["neg_threshold"], axis=1) == iou.shape[1])[0]
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        # fmt: off
        targets[index_x, index_y, np.array(index_z) * 7] = (object_bbx_center[id_pos_gt, 0] - anchor_boxes[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = (object_bbx_center[id_pos_gt, 1] - anchor_boxes[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = (object_bbx_center[id_pos_gt, 2] - anchor_boxes[id_pos, 2]) / anchor_boxes[id_pos, 3]
        # fmt: on
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(object_bbx_center[id_pos_gt, 3] / anchor_boxes[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(object_bbx_center[id_pos_gt, 4] / anchor_boxes[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(object_bbx_center[id_pos_gt, 5] / anchor_boxes[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = object_bbx_center[id_pos_gt, 6] - anchor_boxes[id_pos, 6]

        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, self.anchor_num))
        neg_equal_one[index_x, index_y, index_z] = 0
        if return_dict:
            return {"pos_equal_one": pos_equal_one, "neg_equal_one": neg_equal_one, "targets": targets}
        else:
            return pos_equal_one, neg_equal_one, targets

    def postprocess(self, anchor_boxes, cls_pred, reg_pred, dir_pred, transformation_matrix=np.eye(4)):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS
        """

        # the final bounding box list
        # 因为整个场景里面只有一辆车, 所以这两个变量应该是多余的, 但是为了方便抄, 所以还是保留了这两个变量 (后面也用到了)
        pred_box3d_list, pred_box2d_list = [], []

        prob = F.sigmoid(cls_pred.permute(0, 2, 3, 1))
        prob = prob.reshape(1, -1)

        batch_box3d = self.delta_to_boxes3d(reg_pred, anchor_boxes) if len(reg_pred.shape) == 4 else reg_pred.view(1, -1, 7)

        mask = torch.gt(prob, self.target_args.score_threshold)
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

        assert batch_box3d.shape[0] == 1  # 在验证/测试的时候, batch size 应该为 1
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])

        """计算 dir_pred 有关的东西"""
        dir_offset = self.dir_args.dir_offset
        num_bins = self.dir_args.num_bins

        dm = dir_pred.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
        dir_cls_preds = dm[mask]
        # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
        # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
        period = 2 * np.pi / num_bins  # pi
        dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)  # 限制在0到pi之间
        boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)  # 转化0.25pi到2.5pi
        boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)  # limit to [-pi, pi]

        if len(boxes3d) != 0:
            # (N, 8, 3)
            boxes3d_corner = box_utils.boxes_to_corners_3d(boxes3d, order=self.order)

            # STEP 2
            # (N, 8, 3)
            projected_boxes3d = box_utils.project_box3d(boxes3d_corner, transformation_matrix)
            # convert 3d bbx to 2d, (N,4)
            projected_boxes2d = box_utils.corner_to_standup_box_torch(projected_boxes3d)
            # (N, 5)
            boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

            pred_box2d_list.append(boxes2d_score)
            pred_box3d_list.append(projected_boxes3d)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        # STEP3
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor, scores, self.nms_thresh)

        pred_box3d_tensor = pred_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # filter out the prediction out of the range. with z-dim
        pred_box3d_np = pred_box3d_tensor.cpu().numpy()
        pred_box3d_np, mask = box_utils.mask_boxes_outside_range_numpy(
            pred_box3d_np, self.gt_range, order=None, return_mask=True
        )
        pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
        scores = scores[mask]

        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        return pred_box3d_tensor, scores

    @staticmethod
    def delta_to_boxes3d(deltas, anchors):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, 14, H, W)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[..., [3]]) + anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d
