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
from torch import Tensor

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
        self.feature_stride = self.anchor_args.get("feature_stride", 2)

    def generate_gt_bbx_heal(self, data_dict):
        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space
            # object_bbx_center is clean.
            transformation_matrix = cav_content["transformation_matrix_clean"]
            # transformation_matrix = cav_content.get("transformation_matrix_clean", torch.from_numpy(np.identity(4)).float())
            object_bbx_center = cav_content["object_bbx_center"]
            object_bbx_mask = cav_content["object_bbx_mask"]
            object_ids = cav_content["object_ids"]
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # convert center to corner
            object_bbx_corner = box_utils.boxes_to_corners_3d(object_bbx_center, self.order)
            projected_object_bbx_corner = box_utils.project_box3d(object_bbx_corner.float(), transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)
            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the id list to filter
        gt_box3d_selected_indices = [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range. with z dim
        gt_box3d_np = gt_box3d_tensor.cpu().numpy()
        gt_box3d_np = box_utils.mask_boxes_outside_range_numpy(gt_box3d_np, self.gt_range, order=None)
        gt_box3d_tensor = torch.from_numpy(gt_box3d_np).to(device=gt_box3d_list.device)

        return gt_box3d_tensor

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
        if isinstance(cav_data, dict):
            vehicles = cav_data.cav_info["vehicles"]
        elif isinstance(cav_data, list):
            assert len(cav_data) == 1
            vehicles = cav_data[0]["params"]["vehicles"]
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

        # vw is not precise, vw * feature_stride / 2 should be better?
        x = np.linspace(xrange[0] + vw, xrange[1] - vw, W // self.feature_stride)
        y = np.linspace(yrange[0] + vh, yrange[1] - vh, H // self.feature_stride)
        """等比例变换一下"""
        # l, w = l / self.feature_stride, w / self.feature_stride

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

        """等比例缩放一下"""
        # object_bbx_center = object_bbx_center.copy()
        # object_bbx_center[:, 0] /= self.feature_stride  # x
        # object_bbx_center[:, 1] /= self.feature_stride  # y
        # object_bbx_center[:, 4] /= self.feature_stride  # w
        # object_bbx_center[:, 5] /= self.feature_stride  # l

        anchor_boxes = anchor_boxes.reshape(-1, 7)

        # 三个返回值, (H, W, 2), (H, W, 2), (H, W, self.anchor_num * 7)
        pos_equal_one, neg_equal_one, targets = (
            np.zeros((*feature_map_shape, self.anchor_num)),
            np.zeros((*feature_map_shape, self.anchor_num)),
            np.zeros((*feature_map_shape, self.anchor_num * 7)),
        )

        gt_box_center_valid = object_bbx_center[object_bbx_mask == 1]  # (n, 7)
        # shape: (n, 8, 3), (H * W * anchor_num, 8, 3)
        gt_box_corner_valid, anchors_corner = box_utils.boxes_to_corners_3d(gt_box_center_valid, self.order), box_utils.boxes_to_corners_3d(anchor_boxes, order=self.order) # fmt: skip
        # shape: (n, 4), (H * W * anchor_num, 4)
        gt_standup_2d, anchors_standup_2d = box_utils.corner2d_to_standup_box(gt_box_corner_valid), box_utils.corner2d_to_standup_box(anchors_corner) # fmt: skip

        # (H * W * anchor_num, n (gt_standup_2d 的 shape[0]))
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
        id_highest = np.argmax(iou.T, axis=1)  # 找出每个 gt 框对应的最大 IoU 的 anchor
        # [0, 1, 2, ..., n-1]
        id_highest_gt = np.arange(iou.T.shape[0])
        # make sure all highest iou is larger than 0
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # find anchors iou > params['pos_iou']
        id_pos, id_pos_gt = np.where(iou > self.target_args["pos_threshold"])
        #  find anchors iou  params['neg_iou']
        id_neg = np.where(np.sum(iou < self.target_args["neg_threshold"], axis=1) == iou.shape[1])[0]
        # `id_pos` 和 `id_highest` 之间是一个有交集 (也可能没有) 的关系, 有交集的部分会通过下面的 `np.unique` 去重
        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # cal the target and set the equal one
        # numpy.unravel_index 是一个用于将一个平坦的索引转换为多维数组的索引的函数. 它的作用是将一个一维的索引 (即平坦数组中的位置) 转换成指定形状的多维数组中的对应位置.
        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, self.anchor_num))
        pos_equal_one[index_x, index_y, index_z] = 1

        # calculate the targets
        anchors_d = np.sqrt(anchor_boxes[:, 4] ** 2 + anchor_boxes[:, 5] ** 2)  # normalization factor, (H * W * anchor_num)
        # fmt: off
        targets[index_x, index_y, np.array(index_z) * 7] = (object_bbx_center[id_pos_gt, 0] - anchor_boxes[id_pos, 0]) / anchors_d[id_pos] # x 的偏移量
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = (object_bbx_center[id_pos_gt, 1] - anchor_boxes[id_pos, 1]) / anchors_d[id_pos] # y 的偏移量
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = (object_bbx_center[id_pos_gt, 2] - anchor_boxes[id_pos, 2]) / anchor_boxes[id_pos, 3] # z 的偏移量
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

    def postprocess(self, anchor_boxes: Tensor, cls_pred, reg_pred, dir_pred, transformation_matrix=np.eye(4)):
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
        """等比例变换一下"""
        # anchor_boxes = anchor_boxes.clone()
        # anchor_boxes[:, :, :, 4] *= self.feature_stride
        # anchor_boxes[:, :, :, 5] *= self.feature_stride
        # reg_pred = einops.rearrange(reg_pred, "b anchor_num h w -> b h w anchor_num")
        # reg_pred[:, :, :, 4] *= self.feature_stride
        # reg_pred[:, :, :, 5] *= self.feature_stride
        # reg_pred[:, :, :, 4 + 7] *= self.feature_stride
        # reg_pred[:, :, :, 5 + 7] *= self.feature_stride
        # reg_pred = einops.rearrange(reg_pred, "b anchor_num h w -> b h w anchor_num")

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
        N = deltas.shape[0]  # batch size
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        boxes3d = torch.zeros_like(deltas)

        anchors = anchors.to(device=deltas.device, dtype=deltas.dtype)
        boxes3d = boxes3d.to(device=deltas.device, dtype=deltas.dtype)

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).to(deltas.dtype)
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

    @staticmethod
    def collate_batch(label_batch_list):
        """
        为适配 HEAL 的数据接口新添加的方法
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one, neg_equal_one, targets = [], [], []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]["pos_equal_one"])
            neg_equal_one.append(label_batch_list[i]["neg_equal_one"])
            targets.append(label_batch_list[i]["targets"])

        pos_equal_one = torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = torch.from_numpy(np.array(neg_equal_one))
        targets = torch.from_numpy(np.array(targets))

        return {
            "targets": targets,
            "pos_equal_one": pos_equal_one,
            "neg_equal_one": neg_equal_one,
        }

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.


        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        for cav_id in output_dict.keys():
            assert cav_id in data_dict
            cav_content = data_dict[cav_id]
            # the transformation matrix to ego space
            # transformation_matrix = cav_content["transformation_matrix"]  # no clean
            transformation_matrix = cav_content.get("transformation_matrix", torch.from_numpy(np.identity(4)).float())
            # rename variable
            if "psm" in output_dict[cav_id]:
                output_dict[cav_id]["cls_preds"] = output_dict[cav_id]["psm"]
            if "rm" in output_dict:
                output_dict[cav_id]["reg_preds"] = output_dict[cav_id]["rm"]
            if "dm" in output_dict:
                output_dict[cav_id]["dir_preds"] = output_dict[cav_id]["dm"]

            # (H, W, anchor_num, 7)
            anchor_box = cav_content["anchor_box"]

            # classification probability
            prob = output_dict[cav_id]["cls_preds"]
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            prob = prob.reshape(1, -1)

            # regression map
            reg = output_dict[cav_id]["reg_preds"]

            # convert regression map back to bounding box
            if len(reg.shape) == 4:  # anchor-based. PointPillars, SECOND
                batch_box3d = self.delta_to_boxes3d(reg, anchor_box)
            else:  # anchor-free. CenterPoint
                batch_box3d = reg.view(1, -1, 7)

            mask = torch.gt(prob, self.target_args.score_threshold)
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob[0], mask[0])

            # adding dir classifier
            if "dir_preds" in output_dict[cav_id].keys() and len(boxes3d) != 0:
                dir_offset = self.dir_args["dir_offset"]
                num_bins = self.dir_args["num_bins"]

                dm = output_dict[cav_id]["dir_preds"]  # [N, H, W, 4]
                dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)  # [1, N*H*W*2, 2]
                dir_cls_preds = dir_cls_preds[mask]
                # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
                dir_labels = torch.max(dir_cls_preds, dim=-1)[
                    1
                ]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0

                period = 2 * np.pi / num_bins  # pi
                dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)  # 限制在0到pi之间
                boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)  # 转化0.25pi到2.5pi
                boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)  # limit to [-pi, pi]

            if "iou_preds" in output_dict[cav_id].keys() and len(boxes3d) != 0:
                iou = torch.sigmoid(output_dict[cav_id]["iou_preds"].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
                iou = torch.clamp(iou, min=0.0, max=1.0)
                iou = (iou + 1) * 0.5
                scores = scores * torch.pow(iou.masked_select(mask), 4)

            # convert output to bounding box
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
