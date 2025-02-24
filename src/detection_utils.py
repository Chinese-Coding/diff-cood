import os

import numpy as np
import torch
import torch.nn as nn
from opencood.utils import yaml_utils

from modules.detection_head import DetectionHead
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.utils import common_utils
from opencood.utils.eval_utils import voc_ap
from src.modules.before_detection_head import BeforeDetectionHead


def init_detection_modules(args):
    before_detection_head = BeforeDetectionHead(args.before_detection_head_args)
    detection_head = DetectionHead(
        args.detection_head_args.in_channels, args.postprocess_args.anchor_args.num, args.postprocess_args.dir_args
    )
    #  TODO: 这个上采用层用来把提取到的特征上采样到适用于 anchor 的分辨率
    loss_fn = PointPillarLoss(args.loss_args)
    # 优化器参数从 HEAL 中的某个配置文件抄过来的, TODO: 应该写成超参数的形式

    optimizer = torch.optim.Adam(
        list(before_detection_head.parameters()) + list(detection_head.parameters()),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    return detection_head, before_detection_head, loss_fn, optimizer, lr_scheduler


def load_detection_modules(resume_file_det, before_detection_head, detection_head, optimizer=None, lr_scheduler=None):
    checkpoint = torch.load(os.path.expanduser(resume_file_det), weights_only=False)
    before_detection_head.load_state_dict(checkpoint["before_detection_head"])
    detection_head.load_state_dict(checkpoint["detection_head"])
    if optimizer is not None and lr_scheduler is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    elif optimizer is None and lr_scheduler is None:
        pass
    else:
        raise ValueError("必须同时指定 `optimizer`, `lr_scheduler`.")
    return checkpoint


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    inference 的时候会用到
    Calculate the true positive and false positive numbers of the current
    frames.
    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]  # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box, in confidence descending order
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        result_stat[iou_thresh]["score"] += det_score.tolist()
    result_stat[iou_thresh]["fp"] += fp
    result_stat[iou_thresh]["tp"] += tp
    result_stat[iou_thresh]["gt"] += gt


def calculate_ap(result_stat, iou):
    """
    推理的时候会用到
    Calculate the average precision and recall, and save them into a txt.
    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = np.array(iou_5["fp"])
    tp = np.array(iou_5["tp"])
    score = np.array(iou_5["score"])
    assert len(fp) == len(tp) and len(tp) == len(score)

    sorted_index = np.argsort(-score)
    fp = fp[sorted_index].tolist()
    tp = tp[sorted_index].tolist()

    gt_total = iou_5["gt"]

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, infer_info=None):
    """推理的时候会用到"""
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({
        "ap30": ap_30,
        "ap_50": ap_50,
        "ap_70": ap_70,
        "mpre_50": mpre_50,
        "mrec_50": mrec_50,
        "mpre_70": mpre_70,
        "mrec_70": mrec_70,
    })
    if infer_info is None:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, "eval.yaml"))
    else:
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, f"eval_{infer_info}.yaml"))

    print(
        "The Average Precision at IOU 0.3 is %.2f, "
        "The Average Precision at IOU 0.5 is %.2f, "
        "The Average Precision at IOU 0.7 is %.2f" % (ap_30, ap_50, ap_70)
    )

    return ap_30, ap_50, ap_70
