from typing import Optional

import numpy as np


class Converter:
    def __init__(self):
        pass

    @staticmethod
    def to_harmonic(input: np.ndarray):
        """转换成齐次坐标形式"""
        M = input.shape[0]
        input = np.concatenate([input, np.ones([M, 1])], axis=1)
        return input

    @staticmethod
    def proj_3to2(xyz: np.ndarray, extrinsic, intrinsic):
        """
        将 3D 点云投影到 2D 平面上 (注释以及代码中 M 指的是点的个数)
        :param xyz: shape: [M, 3]
        :param extrinsic: shape: [4, 4]
        :param intrinsic: shape: [3, 3]
        """
        # 使用静态方法的调用
        xyz = Converter.to_harmonic(xyz)  # xyz's shape: [M, 4]
        xyz = np.linalg.inv(extrinsic) @ xyz.T  # xyz's shape: [4, M]
        uvd = intrinsic @ xyz[0:3]  # uvd's shape: [3, M]
        uvd = uvd.T  # uvd's shape: [3, M]
        # 将 u 和 v 坐标除以深度值 (dpt) 这样做的目的是进行归一化, 将相机坐标系中的 3D 点投影到 2D 图像平面上
        # 近大远小的道理, 所以需要除以深度信息
        uv, dpt = uvd[:, 0:2] / (uvd[:, -1:] + 1e-5), uvd[:, -1]  # uv's shape: [M, 2], dpt's shape: [M,]
        return uv, dpt

    @staticmethod
    def proj_pc2dpt(pc: np.ndarray, extrinsic, intrinsic, h, w, count: Optional[list[float]] = None):
        """
        将点云投影为深度图 (注释以及代码中 M 指的是点的个数)
        :param pc: shape: [M, 3]
        """
        uv, dpt = Converter.proj_3to2(pc, extrinsic, intrinsic)  # uv's shape: [M, 2], dpt's shape: [M,]
        # 根据 h, w (两者一般相等) 以及点的深度信息来过滤点 四个 mask 的 shape 均为 [M,]
        mask_h, mask_w, mask_d = (uv[:, 1] < h) & (uv[:, 1] >= 0), (uv[:, 0] < w) & (uv[:, 0] >= 0), dpt > 0.05
        mask = mask_h & mask_w & mask_d
        uv, dpt = uv[mask].astype(np.int32), dpt[mask]  # uv's shape: [M', 2], dpt's shape: [M',], M' 为投影后剩余的点

        # 统计一下还有多少个点剩余下来了
        if count is not None:
            count.append(uv.shape[0] / pc.shape[0]) # fmt: skip

        result = np.ones([h, w]) * 10000
        for i in range(uv.shape[0]):
            (u, v), d = uv[i], dpt[i]
            result[v, u] = min(result[v, u], d)
        result[result > 9999] = 0.0
        return result

    @staticmethod
    def proj_pc2dpt_debug(pc: np.ndarray, extrinsic, intrinsic, h, w):
        """
        将点云投影为深度图 的 debug 版本 (注释以及代码中 M 指的是点的个数)
        :param pc: shape: [M, 3]
        """
        from loguru import logger

        uv, dpt = Converter.proj_3to2(pc, extrinsic, intrinsic)  # uv's shape: [M, 2], dpt's shape: [M,]
        # 根据 h, w (两者一般相等) 以及点的深度信息来过滤点 四个 mask 的 shape 均为 [M,]
        # 经过调试后发现, 里面有很多的负数 TODO: 如果这里我简单的修改一下取值范围是否会更好一些呢? 例如把 uv 的范围改成 [- resolution / 2, resolution / 2]
        # mask_h, mask_w, mask_d = (uv[:, 1] < h) & (uv[:, 1] >= 0), (uv[:, 0] < w) & (uv[:, 0] >= 0), dpt > 0.05
        mask_h, mask_w, mask_d = (
            (uv[:, 1] <= h / 2) & (uv[:, 1] >= -h / 2),
            (uv[:, 0] <= w / 2) & (uv[:, 0] >= -w / 2),
            dpt > 0.05,
        )

        resolution_uv, resolution_dpt = uv[mask_h & mask_w], dpt[mask_h & mask_w]
        logger.success(f"只经过 resolution 过滤后剩余: {(resolution_uv.shape[0] / pc.shape[0]) * 100}%")
        dpt_uv, dpt_dpt = uv[mask_d], uv[mask_d]
        logger.success(f"只经过 depth 过滤后剩余: {(dpt_uv.shape[0] / pc.shape[0]) * 100}%")

        mask = mask_h & mask_w & mask_d
        uv, dpt = uv[mask].astype(np.int32), dpt[mask]  # uv's shape: [M', 2], dpt's shape: [M',], M' 为投影后剩余的点
        logger.success(f"经过 resolution 和 depth 的双重过滤: {(uv.shape[0] / pc.shape[0]) * 100}%")
        result = np.ones([h, w]) * 10000
        for i in range(uv.shape[0]):
            (u, v), d = uv[i], dpt[i]
            result[v, u] = min(result[v, u], d)
        result[result > 9999] = 0.0
        return result
