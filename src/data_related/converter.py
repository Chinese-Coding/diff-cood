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
        :param extrinsic: shape: [3, 3]
        :param intrinsic: shape: [4, 4]
        """
        # 使用静态方法的调用
        xyz = Converter.to_harmonic(xyz)  # xyz's shape: [M, 4]
        xyz = np.linalg.inv(extrinsic) @ xyz.T  # xyz's shape: [4, M]
        uvd = intrinsic @ xyz[0:3]  # uvd's shape: [3, M]
        uvd = uvd.T  # uvd's shape: [3, M]
        uv, dpt = uvd[:, 0:2] / (uvd[:, -1:] + 1e-5), uvd[:, -1]  # uv's shape: [M, 2], dpt's shape: [M,]
        return uv, dpt

    @staticmethod
    def proj_pc2dpt(pc: np.ndarray, extrinsic, intrinsic, h, w, count=Optional[list[float]]):
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
