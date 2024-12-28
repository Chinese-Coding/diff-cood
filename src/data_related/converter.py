import numpy as np


class Converter:
    def __init__(self):
        pass

    @staticmethod
    def to_harmonic(self, input: np.ndarray):
        """转换成齐次坐标形式"""
        M = input.shape[0]
        input = np.concatenate([input, np.ones([M, 1])], axis=1)
        return input

    @staticmethod
    def proj_3to2(self, xyz: np.ndarray, intrinsic, extrinsic):
        """
        不懂原理, 直接照抄
        :param xyz: shape: [M, 3]
        :param extrinsic: shape: [3, 3]
        :param intrinsic: shape: [4, 4]
        """
        xyz = self.to_harmonic(xyz)
        xyz = np.linalg.inv(extrinsic) @ xyz.T
        uvd = intrinsic @ xyz[0:3]
        uvd = uvd.T
        uv, d = uvd[:, 0:2] / (uvd[:, -1:] + 1e-5), uvd[:, -1]
        return uv, d

    @staticmethod
    def proj_pc2dpt(self, point_cloud: np.ndarray, extrinsic, intrinsic, h, w):
        uv, dpt = self.proj_3to2(point_cloud, intrinsic, extrinsic)
        mask_w = (uv[:, 0] < w) & (uv[:, 0] >= 0)
        mask_h = (uv[:, 1] < h) & (uv[:, 1] >= 0)
        # mask mask off the back-project points
        mask_d = dpt > 0.05
        mask = mask_h & mask_w & mask_d
        uv = uv[mask].astype(np.int32)
        dpt = dpt[mask]
        result = np.ones([h, w]) * 10000
        for i in range(uv.shape[0]):
            u, v = uv[i]
            d = dpt[i]
            result[v, u] = min(result[v, u], d)
        result[result > 9999] = 0.0
        return result
