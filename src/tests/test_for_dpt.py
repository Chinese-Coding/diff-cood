import multiprocessing

import numpy as np
from loguru import logger

from data_related.converter import Converter
from data_related.entity import CAVData
from data_related.stable_diffusion_dataset import StableDiffusionDataset


def _process_data_chunk(dataset, start, end, resolution):
    count = []  # 每个进程内部的计数器
    for i in range(start, end):
        cav_data = dataset[i]
        Converter.proj_pc2dpt(
            cav_data.lidar_np, extrinsic=np.eye(4), intrinsic=np.eye(3), h=resolution, w=resolution, count=count
        )
    mean = np.mean(count)
    logger.success(f"从 {start} 到 {end} 中有 {len(count)} 个深度图, 平均有 {mean * 100}% 被投影到深度图")
    return mean


def _test_for_mean_proj_point_num(dataset, resolution, num_processes=None):
    # 获取数据集的大小
    dataset_length = len(dataset)
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    chunk_size = dataset_length // num_processes
    # 使用 Manager 来共享数据（创建共享计数器）
    with multiprocessing.Manager():
        # 使用进程池进行并行处理
        with multiprocessing.Pool(processes=num_processes) as pool:
            logger.success("开始遍历数据集")
            args_list = []
            for start in range(0, dataset_length, chunk_size):
                end = start + chunk_size if start + chunk_size < dataset_length else dataset_length
                args_list.append((dataset, start, end, resolution))
            # 使用 starmap 分配工作到每个进程
            results = pool.starmap(_process_data_chunk, args_list)
            logger.success(f"遍历完成. 平均有 {np.mean(results)*100}% 被投影到深度图上")


if __name__ == "__main__":
    resolution = 512
    train_dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    train_dataset.reinitialize()

    # _test_for_mean_proj_point_num(train_dataset, resolution)
    cav_data: CAVData = train_dataset[0]
    extrinsic, intrinsic = np.eye(4), np.eye(3)  # np.ones((4, 4)), np.ones((3, 3))
    lidar_np = cav_data.lidar_np
    z = lidar_np[:, 2]
    count = np.sum((z < -0.05) | (z > 0.05))
    logger.success(f"z 坐标在区间 [-0.05, 0.05] 之外的点的个数为: {count}")
    dpt = Converter.proj_pc2dpt_debug(cav_data.lidar_np, extrinsic=extrinsic, intrinsic=intrinsic, h=resolution, w=resolution)
"""
# _test_for_mean_proj_point_num
测试结果:  遍历完成. 平均有 0.0155770775946507 被投影到深度图上
投影得到的点数太少, 需要修改投影方式

# 分别测试 resolution 和 depth 过滤剩余点的数量
2024-12-30 10:47:19.935 | SUCCESS  | data_related.converter:proj_pc2dpt_debug:69 - 只经过 resolution 过滤后剩余: 24.9612538093165%
2024-12-30 10:47:19.935 | SUCCESS  | data_related.converter:proj_pc2dpt_debug:71 - 只经过 depth 过滤后剩余: 6.293426208097519%
2024-12-30 10:47:19.935 | SUCCESS  | data_related.converter:proj_pc2dpt_debug:75 - 经过 resolution 和 depth 的双重过滤: 1.4140182847191989%

修改过滤范围之后 (仅修改 x, y 两个范围上的坐标):
2024-12-30 11:25:54.210 | SUCCESS  | data_related.converter:proj_pc2dpt_debug:77 - 只经过 resolution 过滤后剩余: 98.63996517196342%
2024-12-30 11:25:54.211 | SUCCESS  | data_related.converter:proj_pc2dpt_debug:79 - 只经过 depth 过滤后剩余: 6.293426208097519%
2024-12-30 11:25:54.211 | SUCCESS  | data_related.converter:proj_pc2dpt_debug:83 - 经过 resolution 和 depth 的双重过滤: 6.293426208097519%

z 坐标在区间 [-0.05, 0.05] 之外的点的个数为: 56956

找一个合理的相机的内参和外参应该就能很好地解决这个问题
"""
