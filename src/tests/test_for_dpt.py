import multiprocessing

import numpy as np
from loguru import logger

from data_related.converter import Converter
from data_related.stable_diffusion_dataset import StableDiffusionDataset


def process_data_chunk(dataset, start, end, resolution):
    count = []  # 每个进程内部的计数器
    for i in range(start, end):
        cav_data = dataset[i]
        Converter.proj_pc2dpt(
            cav_data.lidar_np, extrinsic=np.eye(4), intrinsic=np.eye(3), h=resolution, w=resolution, count=count
        )
    mean = np.mean(count)
    logger.success(f"从 {start} 到 {end} 中有 {len(count)} 个深度图, 平均有 {mean * 100}% 被投影到深度图")
    return mean


def parallel_process(dataset, resolution, num_processes=None):
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
            results = pool.starmap(process_data_chunk, args_list)
            logger.success(f"遍历完成. 平均有 {np.mean(results)*100}% 被投影到深度图上")


if __name__ == "__main__":
    resolution = 512
    train_dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    train_dataset.reinitialize()

    # 获取进程数
    parallel_process(train_dataset, resolution)
"""
测试结果:  遍历完成. 平均有 0.0155770775946507 被投影到深度图上
投影得到的点数太少, 需要修改投影方式
"""
