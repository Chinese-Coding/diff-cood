import os

from icecream import ic
from loguru import logger
from omegaconf import OmegaConf

from data_related.entity import ObjectBbxData
from data_related.stable_diffusion_dataset import StableDiffusionDataset
from data_related.transform_funs import pcd_transform
from opencood.data_utils.post_processor.diff_base_post_processor import DiffPostProcessor


def _load_lidar_splitted_test():
    """
    测试是否每个数据下面都有分割之后的点云数据
    简单的遍历一下, 如果没有数据, 会有: `[Open3D WARNING]`
    """
    import multiprocess

    dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    dataset.reinitialize()

    dataset_length = len(dataset)
    num_processes = multiprocess.cpu_count()
    chunk_size = dataset_length // num_processes

    args_list = []
    for start in range(0, dataset_length, chunk_size):
        end = start + chunk_size if start + chunk_size < dataset_length else dataset_length
        args_list.append((start, end))

    def _loop(start, end):
        for i in range(start, end):
            data = dataset[i]

    # 使用进程池进行并行处理
    with multiprocess.Pool(processes=num_processes) as pool:
        logger.success(f"开始遍历数据集, 使用 {num_processes} 个线 (进) 程")
        pool.starmap(_loop, args_list)


def _box_related_test():
    """测试一切和 box 有关的方法或函数"""
    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/config.yaml"))
    dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    dataset.reinitialize()

    cav_lidar_range, ratio = args.cav_lidar_range, args.ratio
    post_processor = DiffPostProcessor(args.postprocess_args)
    transform = pcd_transform(cav_lidar_range, ratio)
    for item in dataset:
        object_np, mask, object_ids = post_processor.generate_object_center_lidar(item, item.cav_info["lidar_pose"])
        object_bbx_data = ObjectBbxData(object_bbx_center=object_np, object_bbx_mask=mask, object_ids=object_ids)
        gt_box = post_processor.generate_gt_bbx(object_bbx_data)
        anchor_box = post_processor.generate_anchor_boxes()
        bev_maps = transform(item.lidar_splitted)

        label = post_processor.generate_label(object_np, anchor_box, mask)
        for k, v in label.items():
            logger.success(f"{k=}")
            logger.success(f"{v.shape=}")
        logger.success(f"{ratio=} 下的 bev_maps 的 shape {bev_maps[0].shape}")
        logger.success(f"共生成了 {gt_box.shape[0]} 个 gt box")
        logger.success(f"{anchor_box.shape=}")
        break


if __name__ == "__main__":
    # _load_lidar_splitted_test()
    _box_related_test()
