from icecream import ic

from data_related.entity import ObjectBbxData
from data_related.stable_diffusion_dataset import StableDiffusionDataset
from opencood.data_utils.post_processor.diff_base_post_processor import DiffBasePostProcessor
from loguru import logger


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
        """真不明白为什么一定要弄成一个函数, 不能写到 `_load_lidar_splitted_test` 里面"""
        for i in range(start, end):
            data = dataset[i]

    # 使用进程池进行并行处理
    with multiprocess.Pool(processes=num_processes) as pool:
        logger.success(f"开始遍历数据集, 使用 {num_processes} 个线 (进) 程")
        pool.starmap(_loop, args_list)


def _box_related_test():
    """测试一切和 box 有关的方法或函数"""
    dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    dataset.reinitialize()
    item = dataset[0]
    ic(item.cav_info)  # 打印效果不是很好, 通过 debug 看又看不到. 很烦

    # 测试框生成 (参数都是从 HEAL 里面抄来的)
    cav_lidar_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]
    order = "lwh"  # hwl or lwh
    max_num = 150
    post_processor = DiffBasePostProcessor(order, max_num, cav_lidar_range, cav_lidar_range)
    for item in dataset:
        object_np, mask, object_ids = post_processor.generate_object_center_lidar(item, item.cav_info["lidar_pose"])
        object_bbx_data = ObjectBbxData(object_bbx_center=object_np, object_bbx_mask=mask, object_ids=object_ids)
        gt_box = post_processor.generate_gt_bbx(object_bbx_data)
        logger.success(f"共生成了 {gt_box.shape[0]} 个 gt box")


if __name__ == "__main__":
    _load_lidar_splitted_test()
    # _box_related_test()
